#!/usr/bin/env python3
"""Train a lightweight RGB refinement head for O256 diagnostics.

This is a post-hoc diagnostic, not the main watermark model training.
It freezes the O256 checkpoint and learns a small residual CNN:

    refined = watermarked + delta(watermarked, cover)

The loss combines RGB fidelity to the cover image and message recoverability
through the frozen O256 decoder. Outputs are saved as images so the existing
PSNR/SSIM and BA evaluators can be reused.
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from models.ddm import DenoisingDiffusion, data_transform  # noqa: E402
from utils.metrics import calculate_psnr, calculate_ssim  # noqa: E402


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        config_path = PROJECT_ROOT / "configs" / str(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return dict2namespace(yaml.safe_load(f))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_to_tensor(path, image_size):
    image = Image.open(path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def tensor_to_uint8(tensor):
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    array = tensor.permute(1, 2, 0).numpy()
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def tensor_to_pil(tensor):
    return Image.fromarray(tensor_to_uint8(tensor))


def metric_pair(pred, ref):
    pred_uint8 = tensor_to_uint8(pred)
    ref_uint8 = tensor_to_uint8(ref)
    return (
        float(calculate_psnr(pred_uint8, ref_uint8, test_y_channel=False)),
        float(calculate_ssim(pred_uint8, ref_uint8, test_y_channel=False)),
    )


def gaussian_kernel2d(kernel_size, sigma, device, dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.view(1, 1, kernel_size, kernel_size)


def gaussian_blur_tensor(x, sigma):
    if sigma <= 0:
        return x
    kernel_size = max(3, int(round(sigma * 6)) | 1)
    kernel = gaussian_kernel2d(kernel_size, sigma, x.device, x.dtype)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=x.shape[1])


def laplacian_tensor(x):
    kernel = x.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])


def batch_cosine(a, b):
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    numerator = (a_flat * b_flat).sum(dim=1)
    denominator = torch.linalg.norm(a_flat, dim=1) * torch.linalg.norm(b_flat, dim=1)
    return numerator / denominator.clamp_min(1.0e-12)


def straight_through_quantize(x, levels):
    levels = max(2, int(levels))
    quantized = torch.round(x * (levels - 1)) / (levels - 1)
    return x + (quantized - x).detach()


def apply_proxy_attack(refined, cover, args):
    """Differentiable/lightweight attack proxies for message-aware refinement.

    These proxies are not the formal paper-matched attacks. They are used only
    during RGB-refiner training to discourage watermark signals that disappear
    after mild blur/noise/compression-like changes.
    """
    if args.attack_profile == "none" or args.attack_message_weight <= 0:
        return refined, "none"

    if args.attack_profile == "nongeom_light":
        attack_names = ("noise", "blur", "jpeg_like", "dropout")
    elif args.attack_profile == "nongeom_formal":
        attack_names = ("noise", "blur", "jpeg_like", "dropout", "salt_pepper")
    elif args.attack_profile == "targeted_formal_weak":
        # "Weak" refers to the weakest measured BA categories, not mild attack
        # strength. These four proxies match the native formal attack scale in
        # the refiner's [0, 1] RGB domain.
        attack_names = (
            "noise_arg03",
            "noise_sigma03",
            "dropout_r03",
            "salt_pepper_r02",
        )
    elif args.attack_profile == "dropout_r03_only":
        attack_names = ("dropout_r03",)
    elif args.attack_profile == "dropout_r03_native_only":
        attack_names = ("dropout_r03_native",)
    else:
        raise ValueError(f"Unsupported attack_profile: {args.attack_profile}")

    index = int(torch.randint(len(attack_names), (1,), device=refined.device).item())
    name = attack_names[index]

    if name == "noise":
        attacked = refined + torch.randn_like(refined) * args.attack_noise_sigma
    elif name == "noise_arg03":
        attacked = refined + torch.randn_like(refined) * args.attack_noise_sigma
    elif name == "noise_sigma03":
        attacked = refined + torch.randn_like(refined) * args.attack_noise_sigma_secondary
    elif name == "blur":
        attacked = gaussian_blur_tensor(refined, args.attack_blur_sigma)
    elif name == "jpeg_like":
        attacked = straight_through_quantize(refined, args.attack_jpeg_levels)
    elif name == "dropout":
        keep = (torch.rand_like(refined[:, :1]) > args.attack_dropout_prob).to(refined.dtype)
        attacked = refined * keep + cover * (1.0 - keep)
    elif name == "dropout_r03":
        # Native DiffW Dropout(0.3) keeps the watermarked value when r <= 0.3
        # and replaces it with cover otherwise. This legacy proxy is retained
        # for reproducibility of exp32-35, but its keep/drop direction is
        # intentionally not changed.
        keep = (torch.rand_like(refined) > args.attack_dropout_prob).to(refined.dtype)
        attacked = refined * keep + cover * (1.0 - keep)
    elif name == "dropout_r03_native":
        # Matches noise_layers.crop.Dropout(prob): torch.where(rdn > prob, cover, image).
        keep = (torch.rand_like(refined) <= args.attack_dropout_prob).to(refined.dtype)
        attacked = refined * keep + cover * (1.0 - keep)
    elif name == "salt_pepper":
        random_map = torch.rand_like(refined[:, :1])
        half_prob = args.attack_salt_pepper_prob / 2.0
        salt = random_map < half_prob
        pepper = random_map > (1.0 - half_prob)
        attacked = torch.where(salt, torch.ones_like(refined), refined)
        attacked = torch.where(pepper, torch.zeros_like(refined), attacked)
    elif name == "salt_pepper_r02":
        # Native DiffW SP samples independently for every channel.
        random_map = torch.rand_like(refined)
        half_prob = args.attack_salt_pepper_prob / 2.0
        salt = random_map < half_prob
        pepper = random_map > (1.0 - half_prob)
        attacked = torch.where(salt, torch.ones_like(refined), refined)
        attacked = torch.where(pepper, torch.zeros_like(refined), attacked)
    else:
        attacked = refined

    return attacked.clamp(0.0, 1.0), name


class RefinementPairDataset(Dataset):
    def __init__(self, watermarked_dir, gt_dir, image_size, max_images=0):
        self.watermarked_dir = Path(watermarked_dir)
        self.gt_dir = Path(gt_dir)
        self.image_size = image_size
        self.gt_by_stem = {
            path.stem: path
            for path in sorted(self.gt_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        }
        items = []
        for wm_path in sorted(self.watermarked_dir.iterdir()):
            if not wm_path.is_file() or wm_path.suffix.lower() not in IMAGE_EXTS:
                continue
            gt_path = self.gt_by_stem.get(wm_path.stem)
            msg_path = self.watermarked_dir / f"{wm_path.stem}_message.npy"
            if gt_path is None or not msg_path.exists():
                continue
            items.append((wm_path, gt_path, msg_path))
        if max_images and max_images > 0:
            items = items[:max_images]
        if not items:
            raise RuntimeError(
                f"No valid pairs found. watermarked_dir={self.watermarked_dir}, gt_dir={self.gt_dir}"
            )
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        wm_path, gt_path, msg_path = self.items[index]
        wm = pil_to_tensor(wm_path, self.image_size)
        gt = pil_to_tensor(gt_path, self.image_size)
        message = torch.from_numpy(np.load(msg_path).astype(np.float32)).view(-1)
        return {
            "wm": wm,
            "gt": gt,
            "message": message,
            "name": wm_path.name,
            "message_name": msg_path.name,
        }


class ResidualRefiner(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=48, depth=5, max_delta=0.25):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        layers = [
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)
        self.max_delta = float(max_delta)

    def forward(self, wm, gt=None):
        if gt is None:
            x = wm
        else:
            x = torch.cat([wm, gt], dim=1)
        delta = torch.tanh(self.net(x)) * self.max_delta
        return torch.clamp(wm + delta, 0.0, 1.0), delta


def parse_args():
    parser = argparse.ArgumentParser(description="Train RGB refinement head for O256")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--watermarked_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--resume_refiner",
        default="",
        help="Optional refiner checkpoint to initialize from for two-stage refinement.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_channels", type=int, default=48)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--max_delta", type=float, default=0.25)
    parser.add_argument("--conditioning", choices=("cover", "none"), default="cover")
    parser.add_argument("--image_mse_weight", type=float, default=1.0)
    parser.add_argument("--image_l1_weight", type=float, default=0.25)
    parser.add_argument("--message_weight", type=float, default=10.0)
    parser.add_argument(
        "--bit_weight_json",
        default="",
        help="Optional JSON list/dict of per-bit message-loss weights.",
    )
    parser.add_argument(
        "--attack_message_weight",
        type=float,
        default=0.0,
        help="Extra message loss after a lightweight differentiable attack proxy.",
    )
    parser.add_argument(
        "--attack_profile",
        choices=(
            "none",
            "nongeom_light",
            "nongeom_formal",
            "targeted_formal_weak",
            "dropout_r03_only",
            "dropout_r03_native_only",
        ),
        default="none",
        help="Training-only attack proxy profile. Formal attack evaluation is separate.",
    )
    parser.add_argument("--attack_noise_sigma", type=float, default=0.03)
    parser.add_argument("--attack_noise_sigma_secondary", type=float, default=0.15)
    parser.add_argument("--attack_blur_sigma", type=float, default=1.0)
    parser.add_argument("--attack_jpeg_levels", type=int, default=32)
    parser.add_argument("--attack_dropout_prob", type=float, default=0.05)
    parser.add_argument("--attack_salt_pepper_prob", type=float, default=0.05)
    parser.add_argument("--regularize_weight", type=float, default=0.05)
    parser.add_argument(
        "--latent_preserve_weight",
        type=float,
        default=0.0,
        help=(
            "Optional VAE-latent preservation loss: "
            "MSE(Enc(refined), Enc(watermarked)). Use this to improve message "
            "preservation without directly relaxing RGB fidelity."
        ),
    )
    parser.add_argument(
        "--sensitive_residual_weight",
        type=float,
        default=0.0,
        help=(
            "Optional message-sensitive residual penalty. Penalizes RGB residual "
            "where the clean message loss has high image-space gradient."
        ),
    )
    parser.add_argument(
        "--quality_gate_psnr",
        type=float,
        default=0.0,
        help=(
            "Optional PSNR gate in [0,1] RGB MSE space. When >0, adds a hinge "
            "penalty only if refined-vs-cover MSE is worse than this PSNR target."
        ),
    )
    parser.add_argument("--quality_gate_weight", type=float, default=0.0)
    parser.add_argument(
        "--residual_lap_preserve_weight",
        type=float,
        default=0.0,
        help=(
            "Optional guard loss that preserves high-frequency residual evidence: "
            "L1(Laplacian(refined-cover), Laplacian(watermarked-cover))."
        ),
    )
    parser.add_argument(
        "--anti_cancel_weight",
        type=float,
        default=0.0,
        help=(
            "Optional guard loss that discourages the refiner delta from aligning "
            "too strongly with -base_residual, which would cancel watermark residuals."
        ),
    )
    parser.add_argument(
        "--anti_cancel_margin",
        type=float,
        default=0.92,
        help="Cosine margin for anti-cancellation guard; penalize only cos(delta, -base_residual) above this value.",
    )
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--save_eval_checkpoints",
        action="store_true",
        help="Save refiner_stepXXXXXX.pth at each validation point.",
    )
    parser.add_argument(
        "--skip_final_dataset_images",
        action="store_true",
        help="Skip writing refined copies of the full refiner training dataset.",
    )
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def split_indices(length, val_fraction, seed):
    indices = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(length * val_fraction))
    val_count = max(1, min(length - 1, val_count)) if length > 1 else 0
    val_indices = sorted(indices[:val_count])
    train_indices = sorted(indices[val_count:])
    return train_indices, val_indices


def bit_accuracy(decoded, message):
    decoded_np = decoded.detach().cpu().numpy().round().clip(0, 1)
    message_np = message.detach().cpu().numpy()
    return float((1.0 - np.mean(np.abs(decoded_np - message_np))) * 100.0)


def load_bit_weights(path, device, expected_length):
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = payload.get("weights", payload)
    if isinstance(weights, dict):
        values = [float(weights[str(index)]) for index in range(expected_length)]
    else:
        values = [float(value) for value in weights]
    if len(values) != expected_length:
        raise ValueError(
            f"bit_weight_json length mismatch: got {len(values)}, expected {expected_length}"
        )
    tensor = torch.tensor(values, dtype=torch.float32, device=device).view(1, -1)
    return tensor / tensor.mean().clamp_min(1e-8)


def build_diffusion(config, resume, device):
    config.device = device
    dummy_args = argparse.Namespace(resume=resume)
    diffusion = DenoisingDiffusion(dummy_args, config)
    diffusion.load_ddm_ckpt(resume, ema=True)
    diffusion.model.eval()
    for param in diffusion.model.parameters():
        param.requires_grad = False
    if diffusion.vae is not None:
        diffusion.vae.eval()
        for param in diffusion.vae.parameters():
            param.requires_grad = False
    return diffusion


def refine_batch(refiner, batch, conditioning):
    wm = batch["wm"]
    gt = batch["gt"]
    cond = gt if conditioning == "cover" else None
    return refiner(wm, cond)


def get_message_decoder(diffusion):
    return diffusion.model.module.decoder if hasattr(diffusion.model, "module") else diffusion.model.decoder


def decoder_message_loss(diffusion, refined_0_1, message, bit_weights=None, return_latent=False):
    refined_latent = diffusion.encode_images(data_transform(refined_0_1), track_grad=True)
    decoded = get_message_decoder(diffusion)(refined_latent)
    loss = F.mse_loss(decoded, message, reduction="none")
    if bit_weights is not None:
        loss = loss * bit_weights
    loss = loss.mean()
    if return_latent:
        return loss, decoded, refined_latent
    return loss, decoded


@torch.no_grad()
def evaluate(refiner, diffusion, loader, device, conditioning, max_batches=0):
    refiner.eval()
    psnrs = []
    ssims = []
    bas = []
    l1_to_wm = []
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        refined, _delta = refine_batch(refiner, batch, conditioning)
        msg_loss, decoded = decoder_message_loss(diffusion, refined, batch["message"])
        del msg_loss
        for item in range(refined.shape[0]):
            psnr, ssim = metric_pair(refined[item], batch["gt"][item])
            psnrs.append(psnr)
            ssims.append(ssim)
        bas.append(bit_accuracy(decoded, batch["message"]))
        l1_to_wm.append(float(F.l1_loss(refined, batch["wm"]).detach().cpu()))
    refiner.train()
    return {
        "psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
        "ssim": float(np.mean(ssims)) if ssims else float("nan"),
        "ba": float(np.mean(bas)) if bas else float("nan"),
        "l1_to_wm": float(np.mean(l1_to_wm)) if l1_to_wm else float("nan"),
    }


@torch.no_grad()
def save_refined_images(refiner, dataset, diffusion, output_dir, device, conditioning, batch_size, num_workers):
    refined_dir = output_dir / "refined_images"
    clean_dir = output_dir / "attacked_clean" / "Clean"
    refined_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    refiner.eval()
    psnrs = []
    ssims = []
    bas = []
    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        refined, _delta = refine_batch(refiner, batch, conditioning)
        _msg_loss, decoded = decoder_message_loss(diffusion, refined, batch["message"])
        bas.append(bit_accuracy(decoded, batch["message"]))
        for item, name in enumerate(batch["name"]):
            image = tensor_to_pil(refined[item])
            image.save(refined_dir / name)
            image.save(clean_dir / name)
            msg_name = batch["message_name"][item]
            src_msg = dataset.watermarked_dir / msg_name
            if src_msg.exists():
                shutil.copy2(src_msg, refined_dir / msg_name)
                shutil.copy2(src_msg, clean_dir / msg_name)
            psnr, ssim = metric_pair(refined[item], batch["gt"][item])
            psnrs.append(psnr)
            ssims.append(ssim)
    return {
        "psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
        "ssim": float(np.mean(ssims)) if ssims else float("nan"),
        "ba": float(np.mean(bas)) if bas else float("nan"),
        "refined_dir": str(refined_dir),
        "clean_attack_dir": str(output_dir / "attacked_clean"),
    }


def save_reports(output_dir, args, history, split_info, final_metrics):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "split": split_info,
        "history": history,
        "final_metrics": final_metrics,
    }
    (output_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(output_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys() if history else ["step"])
        writer.writeheader()
        writer.writerows(history)
    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("RGB refinement diagnostic report\n")
        f.write("================================\n")
        f.write(f"watermarked_dir: {args.watermarked_dir}\n")
        f.write(f"gt_dir: {args.gt_dir}\n")
        f.write(f"config: {args.config}\n")
        f.write(f"resume: {args.resume}\n")
        f.write(f"conditioning: {args.conditioning}\n")
        f.write(f"steps: {args.steps}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"max_delta: {args.max_delta}\n")
        f.write(f"image_mse_weight: {args.image_mse_weight}\n")
        f.write(f"image_l1_weight: {args.image_l1_weight}\n")
        f.write(f"message_weight: {args.message_weight}\n")
        f.write(f"bit_weight_json: {args.bit_weight_json}\n")
        f.write(f"attack_message_weight: {args.attack_message_weight}\n")
        f.write(f"attack_profile: {args.attack_profile}\n")
        f.write(f"attack_noise_sigma: {args.attack_noise_sigma}\n")
        f.write(f"attack_noise_sigma_secondary: {args.attack_noise_sigma_secondary}\n")
        f.write(f"attack_blur_sigma: {args.attack_blur_sigma}\n")
        f.write(f"attack_jpeg_levels: {args.attack_jpeg_levels}\n")
        f.write(f"attack_dropout_prob: {args.attack_dropout_prob}\n")
        f.write(f"attack_salt_pepper_prob: {args.attack_salt_pepper_prob}\n")
        f.write(f"regularize_weight: {args.regularize_weight}\n")
        f.write(f"latent_preserve_weight: {args.latent_preserve_weight}\n")
        f.write(f"sensitive_residual_weight: {args.sensitive_residual_weight}\n")
        f.write(f"quality_gate_psnr: {args.quality_gate_psnr}\n")
        f.write(f"quality_gate_weight: {args.quality_gate_weight}\n")
        f.write(f"residual_lap_preserve_weight: {args.residual_lap_preserve_weight}\n")
        f.write(f"anti_cancel_weight: {args.anti_cancel_weight}\n")
        f.write(f"anti_cancel_margin: {args.anti_cancel_margin}\n")
        f.write(f"train_count: {split_info['train_count']}\n")
        f.write(f"val_count: {split_info['val_count']}\n")
        f.write("\nFinal metrics\n")
        for key, value in final_metrics.items():
            f.write(f"{key}: {value}\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    diffusion = build_diffusion(config, args.resume, device)
    dataset = RefinementPairDataset(args.watermarked_dir, args.gt_dir, args.image_size, args.max_images)
    sample_message_length = int(dataset[0]["message"].numel())
    bit_weights = load_bit_weights(args.bit_weight_json, device, sample_message_length)
    train_indices, val_indices = split_indices(len(dataset), args.val_fraction, args.seed)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices) if val_indices else train_set
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    in_channels = 6 if args.conditioning == "cover" else 3
    refiner = ResidualRefiner(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        depth=args.depth,
        max_delta=args.max_delta,
    ).to(device)
    if args.resume_refiner:
        checkpoint = torch.load(args.resume_refiner, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        refiner.load_state_dict(state_dict, strict=True)
        print(f"=> loaded refiner checkpoint: {args.resume_refiner}")
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=0.0)

    print(f"device: {device}")
    print(f"pairs: {len(dataset)}, train: {len(train_set)}, val: {len(val_set)}")
    print(f"output_dir: {output_dir}")
    print(f"conditioning: {args.conditioning}, max_delta: {args.max_delta}")
    if bit_weights is not None:
        print(f"bit_weight_json: {args.bit_weight_json}")

    history = []
    start = time.time()
    global_step = 0
    train_iter = iter(train_loader)
    while global_step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        global_step += 1
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        refined, _delta = refine_batch(refiner, batch, args.conditioning)
        mse_loss = F.mse_loss(refined, batch["gt"])
        l1_loss = F.l1_loss(refined, batch["gt"])
        reg_loss = F.l1_loss(refined, batch["wm"])
        needs_refined_latent = args.latent_preserve_weight > 0
        msg_result = decoder_message_loss(
            diffusion,
            refined,
            batch["message"],
            bit_weights,
            return_latent=needs_refined_latent,
        )
        if needs_refined_latent:
            msg_loss, decoded, refined_latent = msg_result
        else:
            msg_loss, decoded = msg_result
            refined_latent = None

        if args.latent_preserve_weight > 0:
            with torch.no_grad():
                wm_latent = diffusion.encode_images(data_transform(batch["wm"]), track_grad=False)
            latent_preserve_loss = F.mse_loss(refined_latent, wm_latent)
        else:
            latent_preserve_loss = refined.new_tensor(0.0)

        if args.sensitive_residual_weight > 0:
            msg_grad = torch.autograd.grad(
                msg_loss,
                refined,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if msg_grad is None:
                sensitive_residual_loss = refined.new_tensor(0.0)
            else:
                sensitive_residual_loss = (_delta * msg_grad.detach()).abs().mean()
        else:
            sensitive_residual_loss = refined.new_tensor(0.0)

        if args.quality_gate_weight > 0 and args.quality_gate_psnr > 0:
            target_mse = 1.0 / (10.0 ** (args.quality_gate_psnr / 10.0))
            quality_gate_loss = torch.relu(mse_loss - mse_loss.new_tensor(target_mse))
        else:
            quality_gate_loss = refined.new_tensor(0.0)

        if args.residual_lap_preserve_weight > 0:
            refined_residual_lap = laplacian_tensor(refined - batch["gt"])
            base_residual_lap = laplacian_tensor(batch["wm"] - batch["gt"]).detach()
            residual_lap_preserve_loss = F.l1_loss(refined_residual_lap, base_residual_lap)
        else:
            residual_lap_preserve_loss = refined.new_tensor(0.0)

        if args.anti_cancel_weight > 0:
            base_residual = (batch["wm"] - batch["gt"]).detach()
            cancel_cos = batch_cosine(_delta, -base_residual)
            anti_cancel_loss = torch.relu(cancel_cos - args.anti_cancel_margin).pow(2).mean()
        else:
            cancel_cos = refined.new_zeros(refined.shape[0])
            anti_cancel_loss = refined.new_tensor(0.0)

        if args.attack_message_weight > 0:
            attacked_refined, attack_name = apply_proxy_attack(refined, batch["gt"], args)
            attack_msg_loss, _attack_decoded = decoder_message_loss(
                diffusion, attacked_refined, batch["message"], bit_weights
            )
        else:
            attack_name = "none"
            attack_msg_loss = refined.new_tensor(0.0)
        loss = (
            args.image_mse_weight * mse_loss
            + args.image_l1_weight * l1_loss
            + args.message_weight * msg_loss
            + args.attack_message_weight * attack_msg_loss
            + args.regularize_weight * reg_loss
            + args.latent_preserve_weight * latent_preserve_loss
            + args.sensitive_residual_weight * sensitive_residual_loss
            + args.quality_gate_weight * quality_gate_loss
            + args.residual_lap_preserve_weight * residual_lap_preserve_loss
            + args.anti_cancel_weight * anti_cancel_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        row = {
            "step": global_step,
            "loss": float(loss.detach().cpu()),
            "mse_loss": float(mse_loss.detach().cpu()),
            "l1_loss": float(l1_loss.detach().cpu()),
            "message_loss": float(msg_loss.detach().cpu()),
            "attack_message_loss": float(attack_msg_loss.detach().cpu()),
            "attack_name": attack_name,
            "regularize_loss": float(reg_loss.detach().cpu()),
            "latent_preserve_loss": float(latent_preserve_loss.detach().cpu()),
            "sensitive_residual_loss": float(sensitive_residual_loss.detach().cpu()),
            "quality_gate_loss": float(quality_gate_loss.detach().cpu()),
            "residual_lap_preserve_loss": float(residual_lap_preserve_loss.detach().cpu()),
            "anti_cancel_loss": float(anti_cancel_loss.detach().cpu()),
            "cancel_cos_mean": float(cancel_cos.detach().mean().cpu()),
            "train_ba": bit_accuracy(decoded, batch["message"]),
            "val_psnr": "",
            "val_ssim": "",
            "val_ba": "",
            "val_l1_to_wm": "",
        }
        should_eval = (
            global_step == 1
            or global_step == args.steps
            or (args.eval_every > 0 and global_step % args.eval_every == 0)
        )
        if should_eval:
            metrics = evaluate(refiner, diffusion, val_loader, device, args.conditioning)
            row["val_psnr"] = f"{metrics['psnr']:.6f}"
            row["val_ssim"] = f"{metrics['ssim']:.6f}"
            row["val_ba"] = f"{metrics['ba']:.6f}"
            row["val_l1_to_wm"] = f"{metrics['l1_to_wm']:.6f}"
            print(
                f"step {global_step:06d}: loss={row['loss']:.6f}, "
                f"mse={row['mse_loss']:.6f}, l1={row['l1_loss']:.6f}, "
                f"msg={row['message_loss']:.6f}, atk_msg={row['attack_message_loss']:.6f}, "
                f"atk={row['attack_name']}, reg={row['regularize_loss']:.6f}, "
                f"lat={row['latent_preserve_loss']:.6f}, sens={row['sensitive_residual_loss']:.6f}, "
                f"qgate={row['quality_gate_loss']:.6f}, lap={row['residual_lap_preserve_loss']:.6f}, "
                f"ac={row['anti_cancel_loss']:.6f}, cos={row['cancel_cos_mean']:.4f}, "
                f"train_ba={row['train_ba']:.2f}, val_psnr={metrics['psnr']:.4f}, "
                f"val_ssim={metrics['ssim']:.4f}, val_ba={metrics['ba']:.2f}"
            )
            if args.save_eval_checkpoints:
                torch.save(
                    {
                        "state_dict": refiner.state_dict(),
                        "args": vars(args),
                        "step": global_step,
                        "validation": metrics,
                        "elapsed_seconds": time.time() - start,
                    },
                    output_dir / f"refiner_step{global_step:06d}.pth",
                )
        elif args.log_every > 0 and global_step % args.log_every == 0:
            print(
                f"step {global_step:06d}: loss={row['loss']:.6f}, "
                f"mse={row['mse_loss']:.6f}, l1={row['l1_loss']:.6f}, "
                f"msg={row['message_loss']:.6f}, atk_msg={row['attack_message_loss']:.6f}, "
                f"atk={row['attack_name']}, reg={row['regularize_loss']:.6f}, "
                f"lat={row['latent_preserve_loss']:.6f}, sens={row['sensitive_residual_loss']:.6f}, "
                f"qgate={row['quality_gate_loss']:.6f}, lap={row['residual_lap_preserve_loss']:.6f}, "
                f"ac={row['anti_cancel_loss']:.6f}, cos={row['cancel_cos_mean']:.4f}, "
                f"train_ba={row['train_ba']:.2f}"
            )
        history.append(row)

    torch.save(
        {
            "state_dict": refiner.state_dict(),
            "args": vars(args),
            "elapsed_seconds": time.time() - start,
        },
        output_dir / "refiner_final.pth",
    )
    if args.skip_final_dataset_images:
        final_eval = evaluate(refiner, diffusion, val_loader, device, args.conditioning)
        final_metrics = {
            "psnr": final_eval["psnr"],
            "ssim": final_eval["ssim"],
            "ba": final_eval["ba"],
            "l1_to_wm": final_eval["l1_to_wm"],
            "refined_dir": "skipped",
            "clean_attack_dir": "skipped",
        }
    else:
        final_metrics = save_refined_images(
            refiner,
            dataset,
            diffusion,
            output_dir,
            device,
            args.conditioning,
            args.batch_size,
            args.num_workers,
        )
    final_metrics["elapsed_seconds"] = time.time() - start
    split_info = {
        "total_count": len(dataset),
        "train_count": len(train_set),
        "val_count": len(val_set),
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    save_reports(output_dir, args, history, split_info, final_metrics)
    print(f"saved refiner: {output_dir / 'refiner_final.pth'}")
    print(f"report: {output_dir / 'report.txt'}")
    print(f"refined images: {final_metrics['refined_dir']}")
    print(f"clean attack dir: {final_metrics['clean_attack_dir']}")


if __name__ == "__main__":
    main()
