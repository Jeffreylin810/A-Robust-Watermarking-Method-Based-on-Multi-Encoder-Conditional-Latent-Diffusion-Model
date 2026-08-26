#!/usr/bin/env python3
"""Fine-tune VAE components for GT reconstruction.

This is a diagnostic experiment for the O256 PSNR bottleneck. It freezes the
VAE encoder by default, encodes GT images into latents, and trains the decode
side to reconstruct the original GT image:

    x -> frozen E(x) -> trainable D'(z) -> x_hat

Optional flags can also train post_quant_conv or the full VAE for upper-bound
diagnostics.

The output is saved with Diffusers ``save_pretrained`` so it can be screened by
``scripts/screen_vae_reconstruction.py --vae name=/path/to/final``.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from models.ddm import data_transform, inverse_data_transform  # noqa: E402
from utils.metrics import calculate_psnr, calculate_ssim  # noqa: E402

try:
    from diffusers import AutoencoderKL
except ImportError as exc:  # pragma: no cover - runtime dependency check
    AutoencoderKL = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_BASE_VAE = "/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base"
DEFAULT_TRAIN_DIR = (
    "/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/"
    "train10000_eval1000_unseen/data/CoCo/train/gt"
)
DEFAULT_VAL_DIR = (
    "/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/"
    "train10000_eval1000_unseen/data/CoCo/val/gt"
)


class ImageFolderDataset(Dataset):
    def __init__(self, root, image_size, max_images=0):
        self.root = Path(root)
        self.image_size = image_size
        self.paths = [
            path
            for path in sorted(self.root.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
        if max_images and max_images > 0:
            self.paths = self.paths[:max_images]
        if not self.paths:
            raise RuntimeError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor, path.name


def parse_args():
    parser = argparse.ArgumentParser(description="VAE reconstruction fine-tune")
    parser.add_argument("--base_vae", default=DEFAULT_BASE_VAE)
    parser.add_argument("--train_dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val_dir", default=DEFAULT_VAL_DIR)
    parser.add_argument(
        "--output_dir",
        default="results/vae_finetune/sd21_decoder_ft_coco256_10ksteps_seed61",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--l1_weight", type=float, default=0.25)
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--eval_images", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=0, help="0 means only save final.")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--train_post_quant_conv",
        action="store_true",
        help="Also train post_quant_conv. Kept off by default for stricter decoder-only diagnosis.",
    )
    parser.add_argument(
        "--train_full_vae",
        action="store_true",
        help=(
            "Train encoder, quant_conv, post_quant_conv, and decoder. "
            "This is an upper-bound diagnosis and changes the latent distribution."
        ),
    )
    parser.add_argument(
        "--disable_cudnn",
        action="store_true",
        help="Disable cuDNN if a CUDA convolution engine error appears.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_uint8_rgb(tensor):
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def metric_pair(pred_uint8, ref_uint8):
    return {
        "psnr": float(calculate_psnr(pred_uint8, ref_uint8, test_y_channel=False)),
        "ssim": float(calculate_ssim(pred_uint8, ref_uint8, test_y_channel=False)),
    }


def set_module_trainable(module, trainable):
    module.train(trainable)
    for param in module.parameters():
        param.requires_grad = trainable


def load_vae(path, device, train_post_quant_conv, train_full_vae):
    if AutoencoderKL is None:
        raise ImportError("diffusers is required for VAE decoder fine-tune") from IMPORT_ERROR
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"VAE path does not exist: {path}")
    if (path / "vae").is_dir():
        vae = AutoencoderKL.from_pretrained(path, subfolder="vae")
    else:
        vae = AutoencoderKL.from_pretrained(path)
    vae.to(device)

    for param in vae.parameters():
        param.requires_grad = False

    if train_full_vae:
        trainable_modules = []
        for name in ("encoder", "quant_conv", "post_quant_conv", "decoder"):
            if hasattr(vae, name):
                trainable_modules.append(getattr(vae, name))
        for module in trainable_modules:
            set_module_trainable(module, True)
    else:
        trainable_modules = [vae.decoder]
        if train_post_quant_conv and hasattr(vae, "post_quant_conv"):
            trainable_modules.append(vae.post_quant_conv)

        for module in trainable_modules:
            set_module_trainable(module, True)

        if hasattr(vae, "encoder"):
            vae.encoder.eval()
        if hasattr(vae, "quant_conv"):
            vae.quant_conv.eval()

    scaling = getattr(vae.config, "scaling_factor", 1.0)
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in vae.parameters())
    return vae, scaling, trainable_params, total_params


def reconstruct(vae, scaling_factor, rgb_0_1, train_full_vae):
    target = data_transform(rgb_0_1)
    if train_full_vae:
        posterior = vae.encode(target).latent_dist
        latent = posterior.mode() * scaling_factor
    else:
        with torch.no_grad():
            posterior = vae.encode(target).latent_dist
            latent = posterior.mode() * scaling_factor
    decoded = vae.decode(latent / scaling_factor).sample
    return decoded, target, posterior


@torch.no_grad()
def evaluate(vae, scaling_factor, val_loader, device, max_batches):
    was_training = vae.decoder.training
    vae.eval()
    rows = []
    for batch_index, (rgb_0_1, _names) in enumerate(val_loader):
        if max_batches and batch_index >= max_batches:
            break
        rgb_0_1 = rgb_0_1.to(device, non_blocking=True)
        target = data_transform(rgb_0_1)
        posterior = vae.encode(target).latent_dist
        latent = posterior.mode() * scaling_factor
        decoded = vae.decode(latent / scaling_factor).sample
        recon_0_1 = inverse_data_transform(decoded)

        for item in range(rgb_0_1.shape[0]):
            gt_uint8 = tensor_to_uint8_rgb(rgb_0_1[item : item + 1])
            recon_uint8 = tensor_to_uint8_rgb(recon_0_1[item : item + 1])
            rows.append(metric_pair(recon_uint8, gt_uint8))

    if was_training:
        vae.decoder.train()
    if not rows:
        return {"psnr": float("nan"), "ssim": float("nan")}
    return {
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ssim": float(np.mean([row["ssim"] for row in rows])),
    }


def save_report(output_dir, args, history, trainable_params, total_params, elapsed):
    output_dir = Path(output_dir)
    report = {
        "args": vars(args),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "elapsed_seconds": elapsed,
        "history": history,
        "final_model_dir": str(output_dir / "final"),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(output_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "mse_loss",
                "l1_loss",
                "kl_loss",
                "eval_psnr",
                "eval_ssim",
            ],
        )
        writer.writeheader()
        writer.writerows(history)
    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("Decoder-only VAE fine-tune report\n")
        f.write("=================================\n")
        f.write(f"base_vae: {args.base_vae}\n")
        f.write(f"train_dir: {args.train_dir}\n")
        f.write(f"val_dir: {args.val_dir}\n")
        f.write(f"image_size: {args.image_size}\n")
        f.write(f"steps: {args.steps}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"mse_weight: {args.mse_weight}\n")
        f.write(f"l1_weight: {args.l1_weight}\n")
        f.write(f"kl_weight: {args.kl_weight}\n")
        f.write(f"train_post_quant_conv: {args.train_post_quant_conv}\n")
        f.write(f"train_full_vae: {args.train_full_vae}\n")
        f.write(f"trainable_params: {trainable_params}\n")
        f.write(f"total_params: {total_params}\n")
        f.write(f"elapsed_seconds: {elapsed:.2f}\n")
        if history:
            last = history[-1]
            f.write("\nLast evaluation\n")
            f.write(f"step: {last['step']}\n")
            f.write(f"eval_psnr: {last['eval_psnr']}\n")
            f.write(f"eval_ssim: {last['eval_ssim']}\n")
        f.write(f"\nfinal_model_dir: {output_dir / 'final'}\n")


def main():
    args = parse_args()
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = ImageFolderDataset(args.train_dir, args.image_size, args.max_train_images)
    val_set = ImageFolderDataset(args.val_dir, args.image_size, args.eval_images)
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

    vae, scaling_factor, trainable_params, total_params = load_vae(
        args.base_vae, device, args.train_post_quant_conv, args.train_full_vae
    )
    optimizer = torch.optim.AdamW(
        [param for param in vae.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )

    print(f"device: {device}")
    print(f"train images: {len(train_set)}, val images: {len(val_set)}")
    print(f"latent scaling_factor: {scaling_factor}")
    print(f"trainable_params: {trainable_params} / total_params: {total_params}")
    print(f"output_dir: {output_dir}")

    history = []
    start = time.time()
    global_step = 0
    loader_iter = iter(train_loader)
    while global_step < args.steps:
        try:
            rgb_0_1, _names = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            rgb_0_1, _names = next(loader_iter)

        global_step += 1
        rgb_0_1 = rgb_0_1.to(device, non_blocking=True)
        decoded, target, posterior = reconstruct(vae, scaling_factor, rgb_0_1, args.train_full_vae)
        mse_loss = F.mse_loss(decoded, target)
        l1_loss = F.l1_loss(decoded, target)
        if args.kl_weight > 0:
            kl_loss = posterior.kl().mean()
        else:
            kl_loss = decoded.new_tensor(0.0)
        loss = args.mse_weight * mse_loss + args.l1_weight * l1_loss + args.kl_weight * kl_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        row = {
            "step": global_step,
            "loss": float(loss.detach().cpu()),
            "mse_loss": float(mse_loss.detach().cpu()),
            "l1_loss": float(l1_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "eval_psnr": "",
            "eval_ssim": "",
        }

        should_eval = (
            global_step == 1
            or global_step == args.steps
            or (args.eval_every > 0 and global_step % args.eval_every == 0)
        )
        if should_eval:
            metrics = evaluate(vae, scaling_factor, val_loader, device, max_batches=0)
            row["eval_psnr"] = f"{metrics['psnr']:.6f}"
            row["eval_ssim"] = f"{metrics['ssim']:.6f}"
            print(
                f"step {global_step:06d}: loss={row['loss']:.6f}, "
                f"mse={row['mse_loss']:.6f}, l1={row['l1_loss']:.6f}, "
                f"kl={row['kl_loss']:.6f}, "
                f"eval_psnr={metrics['psnr']:.4f}, eval_ssim={metrics['ssim']:.4f}"
            )
        elif args.log_every > 0 and global_step % args.log_every == 0:
            print(
                f"step {global_step:06d}: loss={row['loss']:.6f}, "
                f"mse={row['mse_loss']:.6f}, l1={row['l1_loss']:.6f}, "
                f"kl={row['kl_loss']:.6f}"
            )

        history.append(row)

        if args.save_every > 0 and global_step % args.save_every == 0:
            checkpoint_dir = output_dir / f"checkpoint_step_{global_step:06d}"
            vae.save_pretrained(checkpoint_dir)
            print(f"saved checkpoint: {checkpoint_dir}")

    final_dir = output_dir / "final"
    vae.save_pretrained(final_dir)
    elapsed = time.time() - start
    save_report(output_dir, args, history, trainable_params, total_params, elapsed)
    print(f"saved final VAE: {final_dir}")
    print(f"report: {output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
