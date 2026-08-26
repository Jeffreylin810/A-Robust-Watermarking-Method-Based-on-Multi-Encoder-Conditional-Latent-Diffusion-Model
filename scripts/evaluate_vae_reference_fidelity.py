#!/usr/bin/env python3
"""Evaluate fidelity against both original GT and VAE-reconstructed cover.

For latent watermarking, GT-vs-output PSNR can be dominated by the frozen VAE
round-trip. This script separates:

1. GT vs VAE reconstructed cover: the VAE ceiling.
2. GT vs watermarked output: the conventional image fidelity score.
3. VAE reconstructed cover vs watermarked output: watermark-specific delta
   after accounting for VAE reconstruction.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


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


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def parse_args():
    parser = argparse.ArgumentParser(description="VAE-reference image fidelity evaluator")
    parser.add_argument(
        "--config",
        default="CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml",
        help="Config basename under configs/, or an explicit yaml path.",
    )
    parser.add_argument(
        "--gt_dir",
        default=(
            "/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/"
            "train10000_eval1000_unseen/data/CoCo/val/gt"
        ),
        help="Directory containing original GT images.",
    )
    parser.add_argument(
        "--watermarked_dir",
        required=True,
        help="Directory containing generated/watermarked result images.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for report.txt, report.json, summary.csv, and optional images.",
    )
    parser.add_argument("--max_images", type=int, default=0, help="0 means all matched images.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--save_vae_cover",
        action="store_true",
        help="Save VAE reconstructed cover images to output_dir/vae_reconstructed_cover.",
    )
    return parser.parse_args()


def resolve_config_path(config_arg):
    candidate = Path(config_arg)
    if candidate.is_file():
        return candidate
    candidate = PROJECT_ROOT / "configs" / config_arg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Config not found: {config_arg}")


def load_config(config_arg):
    with open(resolve_config_path(config_arg), "r", encoding="utf-8") as f:
        return dict2namespace(yaml.safe_load(f))


def list_images_by_stem(directory):
    directory = Path(directory)
    images = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images[path.stem] = path
    return images


def load_rgb_tensor(path, image_size, device):
    image = Image.open(path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def tensor_to_uint8_rgb(tensor):
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def save_uint8(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def metric_pair(pred_uint8, ref_uint8):
    return {
        "psnr": float(calculate_psnr(pred_uint8, ref_uint8, test_y_channel=False)),
        "ssim": float(calculate_ssim(pred_uint8, ref_uint8, test_y_channel=False)),
    }


def average(rows, key):
    values = [row[key] for row in rows]
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float("inf")
    return float(np.mean(finite))


def build_vae(config, device):
    if AutoencoderKL is None:
        raise ImportError("diffusers is required for VAE evaluation") from IMPORT_ERROR

    if not getattr(config, "latent", None) or not getattr(config.latent, "enabled", False):
        raise ValueError("This evaluator expects latent.enabled=True in the config.")

    vae_path = config.latent.model_path
    if os.path.isdir(os.path.join(vae_path, "vae")):
        vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
    else:
        vae = AutoencoderKL.from_pretrained(vae_path)
    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    scaling = getattr(config.latent, "scaling_factor", None)
    if scaling is None:
        scaling = getattr(vae.config, "scaling_factor", 1.0)
    return vae, scaling


def vae_reconstruct(vae, scaling_factor, rgb_0_1):
    rgb_minus1_1 = data_transform(rgb_0_1)
    posterior = vae.encode(rgb_minus1_1).latent_dist
    latent = posterior.mode() * scaling_factor
    decoded = vae.decode(latent / scaling_factor).sample
    return inverse_data_transform(decoded)


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vae_cover_dir = output_dir / "vae_reconstructed_cover"
    if args.save_vae_cover:
        vae_cover_dir.mkdir(parents=True, exist_ok=True)

    vae, scaling_factor = build_vae(config, device)

    gt_images = list_images_by_stem(args.gt_dir)
    wm_images = list_images_by_stem(args.watermarked_dir)
    stems = sorted(set(gt_images) & set(wm_images))
    if args.max_images > 0:
        stems = stems[: args.max_images]
    if not stems:
        raise RuntimeError(f"No matched images between {args.gt_dir} and {args.watermarked_dir}")

    rows = []
    with torch.no_grad():
        for index, stem in enumerate(stems):
            gt_tensor = load_rgb_tensor(gt_images[stem], config.data.image_size, device)
            wm_tensor = load_rgb_tensor(wm_images[stem], config.data.image_size, device)
            vae_cover_tensor = vae_reconstruct(vae, scaling_factor, gt_tensor)

            gt_uint8 = tensor_to_uint8_rgb(gt_tensor)
            wm_uint8 = tensor_to_uint8_rgb(wm_tensor)
            vae_cover_uint8 = tensor_to_uint8_rgb(vae_cover_tensor)

            if args.save_vae_cover:
                save_uint8(vae_cover_dir / f"{stem}.png", vae_cover_uint8)

            gt_vs_vae = metric_pair(vae_cover_uint8, gt_uint8)
            gt_vs_wm = metric_pair(wm_uint8, gt_uint8)
            vae_vs_wm = metric_pair(wm_uint8, vae_cover_uint8)

            row = {
                "index": index,
                "stem": stem,
                "gt_vs_vae_cover_psnr": gt_vs_vae["psnr"],
                "gt_vs_vae_cover_ssim": gt_vs_vae["ssim"],
                "gt_vs_watermarked_psnr": gt_vs_wm["psnr"],
                "gt_vs_watermarked_ssim": gt_vs_wm["ssim"],
                "vae_cover_vs_watermarked_psnr": vae_vs_wm["psnr"],
                "vae_cover_vs_watermarked_ssim": vae_vs_wm["ssim"],
            }
            rows.append(row)
            print(
                "{:04d} {}: GT-VAE PSNR={:.4f}, GT-WM PSNR={:.4f}, VAE-WM PSNR={:.4f}".format(
                    index + 1,
                    stem,
                    row["gt_vs_vae_cover_psnr"],
                    row["gt_vs_watermarked_psnr"],
                    row["vae_cover_vs_watermarked_psnr"],
                )
            )

    averages = {
        "gt_vs_vae_cover_psnr": average(rows, "gt_vs_vae_cover_psnr"),
        "gt_vs_vae_cover_ssim": average(rows, "gt_vs_vae_cover_ssim"),
        "gt_vs_watermarked_psnr": average(rows, "gt_vs_watermarked_psnr"),
        "gt_vs_watermarked_ssim": average(rows, "gt_vs_watermarked_ssim"),
        "vae_cover_vs_watermarked_psnr": average(rows, "vae_cover_vs_watermarked_psnr"),
        "vae_cover_vs_watermarked_ssim": average(rows, "vae_cover_vs_watermarked_ssim"),
    }
    summary = {
        "config": str(resolve_config_path(args.config)),
        "gt_dir": str(Path(args.gt_dir)),
        "watermarked_dir": str(Path(args.watermarked_dir)),
        "output_dir": str(output_dir),
        "num_images": len(rows),
        "device": str(device),
        "latent_model_path": config.latent.model_path,
        "latent_scaling_factor": scaling_factor,
        "averages": averages,
    }

    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": rows}, f, indent=2)

    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("VAE-reference fidelity report\n")
        f.write("=" * 40 + "\n")
        f.write(f"config: {summary['config']}\n")
        f.write(f"gt_dir: {summary['gt_dir']}\n")
        f.write(f"watermarked_dir: {summary['watermarked_dir']}\n")
        f.write(f"processed: {summary['num_images']}\n")
        f.write(f"latent_model_path: {summary['latent_model_path']}\n")
        f.write(f"latent_scaling_factor: {summary['latent_scaling_factor']}\n\n")
        f.write("Average metrics\n")
        f.write(f"{'comparison':<30} | {'PSNR':>9} | {'SSIM':>9}\n")
        f.write("-" * 55 + "\n")
        f.write(
            f"{'GT vs VAE cover':<30} | {averages['gt_vs_vae_cover_psnr']:>9.4f} | "
            f"{averages['gt_vs_vae_cover_ssim']:>9.4f}\n"
        )
        f.write(
            f"{'GT vs watermarked':<30} | {averages['gt_vs_watermarked_psnr']:>9.4f} | "
            f"{averages['gt_vs_watermarked_ssim']:>9.4f}\n"
        )
        f.write(
            f"{'VAE cover vs watermarked':<30} | {averages['vae_cover_vs_watermarked_psnr']:>9.4f} | "
            f"{averages['vae_cover_vs_watermarked_ssim']:>9.4f}\n"
        )
        f.write("\nInterpretation guide:\n")
        f.write("- GT vs VAE cover is the frozen VAE reconstruction ceiling.\n")
        f.write("- GT vs watermarked is the conventional PSNR/SSIM used by calculate_psnr_ssim.py.\n")
        f.write("- VAE cover vs watermarked isolates watermark-specific distortion after VAE reconstruction.\n")

    print("\nAverage metrics")
    print(f"GT vs VAE cover: PSNR={averages['gt_vs_vae_cover_psnr']:.4f}, SSIM={averages['gt_vs_vae_cover_ssim']:.4f}")
    print(
        f"GT vs watermarked: PSNR={averages['gt_vs_watermarked_psnr']:.4f}, "
        f"SSIM={averages['gt_vs_watermarked_ssim']:.4f}"
    )
    print(
        f"VAE cover vs watermarked: PSNR={averages['vae_cover_vs_watermarked_psnr']:.4f}, "
        f"SSIM={averages['vae_cover_vs_watermarked_ssim']:.4f}"
    )
    print(f"Report saved to: {output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
