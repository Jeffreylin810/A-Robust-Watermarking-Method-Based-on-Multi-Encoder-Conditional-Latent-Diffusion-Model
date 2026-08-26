#!/usr/bin/env python3
"""Oracle VAE residual sweep for latent watermarking fidelity diagnosis.

This is not a deployable watermarking method. It uses the original cover image
to estimate the image detail lost by the frozen VAE round-trip:

    residual = cover - VAE(cover)
    corrected = watermarked + beta * residual

The goal is to test whether the GT-PSNR bottleneck is mainly caused by the VAE
reconstruction ceiling and whether adding back cover-side detail can in
principle recover PSNR without rerunning watermark training.
"""

import argparse
import csv
import json
import os
import re
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
except ImportError as exc:  # pragma: no cover
    AutoencoderKL = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_GT_DIR = (
    "/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/"
    "train10000_eval1000_unseen/data/CoCo/val/gt"
)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def parse_args():
    parser = argparse.ArgumentParser(description="Oracle VAE residual correction sweep")
    parser.add_argument(
        "--config",
        default="CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml",
        help="Config basename under configs/, or an explicit yaml path.",
    )
    parser.add_argument("--gt_dir", default=DEFAULT_GT_DIR, help="Original cover image directory.")
    parser.add_argument("--watermarked_dir", required=True, help="Saved watermarked output directory.")
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output root. Corrected images are saved under beta_*/.",
    )
    parser.add_argument(
        "--betas",
        default="0,0.25,0.5,0.75,1.0",
        help="Comma-separated beta values for watermarked + beta * (cover - vae_cover).",
    )
    parser.add_argument("--max_images", type=int, default=0, help="0 means all matched images.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--save_debug",
        action="store_true",
        help="Save a small set of cover/vae/watermarked/diff images for visual inspection.",
    )
    parser.add_argument("--debug_images", type=int, default=12)
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


def parse_betas(raw):
    betas = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            betas.append(float(item))
    if not betas:
        raise ValueError("No beta values were provided.")
    return betas


def beta_name(beta):
    return "beta_" + re.sub(r"[^0-9A-Za-z-]+", "p", f"{beta:.2f}")


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
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_uint8_rgb(tensor):
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def load_uint8_resized(path, image_size):
    image = Image.open(path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


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
        raise ImportError("diffusers is required for VAE residual sweep") from IMPORT_ERROR
    if not getattr(config, "latent", None) or not getattr(config.latent, "enabled", False):
        raise ValueError("This script expects latent.enabled=True.")

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


def diff_map(a, b):
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32)
    if diff.max() > 0:
        diff = diff / diff.max() * 255.0
    return np.clip(np.rint(diff), 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    config = load_config(args.config)
    image_size = int(config.data.image_size)
    betas = parse_betas(args.betas)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    debug_root = output_root / "debug"

    gt_images = list_images_by_stem(args.gt_dir)
    wm_images = list_images_by_stem(args.watermarked_dir)
    stems = sorted(set(gt_images) & set(wm_images))
    if args.max_images > 0:
        stems = stems[: args.max_images]
    if not stems:
        raise RuntimeError(f"No matched images between {args.gt_dir} and {args.watermarked_dir}")

    vae, scaling_factor = build_vae(config, device)
    rows_by_beta = {beta: [] for beta in betas}
    baseline_rows = []

    with torch.no_grad():
        for index, stem in enumerate(stems):
            gt_tensor = load_rgb_tensor(gt_images[stem], image_size, device)
            gt_uint8 = tensor_to_uint8_rgb(gt_tensor)
            wm_uint8 = load_uint8_resized(wm_images[stem], image_size)
            vae_cover_uint8 = tensor_to_uint8_rgb(vae_reconstruct(vae, scaling_factor, gt_tensor))

            gt_vs_vae = metric_pair(vae_cover_uint8, gt_uint8)
            gt_vs_wm = metric_pair(wm_uint8, gt_uint8)
            vae_vs_wm = metric_pair(wm_uint8, vae_cover_uint8)
            baseline_rows.append(
                {
                    "stem": stem,
                    "gt_vs_vae_psnr": gt_vs_vae["psnr"],
                    "gt_vs_vae_ssim": gt_vs_vae["ssim"],
                    "gt_vs_wm_psnr": gt_vs_wm["psnr"],
                    "gt_vs_wm_ssim": gt_vs_wm["ssim"],
                    "vae_vs_wm_psnr": vae_vs_wm["psnr"],
                    "vae_vs_wm_ssim": vae_vs_wm["ssim"],
                }
            )

            residual = gt_uint8.astype(np.float32) - vae_cover_uint8.astype(np.float32)
            for beta in betas:
                corrected = wm_uint8.astype(np.float32) + beta * residual
                corrected_uint8 = np.clip(np.rint(corrected), 0, 255).astype(np.uint8)
                beta_dir = output_root / beta_name(beta)
                save_uint8(beta_dir / f"{stem}.png", corrected_uint8)
                corrected_metrics = metric_pair(corrected_uint8, gt_uint8)
                rows_by_beta[beta].append(
                    {
                        "stem": stem,
                        "beta": beta,
                        "gt_vs_corrected_psnr": corrected_metrics["psnr"],
                        "gt_vs_corrected_ssim": corrected_metrics["ssim"],
                    }
                )

            if args.save_debug and index < args.debug_images:
                save_uint8(debug_root / "cover" / f"{stem}.png", gt_uint8)
                save_uint8(debug_root / "vae_cover" / f"{stem}.png", vae_cover_uint8)
                save_uint8(debug_root / "watermarked" / f"{stem}.png", wm_uint8)
                save_uint8(debug_root / "diff_gt_vae" / f"{stem}.png", diff_map(gt_uint8, vae_cover_uint8))
                save_uint8(debug_root / "diff_gt_wm" / f"{stem}.png", diff_map(gt_uint8, wm_uint8))

            if (index + 1) % 50 == 0 or index == 0 or index + 1 == len(stems):
                print(f"{index + 1:04d}/{len(stems)} {stem}")

    summary_rows = []
    baseline_summary = {
        "gt_vs_vae_psnr": average(baseline_rows, "gt_vs_vae_psnr"),
        "gt_vs_vae_ssim": average(baseline_rows, "gt_vs_vae_ssim"),
        "gt_vs_wm_psnr": average(baseline_rows, "gt_vs_wm_psnr"),
        "gt_vs_wm_ssim": average(baseline_rows, "gt_vs_wm_ssim"),
        "vae_vs_wm_psnr": average(baseline_rows, "vae_vs_wm_psnr"),
        "vae_vs_wm_ssim": average(baseline_rows, "vae_vs_wm_ssim"),
    }
    for beta in betas:
        rows = rows_by_beta[beta]
        summary_rows.append(
            {
                "beta": beta,
                "output_dir": str(output_root / beta_name(beta)),
                "num_images": len(rows),
                "gt_vs_corrected_psnr": average(rows, "gt_vs_corrected_psnr"),
                "gt_vs_corrected_ssim": average(rows, "gt_vs_corrected_ssim"),
            }
        )

        with open(output_root / f"{beta_name(beta)}_per_image.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["stem", "beta", "gt_vs_corrected_psnr", "gt_vs_corrected_ssim"])
            writer.writeheader()
            writer.writerows(rows)

    with open(output_root / "baseline_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
        writer.writeheader()
        writer.writerows(baseline_rows)

    with open(output_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["beta", "output_dir", "num_images", "gt_vs_corrected_psnr", "gt_vs_corrected_ssim"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    report = {
        "config": str(resolve_config_path(args.config)),
        "gt_dir": str(Path(args.gt_dir)),
        "watermarked_dir": str(Path(args.watermarked_dir)),
        "output_root": str(output_root),
        "num_images": len(stems),
        "latent_model_path": config.latent.model_path,
        "latent_scaling_factor": scaling_factor,
        "baseline": baseline_summary,
        "betas": summary_rows,
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with open(output_root / "report.txt", "w", encoding="utf-8") as f:
        f.write("Oracle VAE residual sweep report\n")
        f.write("=" * 40 + "\n")
        f.write("This is an upper-bound diagnostic, not a deployable method.\n\n")
        f.write(f"config: {report['config']}\n")
        f.write(f"gt_dir: {report['gt_dir']}\n")
        f.write(f"watermarked_dir: {report['watermarked_dir']}\n")
        f.write(f"processed: {report['num_images']}\n")
        f.write(f"latent_model_path: {report['latent_model_path']}\n")
        f.write(f"latent_scaling_factor: {report['latent_scaling_factor']}\n\n")
        f.write("Baseline\n")
        f.write(f"{'comparison':<26} | {'PSNR':>9} | {'SSIM':>9}\n")
        f.write("-" * 51 + "\n")
        f.write(f"{'GT vs VAE cover':<26} | {baseline_summary['gt_vs_vae_psnr']:>9.4f} | {baseline_summary['gt_vs_vae_ssim']:>9.4f}\n")
        f.write(f"{'GT vs watermarked':<26} | {baseline_summary['gt_vs_wm_psnr']:>9.4f} | {baseline_summary['gt_vs_wm_ssim']:>9.4f}\n")
        f.write(f"{'VAE cover vs watermarked':<26} | {baseline_summary['vae_vs_wm_psnr']:>9.4f} | {baseline_summary['vae_vs_wm_ssim']:>9.4f}\n\n")
        f.write("Residual correction sweep\n")
        f.write(f"{'beta':>8} | {'PSNR':>9} | {'SSIM':>9} | output_dir\n")
        f.write("-" * 80 + "\n")
        for row in summary_rows:
            f.write(
                f"{row['beta']:>8.3f} | {row['gt_vs_corrected_psnr']:>9.4f} | "
                f"{row['gt_vs_corrected_ssim']:>9.4f} | {row['output_dir']}\n"
            )
        f.write("\nInterpretation:\n")
        f.write("- If beta raises PSNR toward 30+, VAE-lost cover detail is a major bottleneck.\n")
        f.write("- If corrected images later fail BA, residual restoration disrupts watermark recoverability.\n")
        f.write("- Because this uses the original cover image, it is an oracle diagnostic, not a final method.\n")

    print(f"Report saved to: {output_root / 'report.txt'}")


if __name__ == "__main__":
    main()
