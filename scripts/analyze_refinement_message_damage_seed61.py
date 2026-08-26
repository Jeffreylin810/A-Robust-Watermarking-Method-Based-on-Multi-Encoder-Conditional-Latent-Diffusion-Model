#!/usr/bin/env python3
"""Diagnose how RGB refinement changes clean message recovery.

This script compares O256-Base watermarked outputs with O256-RGB refined
outputs on matched images. It is intentionally diagnostic-only: it does not
train a model and it does not generate attack images.

Outputs:
  - per_image_damage.csv
  - bit_error_summary.csv
  - summary.md
  - optional top-damage residual/sensitivity maps
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from models.ddm import DenoisingDiffusion, data_transform  # noqa: E402
from train_rgb_refinement import (  # noqa: E402
    IMAGE_EXTS,
    dict2namespace,
    get_message_decoder,
    metric_pair,
    pil_to_tensor,
    tensor_to_uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RGB refinement clean-BA damage.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--base_dir", required=True, help="O256-Base image folder with *_message.npy.")
    parser.add_argument("--refined_dir", required=True, help="O256-RGB refined image folder.")
    parser.add_argument("--gt_dir", required=True, help="Cover/reference GT folder.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--with_sensitivity", action="store_true")
    parser.add_argument(
        "--save_top_maps",
        type=int,
        default=0,
        help="Save diagnostic maps for top-K BA-drop images. Keep this small to save disk.",
    )
    parser.add_argument("--diff_gain", type=float, default=12.0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def load_config(config_path: str):
    path = Path(config_path)
    if not path.exists():
        path = PROJECT_ROOT / "configs" / config_path
    with path.open("r", encoding="utf-8") as handle:
        return dict2namespace(yaml.safe_load(handle))


def build_diffusion(config, resume: str, device: torch.device) -> DenoisingDiffusion:
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


def image_files_by_stem(path: Path) -> dict[str, Path]:
    return {
        item.stem: item
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    }


def load_message(base_dir: Path, stem: str, device: torch.device) -> torch.Tensor:
    message_path = base_dir / f"{stem}_message.npy"
    if not message_path.exists():
        raise FileNotFoundError(message_path)
    message = np.load(message_path).astype(np.float32).reshape(1, -1)
    return torch.from_numpy(message).to(device)


def decode_message(diffusion: DenoisingDiffusion, image_0_1: torch.Tensor, requires_grad: bool = False):
    if requires_grad:
        latent = diffusion.encode_images(data_transform(image_0_1), track_grad=True)
        decoded = get_message_decoder(diffusion)(latent)
        return decoded, latent
    with torch.no_grad():
        latent = diffusion.encode_images(data_transform(image_0_1), track_grad=False)
        decoded = get_message_decoder(diffusion)(latent)
    return decoded, latent


def bit_accuracy_and_errors(decoded: torch.Tensor, message: torch.Tensor) -> tuple[float, np.ndarray]:
    decoded_np = decoded.detach().cpu().numpy().round().clip(0, 1)
    message_np = message.detach().cpu().numpy()
    errors = np.abs(decoded_np - message_np).reshape(-1)
    return float((1.0 - errors.mean()) * 100.0), errors


def save_map(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8)).save(path)


def make_heatmap_rgb(values: torch.Tensor, gain: float) -> np.ndarray:
    values = values.detach().float().cpu().clamp(min=0.0)
    if values.ndim == 4:
        values = values[0]
    if values.shape[0] == 1:
        values = values.repeat(3, 1, 1)
    values = values / values.max().clamp_min(1.0e-12)
    return tensor_to_uint8((values * gain).clamp(0.0, 1.0))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    diffusion = build_diffusion(config, args.resume, device)

    base_dir = Path(args.base_dir)
    refined_dir = Path(args.refined_dir)
    gt_dir = Path(args.gt_dir)
    base_images = image_files_by_stem(base_dir)
    refined_images = image_files_by_stem(refined_dir)
    gt_images = image_files_by_stem(gt_dir)
    stems = sorted(set(base_images) & set(refined_images) & set(gt_images))
    stems = [stem for stem in stems if (base_dir / f"{stem}_message.npy").exists()]
    if args.max_images > 0:
        stems = stems[: args.max_images]
    if not stems:
        raise RuntimeError("No matched base/refined/GT/message samples found.")

    rows: list[dict[str, object]] = []
    bit_base_errors: list[np.ndarray] = []
    bit_refined_errors: list[np.ndarray] = []

    for index, stem in enumerate(stems, start=1):
        base = pil_to_tensor(base_images[stem], args.image_size).unsqueeze(0).to(device)
        refined = pil_to_tensor(refined_images[stem], args.image_size).unsqueeze(0).to(device)
        gt = pil_to_tensor(gt_images[stem], args.image_size).unsqueeze(0).to(device)
        message = load_message(base_dir, stem, device)

        base_decoded, base_latent = decode_message(diffusion, base)
        refined_decoded, refined_latent = decode_message(diffusion, refined)
        base_ba, base_errors = bit_accuracy_and_errors(base_decoded, message)
        refined_ba, refined_errors = bit_accuracy_and_errors(refined_decoded, message)
        bit_base_errors.append(base_errors)
        bit_refined_errors.append(refined_errors)

        base_psnr, base_ssim = metric_pair(base[0], gt[0])
        refined_psnr, refined_ssim = metric_pair(refined[0], gt[0])
        residual = refined - base
        residual_abs = residual.abs()
        latent_diff = refined_latent - base_latent

        sensitivity_overlap = ""
        sensitivity_cosine_abs = ""
        grad_mean_abs = ""
        msg_loss_refined = ""
        if args.with_sensitivity:
            refined_grad_image = refined.detach().clone().requires_grad_(True)
            decoded_grad, _latent_grad = decode_message(diffusion, refined_grad_image, requires_grad=True)
            msg_loss = F.mse_loss(decoded_grad, message)
            grad = torch.autograd.grad(msg_loss, refined_grad_image, retain_graph=False, create_graph=False)[0]
            grad_abs = grad.detach().abs()
            residual_flat = residual.detach().reshape(-1)
            grad_flat = grad.detach().reshape(-1)
            sensitivity_overlap = float(
                (residual_abs.detach() * grad_abs).mean()
                / (residual_abs.detach().mean() * grad_abs.mean()).clamp_min(1.0e-12)
            )
            sensitivity_cosine_abs = float(
                torch.abs(torch.dot(residual_flat, grad_flat))
                / (torch.linalg.norm(residual_flat) * torch.linalg.norm(grad_flat)).clamp_min(1.0e-12)
            )
            grad_mean_abs = float(grad_abs.mean().detach().cpu())
            msg_loss_refined = float(msg_loss.detach().cpu())

        row = {
            "rank_input_order": index,
            "stem": stem,
            "base_ba": f"{base_ba:.6f}",
            "refined_ba": f"{refined_ba:.6f}",
            "ba_drop_pp": f"{base_ba - refined_ba:.6f}",
            "base_psnr": f"{base_psnr:.6f}",
            "refined_psnr": f"{refined_psnr:.6f}",
            "psnr_delta": f"{refined_psnr - base_psnr:.6f}",
            "base_ssim": f"{base_ssim:.6f}",
            "refined_ssim": f"{refined_ssim:.6f}",
            "ssim_delta": f"{refined_ssim - base_ssim:.6f}",
            "residual_mean_abs": f"{float(residual_abs.mean().detach().cpu()):.8f}",
            "residual_p95_abs": f"{float(torch.quantile(residual_abs.detach().flatten(), 0.95).cpu()):.8f}",
            "residual_max_abs": f"{float(residual_abs.max().detach().cpu()):.8f}",
            "latent_mse_refined_vs_base": f"{float(F.mse_loss(refined_latent, base_latent).detach().cpu()):.8f}",
            "latent_l1_refined_vs_base": f"{float(F.l1_loss(refined_latent, base_latent).detach().cpu()):.8f}",
            "sensitivity_overlap": sensitivity_overlap,
            "sensitivity_cosine_abs": sensitivity_cosine_abs,
            "grad_mean_abs": grad_mean_abs,
            "msg_loss_refined": msg_loss_refined,
        }
        rows.append(row)
        if index % 50 == 0:
            print(f"processed {index}/{len(stems)}")

    rows_sorted = sorted(rows, key=lambda item: float(item["ba_drop_pp"]), reverse=True)
    per_image_csv = output_dir / "per_image_damage.csv"
    fieldnames = list(rows[0].keys())
    with per_image_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    base_err = np.stack(bit_base_errors, axis=0)
    refined_err = np.stack(bit_refined_errors, axis=0)
    bit_rows = []
    for bit in range(base_err.shape[1]):
        bit_rows.append(
            {
                "bit": bit,
                "base_error_rate_percent": f"{base_err[:, bit].mean() * 100.0:.6f}",
                "refined_error_rate_percent": f"{refined_err[:, bit].mean() * 100.0:.6f}",
                "error_delta_pp": f"{(refined_err[:, bit].mean() - base_err[:, bit].mean()) * 100.0:.6f}",
            }
        )
    bit_csv = output_dir / "bit_error_summary.csv"
    with bit_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bit_rows)

    if args.save_top_maps > 0:
        map_dir = output_dir / "top_damage_maps"
        for map_rank, row in enumerate(rows_sorted[: args.save_top_maps], start=1):
            stem = str(row["stem"])
            base = pil_to_tensor(base_images[stem], args.image_size).unsqueeze(0).to(device)
            refined = pil_to_tensor(refined_images[stem], args.image_size).unsqueeze(0).to(device)
            residual = (refined - base).abs()[0]
            save_map(map_dir / f"{map_rank:02d}_{stem}_base.png", tensor_to_uint8(base[0]))
            save_map(map_dir / f"{map_rank:02d}_{stem}_refined.png", tensor_to_uint8(refined[0]))
            save_map(map_dir / f"{map_rank:02d}_{stem}_residual_x{args.diff_gain:g}.png", tensor_to_uint8((residual * args.diff_gain).clamp(0.0, 1.0)))
            if args.with_sensitivity:
                message = load_message(base_dir, stem, device)
                refined_grad_image = refined.detach().clone().requires_grad_(True)
                decoded_grad, _latent_grad = decode_message(diffusion, refined_grad_image, requires_grad=True)
                msg_loss = F.mse_loss(decoded_grad, message)
                grad = torch.autograd.grad(msg_loss, refined_grad_image, retain_graph=False, create_graph=False)[0]
                save_map(map_dir / f"{map_rank:02d}_{stem}_sensitivity_norm.png", make_heatmap_rgb(grad.abs(), 1.0))

    ba_drops = [float(row["ba_drop_pp"]) for row in rows]
    residual_means = [float(row["residual_mean_abs"]) for row in rows]
    latent_mses = [float(row["latent_mse_refined_vs_base"]) for row in rows]
    summary = {
        "count": len(rows),
        "mean_base_ba": float(np.mean([float(row["base_ba"]) for row in rows])),
        "mean_refined_ba": float(np.mean([float(row["refined_ba"]) for row in rows])),
        "mean_ba_drop_pp": float(np.mean(ba_drops)),
        "median_ba_drop_pp": float(np.median(ba_drops)),
        "top10_mean_ba_drop_pp": float(np.mean([float(row["ba_drop_pp"]) for row in rows_sorted[:10]])),
        "mean_base_psnr": float(np.mean([float(row["base_psnr"]) for row in rows])),
        "mean_refined_psnr": float(np.mean([float(row["refined_psnr"]) for row in rows])),
        "corr_ba_drop_vs_residual_mean_abs": pearson(ba_drops, residual_means),
        "corr_ba_drop_vs_latent_mse": pearson(ba_drops, latent_mses),
        "per_image_csv": str(per_image_csv),
        "bit_error_csv": str(bit_csv),
    }
    if args.with_sensitivity:
        valid_pairs = [
            (float(row["ba_drop_pp"]), float(row["sensitivity_overlap"]))
            for row in rows
            if row["sensitivity_overlap"] != ""
        ]
        summary["corr_ba_drop_vs_sensitivity_overlap"] = pearson(
            [pair[0] for pair in valid_pairs],
            [pair[1] for pair in valid_pairs],
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    top_rows_md = "\n".join(
        f"| {i} | {row['stem']} | {float(row['ba_drop_pp']):.2f} | "
        f"{float(row['base_ba']):.2f} | {float(row['refined_ba']):.2f} | "
        f"{float(row['residual_mean_abs']):.6f} | {float(row['latent_mse_refined_vs_base']):.6f} |"
        for i, row in enumerate(rows_sorted[:10], start=1)
    )
    md = f"""# RGB Refinement Message-Damage Diagnostic

## Inputs

- base_dir: `{base_dir}`
- refined_dir: `{refined_dir}`
- gt_dir: `{gt_dir}`
- config: `{args.config}`
- resume: `{args.resume}`
- with_sensitivity: `{args.with_sensitivity}`

## Summary

| Metric | Value |
|---|---:|
| sample count | {summary['count']} |
| mean O256-Base clean BA | {summary['mean_base_ba']:.4f}% |
| mean O256-RGB clean BA | {summary['mean_refined_ba']:.4f}% |
| mean clean BA drop | {summary['mean_ba_drop_pp']:.4f} pp |
| median clean BA drop | {summary['median_ba_drop_pp']:.4f} pp |
| top-10 mean clean BA drop | {summary['top10_mean_ba_drop_pp']:.4f} pp |
| mean O256-Base PSNR | {summary['mean_base_psnr']:.4f} dB |
| mean O256-RGB PSNR | {summary['mean_refined_psnr']:.4f} dB |
| corr(BA drop, residual mean abs) | {summary['corr_ba_drop_vs_residual_mean_abs']:.4f} |
| corr(BA drop, latent MSE) | {summary['corr_ba_drop_vs_latent_mse']:.4f} |
"""
    if args.with_sensitivity:
        md += f"| corr(BA drop, sensitivity overlap) | {summary.get('corr_ba_drop_vs_sensitivity_overlap', float('nan')):.4f} |\n"
    md += f"""
## Top Clean-BA Drop Samples

| Rank | Sample | BA drop pp | Base BA | Refined BA | Residual mean abs | Latent MSE |
|---:|---|---:|---:|---:|---:|---:|
{top_rows_md}

## Output Files

- per-image diagnostics: `{per_image_csv}`
- bit-level error summary: `{bit_csv}`
- summary json: `{output_dir / 'summary.json'}`
"""
    if args.save_top_maps > 0:
        md += f"- top-damage maps: `{output_dir / 'top_damage_maps'}`\n"
    (output_dir / "summary.md").write_text(md, encoding="utf-8")
    print(f"saved: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
