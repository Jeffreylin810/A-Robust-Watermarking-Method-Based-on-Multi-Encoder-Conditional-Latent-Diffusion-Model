#!/usr/bin/env python3
"""Scale saved watermark residuals to probe PSNR/BA trade-off without retraining."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate cover + alpha * (watermarked - cover) images for a cheap "
            "post-hoc residual attenuation sweep."
        )
    )
    parser.add_argument(
        "--cover_dir",
        default=(
            "/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/"
            "train10000_eval1000_unseen/data/CoCo/val/gt"
        ),
        help="Directory containing original cover images.",
    )
    parser.add_argument(
        "--watermarked_dir",
        default=(
            "/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/"
            "results/test_result/O256_MSG32_SANITY20K_seed61/20260602_004053/CoCo"
        ),
        help="Directory containing saved eval watermarked images.",
    )
    parser.add_argument(
        "--output_root",
        default=(
            "/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/"
            "results/test_result/O256_MSG32_SANITY20K_seed61_residual_sweep"
        ),
        help="Output root. Each alpha is saved under alpha_0pXX/CoCo.",
    )
    parser.add_argument(
        "--alphas",
        default="0.25,0.30",
        help="Comma-separated residual scale factors. Example: 0.20,0.25,0.30",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional limit for quick checks. 0 means all matched images.",
    )
    parser.add_argument(
        "--copy_messages",
        action="store_true",
        help="Copy *_message.npy files into each alpha output directory.",
    )
    return parser.parse_args()


def parse_alphas(raw):
    alphas = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        alpha = float(item)
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        alphas.append(alpha)
    if not alphas:
        raise ValueError("No alphas were provided.")
    return alphas


def alpha_name(alpha):
    return f"alpha_{alpha:.2f}".replace(".", "p")


def list_images_by_stem(directory):
    directory = Path(directory)
    images = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images[path.stem] = path
    return images


def load_rgb_array(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def psnr_from_arrays(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse == 0:
        return float("inf"), mse
    return 20.0 * math.log10(255.0 / math.sqrt(mse)), mse


def save_rgb_array(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), mode="RGB").save(path)


def maybe_copy_message_files(watermarked_dir, output_dir):
    for message_path in sorted(watermarked_dir.glob("*_message.npy")):
        target = output_dir / message_path.name
        if not target.exists():
            target.write_bytes(message_path.read_bytes())


def main():
    args = parse_args()
    cover_dir = Path(args.cover_dir)
    watermarked_dir = Path(args.watermarked_dir)
    output_root = Path(args.output_root)
    alphas = parse_alphas(args.alphas)

    if not cover_dir.exists():
        raise FileNotFoundError(f"cover_dir does not exist: {cover_dir}")
    if not watermarked_dir.exists():
        raise FileNotFoundError(f"watermarked_dir does not exist: {watermarked_dir}")

    covers = list_images_by_stem(cover_dir)
    watermarked = list_images_by_stem(watermarked_dir)
    stems = sorted(set(covers) & set(watermarked))
    if args.max_images > 0:
        stems = stems[: args.max_images]
    if not stems:
        raise RuntimeError(f"No matched images between {cover_dir} and {watermarked_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for alpha in alphas:
        out_dir = output_root / alpha_name(alpha) / "CoCo"
        out_dir.mkdir(parents=True, exist_ok=True)
        psnrs = []
        mses = []

        for stem in stems:
            cover = load_rgb_array(covers[stem])
            wm_img = Image.open(watermarked[stem]).convert("RGB")
            if wm_img.size != (cover.shape[1], cover.shape[0]):
                wm_img = wm_img.resize((cover.shape[1], cover.shape[0]), Image.Resampling.LANCZOS)
            wm = np.asarray(wm_img, dtype=np.float32)

            scaled = cover + alpha * (wm - cover)
            scaled_uint8 = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
            save_rgb_array(out_dir / f"{stem}.png", scaled_uint8)

            psnr, mse = psnr_from_arrays(cover, scaled_uint8)
            psnrs.append(psnr)
            mses.append(mse)

        if args.copy_messages:
            maybe_copy_message_files(watermarked_dir, out_dir)

        finite_psnrs = [p for p in psnrs if math.isfinite(p)]
        avg_psnr = float(np.mean(finite_psnrs)) if finite_psnrs else float("inf")
        avg_mse = float(np.mean(mses))
        summary_rows.append(
            {
                "alpha": f"{alpha:.6g}",
                "output_dir": str(out_dir),
                "num_images": len(stems),
                "avg_psnr": "inf" if math.isinf(avg_psnr) else f"{avg_psnr:.4f}",
                "avg_mse": f"{avg_mse:.6f}",
            }
        )
        print(
            f"{alpha_name(alpha)}: images={len(stems)}, "
            f"avg_psnr={summary_rows[-1]['avg_psnr']}, output={out_dir}"
        )

    with open(output_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["alpha", "output_dir", "num_images", "avg_psnr", "avg_mse"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Summary: {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
