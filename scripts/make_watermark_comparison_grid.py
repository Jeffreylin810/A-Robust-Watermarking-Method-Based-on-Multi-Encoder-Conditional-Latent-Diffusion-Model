#!/usr/bin/env python3
"""Create cover / watermarked / difference-map panels for paper inspection."""

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate cover-watermarked-difference panels from saved eval images."
    )
    parser.add_argument(
        "--gt_dir",
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
            "results/test_result/O256_PAPER_seed61/CoCo/CoCo"
        ),
        help="Directory containing eval_diffusion.py watermarked PNG images.",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/"
            "results/figures/O256_PAPER_seed61_comparison"
        ),
        help="Directory where panels and summary.csv will be saved.",
    )
    parser.add_argument("--num_images", type=int, default=12, help="Number of matched images to export.")
    parser.add_argument(
        "--diff_gain",
        type=float,
        default=10.0,
        help="Multiplier for visualizing absolute RGB difference.",
    )
    parser.add_argument(
        "--label_height",
        type=int,
        default=28,
        help="Height reserved for column labels in each panel.",
    )
    return parser.parse_args()


def list_images_by_stem(directory):
    directory = Path(directory)
    images = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images[path.stem] = path
    return images


def load_rgb(path):
    return Image.open(path).convert("RGB")


def psnr_from_arrays(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse == 0:
        return float("inf"), mse
    return 20.0 * math.log10(255.0 / math.sqrt(mse)), mse


def draw_label(canvas, x, y, text):
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((x + 8, y + 6), text, fill=(255, 255, 255), font=font)


def make_panel(cover, watermarked, diff, labels, label_height):
    w, h = cover.size
    panel = Image.new("RGB", (w * 3, h + label_height), (20, 20, 20))
    for idx, (img, label) in enumerate(zip((cover, watermarked, diff), labels)):
        x0 = idx * w
        draw_label(panel, x0, 0, label)
        panel.paste(img, (x0, label_height))
    return panel


def main():
    args = parse_args()
    gt_dir = Path(args.gt_dir)
    watermarked_dir = Path(args.watermarked_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not gt_dir.exists():
        raise FileNotFoundError(f"gt_dir does not exist: {gt_dir}")
    if not watermarked_dir.exists():
        raise FileNotFoundError(f"watermarked_dir does not exist: {watermarked_dir}")

    gt_images = list_images_by_stem(gt_dir)
    wm_images = list_images_by_stem(watermarked_dir)
    matched_stems = sorted(set(gt_images) & set(wm_images))
    if not matched_stems:
        raise RuntimeError(
            f"No matched image stems found between {gt_dir} and {watermarked_dir}"
        )

    rows = []
    selected = matched_stems[: args.num_images]
    for stem in selected:
        cover = load_rgb(gt_images[stem])
        watermarked = load_rgb(wm_images[stem])
        if cover.size != watermarked.size:
            cover = cover.resize(watermarked.size, Image.Resampling.LANCZOS)

        cover_arr = np.asarray(cover)
        wm_arr = np.asarray(watermarked)
        abs_diff = np.abs(wm_arr.astype(np.int16) - cover_arr.astype(np.int16)).astype(np.float32)
        diff_vis = np.clip(abs_diff * args.diff_gain, 0, 255).astype(np.uint8)
        diff_img = Image.fromarray(diff_vis, mode="RGB")

        psnr, mse = psnr_from_arrays(cover_arr, wm_arr)
        mae = float(np.mean(abs_diff))
        max_abs = int(np.max(abs_diff))

        panel = make_panel(
            cover,
            watermarked,
            diff_img,
            ("cover", "watermarked", f"abs diff x{args.diff_gain:g}"),
            args.label_height,
        )
        panel.save(output_dir / f"{stem}_comparison.png")
        diff_img.save(output_dir / f"{stem}_diff_x{args.diff_gain:g}.png")

        rows.append(
            {
                "stem": stem,
                "cover_path": str(gt_images[stem]),
                "watermarked_path": str(wm_images[stem]),
                "psnr": "inf" if math.isinf(psnr) else f"{psnr:.4f}",
                "mse": f"{mse:.6f}",
                "mae": f"{mae:.6f}",
                "max_abs_diff": max_abs,
            }
        )

    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "cover_path",
                "watermarked_path",
                "psnr",
                "mse",
                "mae",
                "max_abs_diff",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Matched images: {len(matched_stems)}")
    print(f"Exported panels: {len(rows)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
