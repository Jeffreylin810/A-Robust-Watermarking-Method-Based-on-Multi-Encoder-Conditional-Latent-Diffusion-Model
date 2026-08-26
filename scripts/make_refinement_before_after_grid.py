#!/usr/bin/env python3
"""Create GT / before-refinement / after-refinement comparison panels."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--before_dir", required=True)
    parser.add_argument("--after_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--diff_gain", type=float, default=10.0)
    return parser.parse_args()


def list_images(directory):
    directory = Path(directory)
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def load_rgb(path, size=None):
    image = Image.open(path).convert("RGB")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def draw_label(canvas, x, text):
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((x + 8, 6), text, fill=(255, 255, 255), font=font)


def diff_image(ref, pred, gain):
    ref_arr = np.asarray(ref)
    pred_arr = np.asarray(pred)
    diff = np.abs(pred_arr.astype(np.int16) - ref_arr.astype(np.int16)).astype(np.float32)
    return Image.fromarray(np.clip(diff * gain, 0, 255).astype(np.uint8), mode="RGB")


def make_panel(gt, before, after, before_psnr, after_psnr, gain):
    width, height = gt.size
    label_h = 34
    panel = Image.new("RGB", (width * 5, height + label_h), (18, 18, 18))
    columns = [
        (gt, "GT cover"),
        (before, f"Before refine PSNR {before_psnr:.2f}"),
        (after, f"After refine PSNR {after_psnr:.2f}"),
        (diff_image(gt, before, gain), f"|Before-GT| x{gain:g}"),
        (diff_image(gt, after, gain), f"|After-GT| x{gain:g}"),
    ]
    for idx, (image, label) in enumerate(columns):
        x = idx * width
        draw_label(panel, x, label)
        panel.paste(image, (x, label_h))
    return panel


def choose_stems(rows, count):
    if len(rows) <= count:
        return [row["stem"] for row in rows]

    selected = []

    def add(stem):
        if stem not in selected:
            selected.append(stem)

    for row in rows[:2]:
        add(row["stem"])
    for row in sorted(rows, key=lambda r: r["improvement"], reverse=True):
        add(row["stem"])
        if len(selected) >= max(3, count // 2):
            break
    for row in sorted(rows, key=lambda r: r["after_psnr"]):
        add(row["stem"])
        if len(selected) >= count:
            break
    for row in sorted(rows, key=lambda r: r["after_psnr"]):
        add(row["stem"])
        if len(selected) >= count:
            break
    return selected[:count]


def main():
    args = parse_args()
    gt_paths = list_images(args.gt_dir)
    before_paths = list_images(args.before_dir)
    after_paths = list_images(args.after_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matched = sorted(set(gt_paths) & set(before_paths) & set(after_paths))
    if not matched:
        raise RuntimeError("No matched image stems found.")

    rows = []
    for stem in matched:
        gt = load_rgb(gt_paths[stem])
        before = load_rgb(before_paths[stem], gt.size)
        after = load_rgb(after_paths[stem], gt.size)
        gt_arr = np.asarray(gt)
        before_arr = np.asarray(before)
        after_arr = np.asarray(after)
        before_psnr = psnr(gt_arr, before_arr)
        after_psnr = psnr(gt_arr, after_arr)
        rows.append(
            {
                "stem": stem,
                "before_psnr": before_psnr,
                "after_psnr": after_psnr,
                "improvement": after_psnr - before_psnr,
                "gt_path": str(gt_paths[stem]),
                "before_path": str(before_paths[stem]),
                "after_path": str(after_paths[stem]),
            }
        )

    selected = set(choose_stems(rows, args.num_images))
    selected_rows = [row for row in rows if row["stem"] in selected]
    selected_rows.sort(key=lambda r: r["stem"])

    for row in selected_rows:
        gt = load_rgb(row["gt_path"])
        before = load_rgb(row["before_path"], gt.size)
        after = load_rgb(row["after_path"], gt.size)
        panel = make_panel(
            gt,
            before,
            after,
            row["before_psnr"],
            row["after_psnr"],
            args.diff_gain,
        )
        panel.save(output_dir / f"{row['stem']}_before_after.png")

    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "before_psnr",
                "after_psnr",
                "improvement",
                "gt_path",
                "before_path",
                "after_path",
            ],
        )
        writer.writeheader()
        for row in selected_rows:
            out = dict(row)
            for key in ("before_psnr", "after_psnr", "improvement"):
                out[key] = f"{out[key]:.4f}"
            writer.writerow(out)

    print(f"Matched images: {len(matched)}")
    print(f"Exported panels: {len(selected_rows)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
