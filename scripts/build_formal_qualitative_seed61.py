#!/usr/bin/env python3
"""Build fixed qualitative panels for formal seed61 chapter-4 figures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--refined_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--diff_gain", type=float, default=12.0)
    parser.add_argument("--num_samples", type=int, default=9)
    return parser.parse_args()


def load_map(directory: Path) -> dict[str, Path]:
    mapping = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            mapping[path.stem] = path
    return mapping


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def get_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill):
    draw.text(xy, text, fill=fill, font=font)


def make_panel(
    stem: str,
    cover: Image.Image,
    base: Image.Image,
    refined: Image.Image,
    diff: Image.Image,
    base_psnr: float,
    refined_psnr: float,
) -> Image.Image:
    w, h = cover.size
    top_h = 38
    bottom_h = 38
    panel = Image.new("RGB", (w * 4, h + top_h + bottom_h), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    title_font = get_font(15)
    meta_font = get_font(14)

    labels = [
        "GT",
        "O256 base",
        "O256-RGBREFINE",
        "Abs diff x12",
    ]
    imgs = [cover, base, refined, diff]
    for idx, (label, img) in enumerate(zip(labels, imgs)):
        x0 = idx * w
        draw_text(draw, (x0 + 10, 10), label, title_font, (255, 255, 255))
        panel.paste(img, (x0, top_h))

    meta = (
        f"{stem} | base PSNR={base_psnr:.4f} dB | "
        f"refined PSNR={refined_psnr:.4f} dB | "
        f"gain={refined_psnr - base_psnr:+.4f} dB"
    )
    draw_text(draw, (10, h + top_h + 10), meta, meta_font, (230, 230, 230))
    return panel


def make_contact_sheet(panels: list[Image.Image], columns: int = 1) -> Image.Image:
    if not panels:
        raise ValueError("No panels to arrange")
    w, h = panels[0].size
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new("RGB", (w * columns, h * rows), (12, 12, 12))
    for idx, panel in enumerate(panels):
        x = (idx % columns) * w
        y = (idx // columns) * h
        sheet.paste(panel, (x, y))
    return sheet


def quantile_positions(n: int, k: int) -> list[int]:
    if k <= 1:
        return [n // 2]
    fractions = np.linspace(0.05, 0.95, num=k)
    positions = [min(n - 1, max(0, int(round(f * (n - 1))))) for f in fractions]
    deduped = []
    seen = set()
    for pos in positions:
        if pos not in seen:
            deduped.append(pos)
            seen.add(pos)
    return deduped


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.gt_dir)
    base_dir = Path(args.base_dir)
    refined_dir = Path(args.refined_dir)
    output_dir = Path(args.output_dir)
    panel_dir = output_dir / "panels"
    diff_dir = output_dir / "diff_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    diff_dir.mkdir(parents=True, exist_ok=True)

    gt_map = load_map(gt_dir)
    base_map = load_map(base_dir)
    refined_map = load_map(refined_dir)
    stems = sorted(set(gt_map) & set(base_map) & set(refined_map))
    if len(stems) < args.num_samples:
        raise RuntimeError(f"Only {len(stems)} matched stems; need at least {args.num_samples}")

    rows = []
    for stem in stems:
        cover = load_rgb(gt_map[stem])
        base = load_rgb(base_map[stem])
        refined = load_rgb(refined_map[stem])
        if cover.size != base.size:
            cover = cover.resize(base.size, Image.Resampling.LANCZOS)
        if refined.size != base.size:
            refined = refined.resize(base.size, Image.Resampling.LANCZOS)
        cover_arr = np.asarray(cover)
        base_arr = np.asarray(base)
        refined_arr = np.asarray(refined)
        base_psnr = psnr(cover_arr, base_arr)
        refined_psnr = psnr(cover_arr, refined_arr)
        rows.append(
            {
                "stem": stem,
                "base_psnr": base_psnr,
                "refined_psnr": refined_psnr,
                "gain_psnr": refined_psnr - base_psnr,
            }
        )

    rows_sorted = sorted(rows, key=lambda row: (row["base_psnr"], row["stem"]))
    positions = quantile_positions(len(rows_sorted), args.num_samples)
    selected = [rows_sorted[pos] for pos in positions]

    manifest_path = output_dir / "selected_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "stem", "base_psnr", "refined_psnr", "gain_psnr"],
        )
        writer.writeheader()
        for rank, row in enumerate(selected, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "stem": row["stem"],
                    "base_psnr": f"{row['base_psnr']:.4f}",
                    "refined_psnr": f"{row['refined_psnr']:.4f}",
                    "gain_psnr": f"{row['gain_psnr']:+.4f}",
                }
            )

    panels: list[Image.Image] = []
    for row in selected:
        stem = row["stem"]
        cover = load_rgb(gt_map[stem])
        base = load_rgb(base_map[stem])
        refined = load_rgb(refined_map[stem])
        if cover.size != base.size:
            cover = cover.resize(base.size, Image.Resampling.LANCZOS)
        if refined.size != base.size:
            refined = refined.resize(base.size, Image.Resampling.LANCZOS)
        cover_arr = np.asarray(cover)
        refined_arr = np.asarray(refined)
        diff_vis = np.clip(
            np.abs(refined_arr.astype(np.int16) - cover_arr.astype(np.int16)).astype(np.float32)
            * args.diff_gain,
            0,
            255,
        ).astype(np.uint8)
        diff_img = Image.fromarray(diff_vis, mode="RGB")
        diff_img.save(diff_dir / f"{stem}_refined_diff_x{args.diff_gain:g}.png")
        panel = make_panel(
            stem=stem,
            cover=cover,
            base=base,
            refined=refined,
            diff=diff_img,
            base_psnr=row["base_psnr"],
            refined_psnr=row["refined_psnr"],
        )
        panel.save(panel_dir / f"{stem}_panel.png")
        panels.append(panel)

    contact_sheet = make_contact_sheet(panels, columns=1)
    contact_sheet.save(output_dir / "contact_sheet.png")

    notes = [
        "# Formal Qualitative Selection (seed61)",
        "",
        "- Selection rule: deterministic quantile sampling over the 1000 formal finaltest images, sorted by O256 base PSNR.",
        f"- Number of selected samples: {len(selected)}",
        f"- Difference-map gain: x{args.diff_gain:g}",
        "- Columns: GT / O256 base / O256-RGBREFINE / amplified absolute refined-vs-GT difference map",
        "- This figure is intended for chapter-4 qualitative evidence and should not be described as manually cherry-picked.",
        "",
        f"- Manifest: {manifest_path}",
        f"- Panels: {panel_dir}",
        f"- Contact sheet: {output_dir / 'contact_sheet.png'}",
    ]
    (output_dir / "README.md").write_text("\n".join(notes), encoding="utf-8")

    print(f"selected samples: {len(selected)}")
    print(f"manifest: {manifest_path}")
    print(f"contact sheet: {output_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
