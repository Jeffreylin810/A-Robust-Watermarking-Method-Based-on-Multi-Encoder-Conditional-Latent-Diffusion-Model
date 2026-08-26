#!/usr/bin/env python3
"""Build base difference maps for the existing formal qualitative package."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--refined_dir", required=True)
    parser.add_argument("--refined_diff_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gain", type=float, default=12.0)
    return parser.parse_args()


def get_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def find_gt(gt_dir: Path, stem: str) -> Path:
    for ext in (".jpg", ".png", ".jpeg", ".webp"):
        p = gt_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"GT not found for {stem}")


def diff_image(a: Image.Image, b: Image.Image, gain: float) -> Image.Image:
    arr_a = np.asarray(a, dtype=np.int16)
    arr_b = np.asarray(b, dtype=np.int16)
    diff = np.clip(np.abs(arr_a - arr_b).astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def make_panel(stem: str, cover: Image.Image, base: Image.Image, refined: Image.Image, base_diff: Image.Image, refined_diff: Image.Image) -> Image.Image:
    w, h = cover.size
    top_h = 38
    panel = Image.new("RGB", (w * 5, h + top_h), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    title_font = get_font(15)
    labels = ["GT", "O256 base", "O256-RGBREFINE", "Base diff x12", "Refined diff x12"]
    imgs = [cover, base, refined, base_diff, refined_diff]
    for idx, (label, img) in enumerate(zip(labels, imgs)):
        x0 = idx * w
        draw.text((x0 + 10, 10), label, fill=(255, 255, 255), font=title_font)
        panel.paste(img, (x0, top_h))
    return panel


def make_contact_sheet(panels: list[Image.Image]) -> Image.Image:
    if not panels:
        raise ValueError("No panels")
    w, h = panels[0].size
    sheet = Image.new("RGB", (w, h * len(panels)), (12, 12, 12))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (0, i * h))
    return sheet


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    gt_dir = Path(args.gt_dir)
    base_dir = Path(args.base_dir)
    refined_dir = Path(args.refined_dir)
    refined_diff_dir = Path(args.refined_diff_dir)
    out_root = Path(args.output_dir)
    base_diff_dir = out_root / "base_diff_maps"
    panel_dir = out_root / "panels_with_base_diff"
    out_root.mkdir(parents=True, exist_ok=True)
    base_diff_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    panels = []
    with manifest.open("r", encoding="utf-8") as handle:
      reader = csv.DictReader(handle)
      for rec in reader:
        stem = rec["stem"]
        cover_path = find_gt(gt_dir, stem)
        base_path = base_dir / f"{stem}.png"
        refined_path = refined_dir / f"{stem}.png"
        refined_diff_path = refined_diff_dir / f"{stem}_refined_diff_x{args.gain:g}.png"
        if not (base_path.exists() and refined_path.exists() and refined_diff_path.exists()):
            raise FileNotFoundError(f"Missing qualitative asset for {stem}")
        cover = load_rgb(cover_path)
        base = load_rgb(base_path)
        refined = load_rgb(refined_path)
        if cover.size != base.size:
            cover = cover.resize(base.size, Image.Resampling.LANCZOS)
        if refined.size != base.size:
            refined = refined.resize(base.size, Image.Resampling.LANCZOS)
        refined_diff = load_rgb(refined_diff_path)
        base_diff = diff_image(base, cover, args.gain)
        base_diff_path = base_diff_dir / f"{stem}_base_diff_x{args.gain:g}.png"
        base_diff.save(base_diff_path)
        panel = make_panel(stem, cover, base, refined, base_diff, refined_diff)
        panel_path = panel_dir / f"{stem}_panel.png"
        panel.save(panel_path)
        panels.append(panel)
        rows.append({
            "Sample ID": stem,
            "Cover": str(cover_path),
            "O256 base": str(base_path),
            "O256-RGBREFINE": str(refined_path),
            "Base diff": str(base_diff_path),
            "Refined diff": str(refined_diff_path),
            "Complete?": "Yes",
            "Recommended for thesis": "Yes",
        })

    contact = make_contact_sheet(panels)
    contact_path = out_root / "contact_sheet_with_base_diff.png"
    contact.save(contact_path)

    summary_path = out_root / "updated_qualitative_package_summary.md"
    lines = [
        "# Updated Formal Qualitative Package (with base diff maps)",
        "",
        f"- Gain rule: absolute difference × {args.gain:g}, clipped to [0,255], RGB PNG.",
        f"- Manifest: {manifest}",
        f"- Base diff maps: {base_diff_dir}",
        f"- New contact sheet: {contact_path}",
        "",
        "| Sample ID | Cover | O256 base | O256-RGBREFINE | Base diff | Refined diff | Complete? | Recommended for thesis |",
        "| --------- | ----- | --------- | -------------- | --------- | ------------ | --------- | ---------------------- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['Sample ID']} | `{row['Cover']}` | `{row['O256 base']}` | `{row['O256-RGBREFINE']}` | `{row['Base diff']}` | `{row['Refined diff']}` | {row['Complete?']} | {row['Recommended for thesis']} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print(f"wrote {contact_path}")


if __name__ == "__main__":
    main()
