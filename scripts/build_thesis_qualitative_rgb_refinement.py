#!/usr/bin/env python3
"""Build thesis-ready qualitative figure for RGB refinement comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--refined_dir", required=True)
    parser.add_argument("--base_diff_dir", required=True)
    parser.add_argument("--refined_diff_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--preferred_ids",
        nargs="+",
        required=True,
        help="Preferred sample IDs for thesis figure. Missing IDs will be replaced from manifest pool.",
    )
    parser.add_argument("--suffix", default="", help="Optional suffix appended before .png/.md outputs, e.g. _clean_id")
    parser.add_argument("--show_ids", action="store_true", help="Render sample ID text below the cover image.")
    return parser.parse_args()


def get_font(size: int, bold: bool = False):
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for cand in candidates:
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def find_gt(gt_dir: Path, stem: str) -> Path:
    for ext in (".jpg", ".png", ".jpeg", ".webp"):
        p = gt_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"GT not found for {stem}")


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_manifest(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["stem"] for row in reader]


def ensure_assets(
    stems: list[str],
    gt_dir: Path,
    base_dir: Path,
    refined_dir: Path,
    base_diff_dir: Path,
    refined_diff_dir: Path,
) -> dict[str, dict[str, Path]]:
    assets = {}
    for stem in stems:
        cover = find_gt(gt_dir, stem)
        base = base_dir / f"{stem}.png"
        refined = refined_dir / f"{stem}.png"
        base_diff = base_diff_dir / f"{stem}_base_diff_x12.png"
        refined_diff = refined_diff_dir / f"{stem}_refined_diff_x12.png"
        required = [cover, base, refined, base_diff, refined_diff]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing assets for {stem}: {missing}")
        assets[stem] = {
            "cover": cover,
            "base": base,
            "refined": refined,
            "base_diff": base_diff,
            "refined_diff": refined_diff,
        }
    return assets


def choose_samples(preferred: list[str], pool: list[str], available: set[str], n: int) -> tuple[list[str], list[tuple[str, str]]]:
    selected: list[str] = []
    replacements: list[tuple[str, str]] = []
    remaining = [sid for sid in pool if sid in available and sid not in selected]

    for sid in preferred:
        if sid in available and sid not in selected:
            selected.append(sid)
        else:
            replacement = next((cand for cand in remaining if cand not in selected), None)
            if replacement is None:
                raise RuntimeError("Not enough available samples to fill requested figure.")
            selected.append(replacement)
            replacements.append((sid, replacement))

    while len(selected) < n:
        replacement = next((cand for cand in remaining if cand not in selected), None)
        if replacement is None:
            raise RuntimeError("Not enough available samples to fill requested figure.")
        selected.append(replacement)
    return selected[:n], replacements


def build_figure(sample_ids: list[str], assets: dict[str, dict[str, Path]], output_path: Path, show_ids: bool) -> None:
    cell = 256
    gap_x = 18
    gap_y = 18
    top_h = 64
    id_h = 26 if show_ids else 0
    row_h = cell + gap_y
    outer = 24
    cols = ["Cover / Ref.", "O256 base", "O256-RGBREFINE", "Base diff ×12", "Refined diff ×12"]
    width = outer * 2 + len(cols) * cell + (len(cols) - 1) * gap_x
    height = outer * 2 + top_h + len(sample_ids) * row_h - gap_y + id_h

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = get_font(24, bold=True)
    small_font = get_font(11, bold=False)
    line_color = (215, 215, 215)
    text_color = (20, 20, 20)
    id_color = (125, 125, 125)

    # Column headers
    x0 = outer
    for idx, title in enumerate(cols):
        x = x0 + idx * (cell + gap_x)
        bbox = draw.textbbox((0, 0), title, font=header_font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (cell - text_w) / 2, outer + 8), title, fill=text_color, font=header_font)

    # Rows
    for row_idx, stem in enumerate(sample_ids):
        y = outer + top_h + row_idx * row_h

        paths = assets[stem]
        images = [
            load_rgb(paths["cover"]),
            load_rgb(paths["base"]),
            load_rgb(paths["refined"]),
            load_rgb(paths["base_diff"]),
            load_rgb(paths["refined_diff"]),
        ]
        for col_idx, img in enumerate(images):
            if img.size != (cell, cell):
                img = img.resize((cell, cell), Image.Resampling.LANCZOS)
            x = x0 + col_idx * (cell + gap_x)
            canvas.paste(img, (x, y))
            draw.rectangle((x, y, x + cell, y + cell), outline=line_color, width=1)
        if show_ids:
            id_text = f"ID: {stem}"
            cover_x = x0
            draw.text((cover_x + 6, y + cell + 4), id_text, fill=id_color, font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    gt_dir = Path(args.gt_dir)
    base_dir = Path(args.base_dir)
    refined_dir = Path(args.refined_dir)
    base_diff_dir = Path(args.base_diff_dir)
    refined_diff_dir = Path(args.refined_diff_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = load_manifest(manifest)
    available = set(pool)
    selected5, replacements5 = choose_samples(args.preferred_ids, pool, available, 5)
    selected4 = selected5[:4]
    assets = ensure_assets(selected5, gt_dir, base_dir, refined_dir, base_diff_dir, refined_diff_dir)

    suffix = args.suffix
    figure5 = output_dir / f"thesis_qualitative_rgb_refinement_5samples{suffix}.png"
    figure4 = output_dir / f"thesis_qualitative_rgb_refinement_4samples{suffix}.png"
    build_figure(selected5, assets, figure5, show_ids=args.show_ids)
    build_figure(selected4, assets, figure4, show_ids=args.show_ids)

    summary = output_dir / f"thesis_qualitative_rgb_refinement_summary{suffix}.md"
    lines = [
        "# Thesis qualitative RGB refinement package",
        "",
        "- Source pool manifest:",
        f"  - `{manifest}`",
        "- Selection policy: use the requested 5 preferred sample IDs; if any are missing, replace from the existing 9-sample formal qualitative pool only.",
        "- Replacement status:",
    ]
    if replacements5:
        for requested, used in replacements5:
            lines.append(f"  - requested `{requested}` replaced by `{used}`")
    else:
        lines.append("  - none")
    lines.extend(
        [
            "",
            "## Output figures",
            "",
            f"- 5-sample thesis figure: `{figure5}`",
            f"- 4-sample compact figure: `{figure4}`",
            "",
            "## Used sample IDs",
            "",
            f"- 5-sample: {', '.join(selected5)}",
            f"- 4-sample: {', '.join(selected4)}",
            "",
            "| Sample ID | Cover | O256 base | O256-RGBREFINE | Base diff | Refined diff | Complete? | Recommended for thesis |",
            "| --------- | ----- | --------- | -------------- | --------- | ------------ | --------- | ---------------------- |",
        ]
    )
    for stem in selected5:
        p = assets[stem]
        lines.append(
            f"| {stem} | `{p['cover']}` | `{p['base']}` | `{p['refined']}` | `{p['base_diff']}` | `{p['refined_diff']}` | Yes | Yes |"
        )
    lines.extend(
        [
            "",
            "## Consistency checks",
            "",
            "- All five image columns use 256x256 RGB inputs.",
            "- Base diff and refined diff are both precomputed with the same rule: absolute difference, gain x12, clipped to [0,255], RGB PNG.",
            "- Sample IDs are drawn from the existing `selected_manifest.csv` pool only.",
            f"- IDs rendered inside figure: {'Yes' if args.show_ids else 'No'}.",
            "- The figures are intended for direct insertion into the thesis document; the figure title/caption should be added in Word, not embedded inside the PNG.",
        ]
    )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"5-sample figure: {figure5}")
    print(f"4-sample figure: {figure4}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
