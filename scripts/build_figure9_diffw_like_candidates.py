#!/usr/bin/env python3
"""Build Figure 9 qualitative candidates with DiffW-like output included."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--diffw_dir", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--refined_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_ids", nargs="+", required=True)
    parser.add_argument("--diff_gain", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def find_image(directory: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find image for {stem} in {directory}")


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_for_calc(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    if img.size == size:
        return img
    return img.resize(size, Image.Resampling.LANCZOS)


def diff_x_gain(reference: Image.Image, candidate: Image.Image, gain: float) -> Image.Image:
    candidate = resize_for_calc(candidate, reference.size)
    ref_arr = np.asarray(reference).astype(np.float32)
    cand_arr = np.asarray(candidate).astype(np.float32)
    diff = np.clip(np.abs(cand_arr - ref_arr) * gain, 0, 255).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def center_text(draw: ImageDraw.ImageDraw, xywh: tuple[int, int, int, int], text: str, font, fill) -> None:
    x, y, w, h = xywh
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x + (w - tw) / 2, y + (h - th) / 2 - 1), text, font=font, fill=fill)


def build_grid(
    sample_ids: list[str],
    assets: dict[str, dict[str, Path]],
    output_path: Path,
    columns: list[tuple[str, str]],
    diff_gain: float,
    cell: int,
    dpi: int,
) -> None:
    gap_x = 24
    gap_y = 22
    outer = 28
    header_h = 58
    width = outer * 2 + len(columns) * cell + (len(columns) - 1) * gap_x
    height = outer * 2 + header_h + len(sample_ids) * cell + (len(sample_ids) - 1) * gap_y

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = get_font(27 if len(columns) <= 6 else 23, bold=True)
    border = (218, 218, 218)
    text = (20, 20, 20)

    for col_idx, (title, _) in enumerate(columns):
        x = outer + col_idx * (cell + gap_x)
        center_text(draw, (x, outer, cell, header_h), title, header_font, text)

    for row_idx, stem in enumerate(sample_ids):
        paths = assets[stem]
        cover = load_rgb(paths["cover"])
        diffw = load_rgb(paths["diffw"])
        base = load_rgb(paths["base"])
        refined = load_rgb(paths["refined"])
        image_map = {
            "cover": cover,
            "diffw": diffw,
            "base": base,
            "refined": refined,
            "diffw_diff": diff_x_gain(cover, diffw, diff_gain),
            "base_diff": diff_x_gain(cover, base, diff_gain),
            "refined_diff": diff_x_gain(cover, refined, diff_gain),
        }
        y = outer + header_h + row_idx * (cell + gap_y)
        for col_idx, (_, key) in enumerate(columns):
            x = outer + col_idx * (cell + gap_x)
            img = image_map[key]
            if img.size != (cell, cell):
                img = img.resize((cell, cell), Image.Resampling.LANCZOS)
            canvas.paste(img, (x, y))
            draw.rectangle((x, y, x + cell, y + cell), outline=border, width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True, dpi=(dpi, dpi))


def collect_assets(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    gt_dir = Path(args.gt_dir)
    diffw_dir = Path(args.diffw_dir)
    base_dir = Path(args.base_dir)
    refined_dir = Path(args.refined_dir)
    assets: dict[str, dict[str, Path]] = {}
    for stem in args.sample_ids:
        paths = {
            "cover": find_image(gt_dir, stem),
            "diffw": find_image(diffw_dir, stem),
            "base": find_image(base_dir, stem),
            "refined": find_image(refined_dir, stem),
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing assets for {stem}: {missing}")
        assets[stem] = paths
    return assets


def write_summary(
    output_dir: Path,
    sample_ids: list[str],
    assets: dict[str, dict[str, Path]],
    args: argparse.Namespace,
    figure6: Path,
    figure7: Path,
) -> None:
    lines = [
        "# Figure 9 DiffW-like qualitative candidates",
        "",
        "## Used samples",
        "",
        "- Selection source: reused the first three IDs from the existing thesis qualitative RGB-refinement figure.",
        "- Sample order is identical in the 6-column and 7-column candidates.",
        f"- Sample IDs: {', '.join(sample_ids)}",
        "",
        "## Source paths",
        "",
        f"- Cover / reference directory: `{args.gt_dir}`",
        f"- DiffW-like output directory: `{args.diffw_dir}`",
        f"- O256-Base output directory: `{args.base_dir}`",
        f"- O256-RGB output directory: `{args.refined_dir}`",
        "",
        "| Sample ID | Cover / Ref. | DiffW-like | O256-Base | O256-RGB |",
        "| --- | --- | --- | --- | --- |",
    ]
    for stem in sample_ids:
        row = assets[stem]
        lines.append(
            f"| {stem} | `{row['cover']}` | `{row['diffw']}` | `{row['base']}` | `{row['refined']}` |"
        )

    lines.extend(
        [
            "",
            "## Difference-map computation",
            "",
            f"- Difference maps are computed as `clip(abs(method_output - Cover / Ref.) * {args.diff_gain:g}, 0, 255)` in RGB image space.",
            "- If an input image size differs from the cover image size, it is resized to the cover size before the absolute difference is computed.",
            "- The same gain and computation rule are used in both candidates.",
            "",
            "## Output candidates",
            "",
            f"- 6-column version: `{figure6}`",
            "  - Columns: Cover / Ref., DiffW-like, O256-Base, O256-RGB, Base diff ×12, Refined diff ×12.",
            "  - It omits the DiffW-like difference map to keep the figure readable.",
            f"- 7-column version: `{figure7}`",
            "  - Columns: Cover / Ref., DiffW-like, O256-Base, O256-RGB, DiffW-like diff ×12, Base diff ×12, Refined diff ×12.",
            "  - It fully compares the visible perturbation maps for all three methods.",
            "",
            "## Recommendation",
            "",
            "- For an A4 thesis page, the 6-column version is the safer main-text figure because each image cell is larger and the comparison remains readable after insertion into the document.",
            "- The 7-column version is useful for discussion with the advisor or as an appendix figure, but it is visibly denser. It is not unusable, but the extra DiffW-like diff column makes each cell smaller and may feel crowded in single-column A4 layout.",
        ]
    )
    (output_dir / "figure9_candidate_generation_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = collect_assets(args)

    cols6 = [
        ("Cover / Ref.", "cover"),
        ("DiffW-like", "diffw"),
        ("O256-Base", "base"),
        ("O256-RGB", "refined"),
        ("Base diff ×12", "base_diff"),
        ("Refined diff ×12", "refined_diff"),
    ]
    cols7 = [
        ("Cover / Ref.", "cover"),
        ("DiffW-like", "diffw"),
        ("O256-Base", "base"),
        ("O256-RGB", "refined"),
        ("DiffW-like diff ×12", "diffw_diff"),
        ("Base diff ×12", "base_diff"),
        ("Refined diff ×12", "refined_diff"),
    ]

    figure6 = output_dir / "figure9_candidate_6col_diffw_like.png"
    figure7 = output_dir / "figure9_candidate_7col_diffw_like.png"
    build_grid(args.sample_ids, assets, figure6, cols6, args.diff_gain, cell=372, dpi=args.dpi)
    build_grid(args.sample_ids, assets, figure7, cols7, args.diff_gain, cell=320, dpi=args.dpi)
    write_summary(output_dir, args.sample_ids, assets, args, figure6, figure7)

    print(f"6-column figure: {figure6}")
    print(f"7-column figure: {figure7}")
    print(f"summary: {output_dir / 'figure9_candidate_generation_summary.md'}")


if __name__ == "__main__":
    main()
