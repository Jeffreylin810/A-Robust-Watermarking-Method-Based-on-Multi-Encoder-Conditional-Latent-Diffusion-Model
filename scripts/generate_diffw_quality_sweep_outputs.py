#!/usr/bin/env python3
"""Generate post-hoc gamma-scaled DiffW-like outputs for quality/robustness sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_results_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        required=True,
        help="Gamma values used in x_gamma = x_c + gamma * (x_w - x_c).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def gamma_tag(value: float) -> str:
    return f"gamma_{value:.2f}".replace(".", "p")


def list_images(directory: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            mapping[path.stem] = path
    return mapping


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def main() -> None:
    args = parse_args()
    source_results_dir = Path(args.source_results_dir)
    gt_dir = Path(args.gt_dir)
    output_root = Path(args.output_root)
    scaled_root = output_root / "scaled_outputs"
    scaled_root.mkdir(parents=True, exist_ok=True)

    src_map = list_images(source_results_dir)
    gt_map = list_images(gt_dir)
    stems = sorted(set(src_map) & set(gt_map))
    if not stems:
        raise RuntimeError("No matched images between source_results_dir and gt_dir.")

    manifest_rows: list[dict[str, str]] = []
    for gamma in args.gammas:
        tag = gamma_tag(gamma)
        results_dir = scaled_root / tag / "CoCo" / "CoCo"
        results_dir.mkdir(parents=True, exist_ok=True)
        generated = 0
        for stem in stems:
            src_path = src_map[stem]
            gt_path = gt_map[stem]
            out_path = results_dir / f"{stem}.png"
            message_src = source_results_dir / f"{stem}_message.npy"
            message_out = results_dir / f"{stem}_message.npy"

            if out_path.exists() and message_out.exists() and not args.overwrite:
                continue

            src_arr = load_rgb(src_path)
            gt_arr = load_rgb(gt_path)
            if src_arr.shape != gt_arr.shape:
                gt_img = Image.fromarray(gt_arr.astype(np.uint8), mode="RGB")
                gt_img = gt_img.resize((src_arr.shape[1], src_arr.shape[0]), Image.Resampling.LANCZOS)
                gt_arr = np.asarray(gt_img, dtype=np.float32)

            scaled = np.clip(gt_arr + gamma * (src_arr - gt_arr), 0.0, 255.0)
            save_rgb(out_path, scaled)
            if message_out.exists() or message_out.is_symlink():
                message_out.unlink()
            if message_src.exists():
                message_out.symlink_to(message_src)
            generated += 1

        metadata = {
            "gamma": gamma,
            "tag": tag,
            "count": len(stems),
            "generated_or_overwritten_images": generated,
            "source_results_dir": str(source_results_dir),
            "gt_dir": str(gt_dir),
            "results_dir": str(results_dir),
        }
        (scaled_root / tag / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        manifest_rows.append(
            {
                "gamma": f"{gamma:.2f}",
                "tag": tag,
                "results_dir": str(results_dir),
                "count": str(len(stems)),
            }
        )

    with (output_root / "gamma_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gamma", "tag", "results_dir", "count"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Generated gamma sweep outputs under: {scaled_root}")
    print(f"Manifest: {output_root / 'gamma_manifest.csv'}")


if __name__ == "__main__":
    main()
