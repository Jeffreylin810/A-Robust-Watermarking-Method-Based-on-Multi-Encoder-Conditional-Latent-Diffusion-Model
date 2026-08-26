#!/usr/bin/env python3
"""Create a small CoCo-style dataset root by symlinking a source split.

The local DiffW eval loader always restores the `val` split.  For formal RGB
refinement we need to generate watermarked outputs from training images without
changing the loader behavior globally, so this script creates a dataset alias
where both `train` and `val` point to a fixed subset of the source train split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", required=True, help="Original dataset root.")
    parser.add_argument("--source_split", default="train", choices=("train", "val"))
    parser.add_argument("--output_root", required=True, help="Alias dataset root to create.")
    parser.add_argument("--num_images", type=int, default=2000)
    parser.add_argument("--overwrite_links", action="store_true")
    return parser.parse_args()


def safe_symlink(src: Path, dst: Path, overwrite_links: bool) -> None:
    target = src.resolve(strict=True)
    if dst.is_symlink():
        if dst.resolve(strict=True) == target:
            return
        if overwrite_links:
            dst.unlink()
        else:
            raise FileExistsError(f"Different symlink already exists: {dst}")
    elif dst.exists():
        raise FileExistsError(f"Non-symlink path already exists: {dst}")
    os.symlink(target, dst)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)

    src_input = source_root / "data" / "CoCo" / args.source_split / "input"
    src_gt = source_root / "data" / "CoCo" / args.source_split / "gt"
    if not src_input.is_dir() or not src_gt.is_dir():
        raise FileNotFoundError(f"Missing source split dirs: {src_input}, {src_gt}")

    names = sorted(p.name for p in src_input.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.num_images > 0:
        names = names[: args.num_images]
    if not names:
        raise RuntimeError(f"No source images found in {src_input}")

    for split in ("train", "val"):
        for subdir in ("input", "gt"):
            (output_root / "data" / "CoCo" / split / subdir).mkdir(parents=True, exist_ok=True)

    records = []
    for name in names:
        src_in = src_input / name
        src_gt_path = src_gt / name
        if not src_gt_path.exists():
            raise FileNotFoundError(f"Missing gt image for {name}: {src_gt_path}")
        for split in ("train", "val"):
            safe_symlink(src_in, output_root / "data" / "CoCo" / split / "input" / name, args.overwrite_links)
            safe_symlink(src_gt_path, output_root / "data" / "CoCo" / split / "gt" / name, args.overwrite_links)
        records.append({"name": name, "input": str(src_in.resolve()), "gt": str(src_gt_path.resolve())})

    manifest = {
        "source_root": str(source_root),
        "source_split": args.source_split,
        "output_root": str(output_root),
        "num_images": len(records),
        "records": records,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"created alias dataset: {output_root}")
    print(f"source split: {args.source_split}")
    print(f"num_images: {len(records)}")


if __name__ == "__main__":
    main()
