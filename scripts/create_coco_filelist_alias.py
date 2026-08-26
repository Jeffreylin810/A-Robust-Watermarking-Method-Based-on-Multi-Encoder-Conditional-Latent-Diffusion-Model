#!/usr/bin/env python3
"""Create a CoCo-style alias dataset from an image directory + explicit file list.

The target layout matches the local O256/CoCo loader expectation:

  output_root/data/CoCo/{train,val}/{input,gt}/<filename>

For alias-style datasets used in formal support experiments, we intentionally
point both `input` and `gt` to the same RGB image files, and mirror the same
selected set into both `train` and `val`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--filelist", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--resized_cache_dir")
    parser.add_argument("--image_size", type=int, default=256)
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


def ensure_resized_copy(src: Path, dst: Path, image_size: int) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        save_kwargs = {"quality": 95} if dst.suffix.lower() in {".jpg", ".jpeg", ".webp"} else {}
        image.save(dst, **save_kwargs)
    return dst


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    filelist = Path(args.filelist)
    output_root = Path(args.output_root)
    resized_cache_dir = Path(args.resized_cache_dir) if args.resized_cache_dir else None

    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if not filelist.is_file():
        raise FileNotFoundError(filelist)

    names = [line.strip() for line in filelist.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise RuntimeError(f"No filenames found in {filelist}")

    for split in ("train", "val"):
        for subdir in ("input", "gt"):
            (output_root / "data" / "CoCo" / split / subdir).mkdir(parents=True, exist_ok=True)

    records = []
    for name in names:
        src = image_dir / name
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
            raise FileNotFoundError(f"Missing source image: {src}")
        link_source = src
        if resized_cache_dir is not None:
            link_source = ensure_resized_copy(src, resized_cache_dir / name, args.image_size)
        for split in ("train", "val"):
            safe_symlink(link_source, output_root / "data" / "CoCo" / split / "input" / name, args.overwrite_links)
            safe_symlink(link_source, output_root / "data" / "CoCo" / split / "gt" / name, args.overwrite_links)
        record = {"name": name, "src": str(src.resolve()), "linked": str(link_source.resolve())}
        records.append(record)

    manifest = {
        "image_dir": str(image_dir.resolve()),
        "filelist": str(filelist.resolve()),
        "output_root": str(output_root.resolve()),
        "resized_cache_dir": str(resized_cache_dir.resolve()) if resized_cache_dir else None,
        "image_size": args.image_size,
        "num_images": len(records),
        "records": records,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"created alias dataset: {output_root}")
    print(f"image_dir: {image_dir}")
    print(f"filelist: {filelist}")
    print(f"num_images: {len(records)}")


if __name__ == "__main__":
    main()
