#!/usr/bin/env python3
"""Create the locked COCO256 final-test split used only for thesis reporting.

The existing COCO val-based eval1000 has been used repeatedly during method
development, so it is treated as dev1000.  This script samples a fresh,
deterministic set from official COCO test2017, resizes it with the same PIL
Lanczos policy, and builds the local CoCo loader layout without evaluating any
model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.name in images:
                raise RuntimeError(f"Duplicate filename under {root}: {path.name}")
            images[path.name] = path.resolve()
    if not images:
        raise RuntimeError(f"No images found under {root}")
    return images


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {dst}")
    dst.symlink_to(os.path.relpath(src, dst.parent))


def resize_image(src: Path, dst: Path, image_size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite: {dst}")
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        image.save(dst, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-test-dir",
        default="/home/jeffrey/projects/DataSet/CoCo_Dataset_2017/test2017",
    )
    parser.add_argument(
        "--paper-data-root",
        default="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final",
    )
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_root = Path(args.paper_data_root).resolve()
    raw_test_dir = Path(args.raw_test_dir).resolve()
    output_root = paper_root / "train10000_finaltest1000_locked"
    resized_root = paper_root / "source_images_256" / "test2017_finaltest1000_locked"
    manifest_path = paper_root / "manifests" / "finaltest1000_locked.txt"

    if output_root.exists() or resized_root.exists() or manifest_path.exists():
        raise FileExistsError(
            "Locked final-test output already exists. Refusing to recreate or overwrite it."
        )

    test_images = collect_images(raw_test_dir)
    names = sorted(test_images)
    random.Random(args.seed).shuffle(names)
    selected = names[: args.count]
    if len(selected) != args.count:
        raise RuntimeError(f"Requested {args.count} images, found {len(selected)}")

    train_manifest = paper_root / "manifests" / "train10000.txt"
    dev_manifest = paper_root / "manifests" / "eval_unseen1000.txt"
    train_names = read_names(train_manifest)
    dev_names = read_names(dev_manifest)
    overlap = set(selected) & (set(train_names) | set(dev_names))
    if overlap:
        raise RuntimeError(f"Final-test filenames overlap existing splits: {sorted(overlap)[:10]}")

    resized: dict[str, Path] = {}
    for index, name in enumerate(selected, start=1):
        destination = resized_root / name
        resize_image(test_images[name], destination, args.image_size)
        resized[name] = destination.resolve()
        if index % 100 == 0 or index == len(selected):
            print(f"resized finaltest1000: {index}/{len(selected)}")

    manifest_path.write_text("\n".join(selected) + "\n", encoding="utf-8")

    train_resized_root = paper_root / "source_images_256" / "train2017_selected"
    for split, names_for_split, sources in (
        ("train", train_names, {name: train_resized_root / name for name in train_names}),
        ("val", selected, resized),
    ):
        for subdir in ("input", "gt"):
            for name in names_for_split:
                source = sources[name].resolve(strict=True)
                relative_symlink(
                    source,
                    output_root / "data" / "CoCo" / split / subdir / name,
                )

    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "train.txt").write_text(
        "\n".join(train_names) + "\n", encoding="utf-8"
    )
    (manifests_dir / "val.txt").write_text(
        "\n".join(selected) + "\n", encoding="utf-8"
    )

    metadata = {
        "role": "locked_final_test_only",
        "warning": "Do not use this split for tuning, checkpoint selection, or method design.",
        "source": "official COCO test2017 images; annotations are not needed for image watermarking",
        "raw_test_dir": str(raw_test_dir),
        "seed": args.seed,
        "image_size": args.image_size,
        "train_count": len(train_names),
        "final_test_count": len(selected),
        "resize": "PIL RGB direct 256x256 resize with Lanczos; JPEG quality 95",
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": sha256(train_manifest),
        "dev_manifest": str(dev_manifest),
        "dev_manifest_sha256": sha256(dev_manifest),
        "final_test_manifest": str(manifest_path),
        "final_test_manifest_sha256": sha256(manifest_path),
        "overlap_with_train_or_dev": 0,
    }
    (output_root / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "DO_NOT_USE_FOR_TUNING.md").write_text(
        "# Locked Final Test\n\n"
        "This split is reserved for the frozen Chapter 4 protocol. Do not use it "
        "for hyperparameter tuning, checkpoint selection, early stopping, or "
        "method redesign. If the result is poor, report it without changing the "
        "method based on this split.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
