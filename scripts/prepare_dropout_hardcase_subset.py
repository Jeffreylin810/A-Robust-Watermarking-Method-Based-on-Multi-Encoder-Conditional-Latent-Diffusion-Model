#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ddm import DenoisingDiffusion, data_transform
from noise_layers.crop import Dropout

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key, dict2namespace(value) if isinstance(value, dict) else value)
    return namespace


def load_config(name):
    path = ROOT / "configs" / name
    with path.open("r", encoding="utf-8") as f:
        return dict2namespace(yaml.safe_load(f))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_norm_tensor(path, device):
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)


def build_diffusion(config, resume, device):
    config.device = device
    dummy_args = argparse.Namespace(resume=resume)
    diffusion = DenoisingDiffusion(dummy_args, config)
    diffusion.load_ddm_ckpt(resume, ema=True)
    diffusion.model.eval()
    return diffusion


def symlink_force(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(Path(src).resolve(), dst)


def list_image_paths(directory):
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and p.exists() and (p.is_file() or p.is_symlink())
    )


def main():
    parser = argparse.ArgumentParser(
        description="Select train-side Dropout r03 hard cases and create a refiner subset."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--watermarked_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dropout_prob", type=float, default=0.3)
    parser.add_argument("--threshold_ba", type=float, default=90.0)
    parser.add_argument("--min_hard", type=int, default=512)
    parser.add_argument("--max_hard", type=int, default=1000)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=62)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    diffusion = build_diffusion(config, args.resume, device)
    decoder = diffusion.model.module.decoder if hasattr(diffusion.model, "module") else diffusion.model.decoder
    dropout = Dropout(args.dropout_prob).to(device)
    dropout.eval()

    watermarked_dir = Path(args.watermarked_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    subset_wm_dir = output_dir / "hard_subset" / "watermarked"
    subset_gt_dir = output_dir / "hard_subset" / "gt"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_paths(watermarked_dir)
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No watermarked images found: {watermarked_dir}")

    gt_by_stem = {p.stem: p for p in list_image_paths(gt_dir)}
    if not gt_by_stem:
        raise RuntimeError(f"No GT images found: {gt_dir}")

    rows = []
    missing_gt = 0
    missing_message = 0
    for index, wm_path in enumerate(image_paths, start=1):
        gt_path = gt_by_stem.get(wm_path.stem)
        msg_path = watermarked_dir / f"{wm_path.stem}_message.npy"
        if gt_path is None:
            missing_gt += 1
            continue
        if not msg_path.exists():
            missing_message += 1
            continue

        wm = load_norm_tensor(wm_path, device)
        cover = load_norm_tensor(gt_path, device)
        with torch.no_grad():
            attacked = dropout([wm, cover])
            decode_input = attacked
            if diffusion.latent_enabled:
                decode_input = diffusion.encode_images(decode_input)
            decoded = decoder(decode_input)
        pred = decoded.detach().cpu().numpy().round().clip(0, 1).reshape(-1)
        gt_msg = np.load(msg_path).reshape(-1)
        ba = float((1.0 - np.mean(np.abs(pred - gt_msg))) * 100.0)
        rows.append(
            {
                "image": wm_path.stem,
                "filename": wm_path.name,
                "ba": ba,
                "error_bits": int(round((100.0 - ba) / 100.0 * gt_msg.shape[0])),
                "message_path": str(msg_path),
                "gt_path": str(gt_path),
                "watermarked_path": str(wm_path),
            }
        )
        if index % 500 == 0:
            print(f"Decoded train-side dropout hard cases: {index}/{len(image_paths)}", flush=True)

    if not rows:
        raise RuntimeError(
            "No valid train images/messages processed. "
            f"watermarked_images={len(image_paths)} gt_images={len(gt_by_stem)} "
            f"missing_gt={missing_gt} missing_message={missing_message}"
        )

    rows = sorted(rows, key=lambda row: (row["ba"], row["image"]))
    below_threshold = [row for row in rows if row["ba"] < args.threshold_ba]
    if len(below_threshold) >= args.min_hard:
        selected = below_threshold[: args.max_hard]
    else:
        selected = rows[: min(args.min_hard, len(rows))]
    selected_names = {row["image"] for row in selected}

    for directory in (subset_wm_dir, subset_gt_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.iterdir():
            if old.is_symlink() or old.is_file():
                old.unlink()

    for row in selected:
        wm_path = Path(row["watermarked_path"])
        gt_path = Path(row["gt_path"])
        msg_path = Path(row["message_path"])
        symlink_force(wm_path, subset_wm_dir / wm_path.name)
        symlink_force(msg_path, subset_wm_dir / msg_path.name)
        symlink_force(gt_path, subset_gt_dir / gt_path.name)

    rows_with_selected = [
        {**row, "selected": row["image"] in selected_names}
        for row in rows
    ]
    with (output_dir / "train_dropout_image_ba.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "filename", "ba", "error_bits", "selected", "watermarked_path", "gt_path", "message_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_with_selected)

    with (output_dir / "hard_cases.txt").open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(f"{row['image']}\n")

    summary = {
        "config": args.config,
        "resume": args.resume,
        "watermarked_dir": str(watermarked_dir),
        "gt_dir": str(gt_dir),
        "dropout_prob": args.dropout_prob,
        "threshold_ba": args.threshold_ba,
        "min_hard": args.min_hard,
        "max_hard": args.max_hard,
        "processed_images": len(rows),
        "gt_index_images": len(gt_by_stem),
        "missing_gt_images": missing_gt,
        "missing_message_files": missing_message,
        "below_threshold_images": len(below_threshold),
        "selected_images": len(selected),
        "selected_avg_ba": float(np.mean([row["ba"] for row in selected])),
        "selected_max_ba": float(max(row["ba"] for row in selected)),
        "selected_min_ba": float(min(row["ba"] for row in selected)),
        "subset_watermarked_dir": str(subset_wm_dir),
        "subset_gt_dir": str(subset_gt_dir),
    }
    (output_dir / "hardcase_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        "# Train-side Dropout r03 hard-case subset",
        "",
        f"- processed_images: `{summary['processed_images']}`",
        f"- gt_index_images: `{summary['gt_index_images']}`",
        f"- missing_gt_images: `{summary['missing_gt_images']}`",
        f"- missing_message_files: `{summary['missing_message_files']}`",
        f"- below_threshold_images: `{summary['below_threshold_images']}`",
        f"- selected_images: `{summary['selected_images']}`",
        f"- selected_avg_ba: `{summary['selected_avg_ba']:.2f}`",
        f"- selected_ba_range: `{summary['selected_min_ba']:.2f}` to `{summary['selected_max_ba']:.2f}`",
        f"- subset_watermarked_dir: `{subset_wm_dir}`",
        f"- subset_gt_dir: `{subset_gt_dir}`",
    ]
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Hardcase summary: {output_dir / 'summary.md'}")
    print(f"Hardcase watermarked subset: {subset_wm_dir}")
    print(f"Hardcase GT subset: {subset_gt_dir}")


if __name__ == "__main__":
    main()
