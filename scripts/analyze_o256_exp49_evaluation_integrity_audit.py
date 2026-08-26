#!/usr/bin/env python3
import argparse
import csv
import json
import math
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

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key, dict2namespace(value) if isinstance(value, dict) else value)
    return namespace


def parse_sample(spec):
    parts = spec.split("|")
    if len(parts) != 7:
        raise ValueError(
            "Sample spec must be name|kind|config|resume|image_dir|gt_message_dir|gt_image_dir"
        )
    name, kind, config, resume, image_dir, gt_message_dir, gt_image_dir = parts
    return {
        "name": name,
        "kind": kind,
        "config": config,
        "resume": Path(resume),
        "image_dir": Path(image_dir),
        "gt_message_dir": Path(gt_message_dir),
        "gt_image_dir": Path(gt_image_dir),
    }


def load_config(name):
    with (ROOT / "configs" / name).open("r", encoding="utf-8") as handle:
        return dict2namespace(yaml.safe_load(handle))


def list_images(directory):
    root = Path(directory)
    return {
        path.stem: path
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def message_map(directory):
    root = Path(directory)
    return {
        path.name.removesuffix("_message.npy"): path
        for path in sorted(root.glob("*_message.npy"))
        if path.is_file()
    }


def psnr(pred_path, gt_path):
    pred = np.asarray(Image.open(pred_path).convert("RGB"), dtype=np.float64)
    gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred_path} {pred.shape} vs {gt_path} {gt.shape}")
    mse = float(np.mean((pred - gt) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def get_img_tensor(img_path, device, image_size):
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    return data_transform(tf(img).unsqueeze(0).to(device))


def load_decoder(cache, sample, device):
    key = (sample["config"], str(sample["resume"]))
    if key in cache:
        return cache[key]
    config = load_config(sample["config"])
    config.device = device
    dummy_args = argparse.Namespace(resume=str(sample["resume"]))
    diffusion = DenoisingDiffusion(dummy_args, config)
    diffusion.load_ddm_ckpt(str(sample["resume"]), ema=True)
    diffusion.model.eval()
    decoder = diffusion.model.module.decoder
    cache[key] = (diffusion, decoder, int(config.data.image_size))
    return cache[key]


def decode_image(diffusion, decoder, img_path, device, image_size):
    img_tensor = get_img_tensor(img_path, device, image_size)
    if diffusion.latent_enabled:
        img_tensor = diffusion.encode_images(img_tensor)
    with torch.no_grad():
        decoded = decoder(img_tensor)
    return decoded.detach().cpu().numpy().round().clip(0, 1).reshape(-1).astype(np.int8)


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def audit_psnr(sample, max_images):
    pred_map = list_images(sample["image_dir"])
    gt_map = list_images(sample["gt_image_dir"])
    common = sorted(set(pred_map) & set(gt_map))
    if max_images:
        common = common[:max_images]
    stem_values = [psnr(pred_map[stem], gt_map[stem]) for stem in common]

    pred_stems_sorted = sorted(pred_map)
    gt_stems_sorted = sorted(gt_map)
    sorted_pair_count = min(len(pred_stems_sorted), len(gt_stems_sorted))
    if max_images:
        sorted_pair_count = min(sorted_pair_count, max_images)
    sorted_values = [
        psnr(pred_map[pred_stems_sorted[i]], gt_map[gt_stems_sorted[i]])
        for i in range(sorted_pair_count)
    ]

    shuffled_values = []
    if len(common) > 1:
        shifted = common[1:] + common[:1]
        shuffled_values = [psnr(pred_map[p], gt_map[g]) for p, g in zip(common, shifted)]

    return {
        "pred_count": len(pred_map),
        "gt_count": len(gt_map),
        "psnr_common_count": len(common),
        "stem_sets_equal": set(pred_map) == set(gt_map),
        "sorted_stem_order_equal": pred_stems_sorted[:sorted_pair_count] == gt_stems_sorted[:sorted_pair_count],
        "psnr_stem_matched": mean(stem_values),
        "psnr_sorted_order": mean(sorted_values),
        "psnr_wrong_gt_shifted": mean(shuffled_values),
        "psnr_sorted_minus_stem": mean(sorted_values) - mean(stem_values),
    }


def audit_ba(sample, device, decoder_cache, max_images):
    diffusion, decoder, image_size = load_decoder(decoder_cache, sample, device)
    pred_map = list_images(sample["image_dir"])
    msg_map = message_map(sample["gt_message_dir"])
    common = sorted(set(pred_map) & set(msg_map))
    if max_images:
        common = common[:max_images]
    if not common:
        raise RuntimeError(f"No common image/message stems for {sample['name']}")

    preds = []
    gts = []
    for idx, stem in enumerate(common, start=1):
        preds.append(decode_image(diffusion, decoder, pred_map[stem], device, image_size))
        gts.append(np.load(msg_map[stem]).reshape(-1).astype(np.int8))
        if idx % 100 == 0:
            print(f"[{sample['name']}] decoded {idx}/{len(common)} images", flush=True)

    pred = np.stack(preds).astype(np.int8)
    gt = np.stack(gts).astype(np.int8)
    if pred.shape != gt.shape:
        raise RuntimeError(f"decoded/message shape mismatch for {sample['name']}: {pred.shape} vs {gt.shape}")

    correct_ba = float(np.mean(pred == gt) * 100.0)
    shifted_gt = np.roll(gt, shift=1, axis=0)
    shuffled_ba = float(np.mean(pred == shifted_gt) * 100.0)
    all_zero_ba = float(np.mean(pred == 0) * 100.0)
    all_one_ba = float(np.mean(pred == 1) * 100.0)
    return {
        "message_count": len(msg_map),
        "ba_common_count": len(common),
        "message_stem_sets_equal": set(pred_map) == set(msg_map),
        "ba_correct_message": correct_ba,
        "ba_wrong_message_shifted": shuffled_ba,
        "ba_vs_all_zero": all_zero_ba,
        "ba_vs_all_one": all_one_ba,
        "ba_control_gap": correct_ba - shuffled_ba,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Audit O256 PSNR/BA evaluation integrity.")
    parser.add_argument("--sample", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    samples = [parse_sample(spec) for spec in args.sample]
    for sample in samples:
        for key in ("resume", "image_dir", "gt_message_dir", "gt_image_dir"):
            if not sample[key].exists():
                raise FileNotFoundError(f"Missing {key} for {sample['name']}: {sample[key]}")
        if not (ROOT / "configs" / sample["config"]).exists():
            raise FileNotFoundError(f"Missing config for {sample['name']}: {sample['config']}")
        pred_count = len(list_images(sample["image_dir"]))
        msg_count = len(message_map(sample["gt_message_dir"]))
        gt_count = len(list_images(sample["gt_image_dir"]))
        print(f"[check] {sample['name']}: pred={pred_count} messages={msg_count} gt={gt_count}", flush=True)
    if args.check_only:
        print("Preflight passed.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    decoder_cache = {}
    rows = []
    for sample in samples:
        print(f"Auditing {sample['name']}", flush=True)
        psnr_row = audit_psnr(sample, args.max_images)
        ba_row = audit_ba(sample, device, decoder_cache, args.max_images)
        rows.append(
            {
                "sample": sample["name"],
                "kind": sample["kind"],
                **psnr_row,
                **ba_row,
            }
        )

    fieldnames = [
        "sample",
        "kind",
        "pred_count",
        "gt_count",
        "message_count",
        "psnr_common_count",
        "ba_common_count",
        "stem_sets_equal",
        "message_stem_sets_equal",
        "sorted_stem_order_equal",
        "psnr_stem_matched",
        "psnr_sorted_order",
        "psnr_sorted_minus_stem",
        "psnr_wrong_gt_shifted",
        "ba_correct_message",
        "ba_wrong_message_shifted",
        "ba_control_gap",
        "ba_vs_all_zero",
        "ba_vs_all_one",
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "evaluation_integrity_audit.csv", rows, fieldnames)
    (output_dir / "evaluation_integrity_audit.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# O256 exp49 evaluation integrity audit",
        "",
        "## Purpose",
        "",
        "- Verify PSNR is evaluated by matching output image stems to locked GT image stems.",
        "- Compare old sorted-order PSNR pairing against explicit stem-matched PSNR.",
        "- Run BA negative controls by intentionally pairing decoded messages with shifted wrong messages.",
        "- Check whether high BA disappears under wrong-message controls.",
        "",
        "## Audit Table",
        "",
        "| Sample | Kind | Count | Stem OK | PSNR stem | PSNR sorted-stem diff | PSNR wrong GT | BA correct | BA wrong msg | BA gap |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        stem_ok = bool(row["stem_sets_equal"]) and bool(row["message_stem_sets_equal"])
        lines.append(
            f"| {row['sample']} | {row['kind']} | {row['ba_common_count']} | {stem_ok} | "
            f"{row['psnr_stem_matched']:.4f} | {row['psnr_sorted_minus_stem']:.6f} | "
            f"{row['psnr_wrong_gt_shifted']:.4f} | {row['ba_correct_message']:.2f} | "
            f"{row['ba_wrong_message_shifted']:.2f} | {row['ba_control_gap']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `PSNR sorted-stem diff` should be near 0. If not, sorted-order pairing is unsafe.",
            "- `PSNR wrong GT` should be much lower than the correct PSNR.",
            "- `BA wrong msg` should be near random-message accuracy, roughly around 50%.",
            "- A large `BA gap` means the reported BA depends on the correct message pairing, not a trivial decoder bias.",
            "",
            "## Outputs",
            "",
            "- `evaluation_integrity_audit.csv`",
            "- `evaluation_integrity_audit.json`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary md: {output_dir / 'summary.md'}")
    print(f"Audit csv: {output_dir / 'evaluation_integrity_audit.csv'}")


if __name__ == "__main__":
    main()
