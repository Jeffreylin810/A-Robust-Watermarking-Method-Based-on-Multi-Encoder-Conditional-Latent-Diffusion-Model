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


def load_config(name):
    with (ROOT / "configs" / name).open("r", encoding="utf-8") as handle:
        return dict2namespace(yaml.safe_load(handle))


def parse_sample(spec):
    parts = spec.split("|")
    if len(parts) != 8:
        raise ValueError(
            "Sample spec must be "
            "name|seed|stage|condition|config|resume|image_dir|gt_message_dir"
        )
    name, seed, stage, condition, config, resume, image_dir, gt_message_dir = parts
    return {
        "name": name,
        "seed": seed,
        "stage": stage,
        "condition": condition,
        "config": config,
        "resume": Path(resume),
        "image_dir": Path(image_dir),
        "gt_message_dir": Path(gt_message_dir),
    }


def parse_transition(spec):
    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError("Transition spec must be name|left_sample|right_sample")
    return {"name": parts[0], "left": parts[1], "right": parts[2]}


def image_paths(directory):
    root = Path(directory)
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def message_path(gt_message_dir, image_stem):
    return Path(gt_message_dir) / f"{image_stem}_message.npy"


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


def percentile(values, q):
    values = np.asarray(values, dtype=np.float64)
    return float(np.percentile(values, q)) if values.size else float("nan")


def jaccard(left, right):
    union = set(left) | set(right)
    return float(len(set(left) & set(right)) / len(union)) if union else 1.0


def corr(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size == 0 or right.size == 0 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def diagnose_sample(sample, device, decoder_cache, max_images):
    diffusion, decoder, image_size = load_decoder(decoder_cache, sample, device)
    paths = image_paths(sample["image_dir"])
    if max_images:
        paths = paths[:max_images]
    if not paths:
        raise RuntimeError(f"No images found for {sample['name']}: {sample['image_dir']}")

    names = []
    gt_rows = []
    pred_rows = []
    error_rows = []
    for idx, img_path in enumerate(paths, start=1):
        gt_path = message_path(sample["gt_message_dir"], img_path.stem)
        if not gt_path.exists():
            continue
        gt = np.load(gt_path).reshape(-1).astype(np.int8)
        pred = decode_image(diffusion, decoder, img_path, device, image_size)
        if gt.shape != pred.shape:
            raise RuntimeError(
                f"Message length mismatch for {sample['name']} {img_path.name}: "
                f"gt={gt.shape[0]} pred={pred.shape[0]}"
            )
        names.append(img_path.stem)
        gt_rows.append(gt)
        pred_rows.append(pred)
        error_rows.append(pred != gt)
        if idx % 100 == 0:
            print(f"[{sample['name']}] decoded {idx}/{len(paths)}", flush=True)

    if not error_rows:
        raise RuntimeError(f"No valid image/message pairs for {sample['name']}")

    errors = np.stack(error_rows).astype(bool)
    gt = np.stack(gt_rows).astype(np.int8)
    pred = np.stack(pred_rows).astype(np.int8)
    per_image_ba = (1.0 - errors.mean(axis=1)) * 100.0
    per_bit_ba = (1.0 - errors.mean(axis=0)) * 100.0

    return {
        "sample": sample,
        "names": names,
        "gt": gt,
        "pred": pred,
        "errors": errors,
        "per_image_ba": per_image_ba,
        "per_bit_ba": per_bit_ba,
    }


def sample_summary(data):
    per_image = data["per_image_ba"]
    per_bit = data["per_bit_ba"]
    weak8 = np.argsort(per_bit)[:8]
    return {
        "sample": data["sample"]["name"],
        "seed": data["sample"]["seed"],
        "stage": data["sample"]["stage"],
        "condition": data["sample"]["condition"],
        "images": len(data["names"]),
        "avg_ba": float(np.mean(per_image)),
        "min_image_ba": float(np.min(per_image)),
        "p05_image_ba": percentile(per_image, 5),
        "p10_image_ba": percentile(per_image, 10),
        "hard_lt90_images": int(np.sum(per_image < 90.0)),
        "hard_lt80_images": int(np.sum(per_image < 80.0)),
        "min_bit": int(np.argmin(per_bit)),
        "min_bit_ba": float(np.min(per_bit)),
        "weak_bits_top8": " ".join(str(int(bit)) for bit in weak8),
        "weak_bits_top8_ba": " ".join(f"{per_bit[bit]:.2f}" for bit in weak8),
    }


def compare_transition(name, left, right, threshold):
    left_idx = {image: idx for idx, image in enumerate(left["names"])}
    right_idx = {image: idx for idx, image in enumerate(right["names"])}
    common = sorted(set(left_idx) & set(right_idx))
    if not common:
        raise RuntimeError(f"No common images for transition {name}")

    li = np.array([left_idx[image] for image in common], dtype=np.int64)
    ri = np.array([right_idx[image] for image in common], dtype=np.int64)
    le = left["errors"][li]
    re = right["errors"][ri]
    lba = left["per_image_ba"][li]
    rba = right["per_image_ba"][ri]
    lgt = left["gt"][li]
    rgt = right["gt"][ri]

    left_hard = set(np.array(common)[lba < threshold])
    right_hard = set(np.array(common)[rba < threshold])
    recovered = sorted(left_hard - right_hard)
    newly_failed = sorted(right_hard - left_hard)
    persistent = sorted(left_hard & right_hard)

    damaged = np.logical_and(~le, re)
    repaired = np.logical_and(le, ~re)
    both_wrong = np.logical_and(le, re)
    both_correct = np.logical_and(~le, ~re)
    left_weak8 = set(int(bit) for bit in np.argsort(left["per_bit_ba"])[:8])
    right_weak8 = set(int(bit) for bit in np.argsort(right["per_bit_ba"])[:8])
    left_weak16 = set(int(bit) for bit in np.argsort(left["per_bit_ba"])[:16])
    right_weak16 = set(int(bit) for bit in np.argsort(right["per_bit_ba"])[:16])

    error_inter = int(np.logical_and(le, re).sum())
    error_union = int(np.logical_or(le, re).sum())
    return {
        "name": name,
        "left": left["sample"]["name"],
        "right": right["sample"]["name"],
        "common_images": len(common),
        "gt_equal_rate": float(np.mean(np.all(lgt == rgt, axis=1)) * 100.0),
        "left_avg_ba": float(np.mean(lba)),
        "right_avg_ba": float(np.mean(rba)),
        "delta_avg_ba": float(np.mean(rba) - np.mean(lba)),
        "left_hard_lt90": len(left_hard),
        "right_hard_lt90": len(right_hard),
        "recovered_to_ge90": len(recovered),
        "newly_below90": len(newly_failed),
        "persistent_below90": len(persistent),
        "hard_jaccard": jaccard(left_hard, right_hard),
        "bit_ba_corr": corr(left["per_bit_ba"], right["per_bit_ba"]),
        "image_ba_corr": corr(lba, rba),
        "weak_bit_overlap_top8": len(left_weak8 & right_weak8),
        "weak_bit_jaccard_top8": jaccard(left_weak8, right_weak8),
        "weak_bit_overlap_top16": len(left_weak16 & right_weak16),
        "weak_bit_jaccard_top16": jaccard(left_weak16, right_weak16),
        "error_event_jaccard": float(error_inter / error_union) if error_union else 1.0,
        "damaged_bit_events": int(damaged.sum()),
        "repaired_bit_events": int(repaired.sum()),
        "both_wrong_bit_events": int(both_wrong.sum()),
        "both_correct_bit_events": int(both_correct.sum()),
    }


def transition_per_bit_rows(name, left, right):
    left_idx = {image: idx for idx, image in enumerate(left["names"])}
    right_idx = {image: idx for idx, image in enumerate(right["names"])}
    common = sorted(set(left_idx) & set(right_idx))
    li = np.array([left_idx[image] for image in common], dtype=np.int64)
    ri = np.array([right_idx[image] for image in common], dtype=np.int64)
    le = left["errors"][li]
    re = right["errors"][ri]
    damaged = np.logical_and(~le, re)
    repaired = np.logical_and(le, ~re)
    rows = []
    for bit in range(le.shape[1]):
        rows.append(
            {
                "transition": name,
                "bit": bit,
                "left_ba": float((1.0 - le[:, bit].mean()) * 100.0),
                "right_ba": float((1.0 - re[:, bit].mean()) * 100.0),
                "delta_ba": float((le[:, bit].mean() - re[:, bit].mean()) * 100.0),
                "damaged_count": int(damaged[:, bit].sum()),
                "repaired_count": int(repaired[:, bit].sum()),
                "net_repair_minus_damage": int(repaired[:, bit].sum() - damaged[:, bit].sum()),
            }
        )
    return rows


def transition_per_image_rows(name, left, right, threshold):
    left_idx = {image: idx for idx, image in enumerate(left["names"])}
    right_idx = {image: idx for idx, image in enumerate(right["names"])}
    common = sorted(set(left_idx) & set(right_idx))
    rows = []
    for image in common:
        li = left_idx[image]
        ri = right_idx[image]
        le = left["errors"][li]
        re = right["errors"][ri]
        damaged = np.where(np.logical_and(~le, re))[0]
        repaired = np.where(np.logical_and(le, ~re))[0]
        left_ba = float(left["per_image_ba"][li])
        right_ba = float(right["per_image_ba"][ri])
        if left_ba < threshold and right_ba >= threshold:
            state = "recovered_to_ge90"
        elif left_ba >= threshold and right_ba < threshold:
            state = "newly_below90"
        elif left_ba < threshold and right_ba < threshold:
            state = "persistent_below90"
        else:
            state = "both_ge90"
        rows.append(
            {
                "transition": name,
                "image": image,
                "left_ba": left_ba,
                "right_ba": right_ba,
                "delta_ba": right_ba - left_ba,
                "left_error_bits": int(le.sum()),
                "right_error_bits": int(re.sum()),
                "damaged_count": int(damaged.size),
                "repaired_count": int(repaired.size),
                "state": state,
                "damaged_bits": " ".join(str(int(bit)) for bit in damaged),
                "repaired_bits": " ".join(str(int(bit)) for bit in repaired),
            }
        )
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir, data_by_name, transitions, threshold):
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_rows = [sample_summary(data) for data in data_by_name.values()]
    write_csv(
        output_dir / "variant_summary.csv",
        variant_rows,
        [
            "sample",
            "seed",
            "stage",
            "condition",
            "images",
            "avg_ba",
            "min_image_ba",
            "p05_image_ba",
            "p10_image_ba",
            "hard_lt90_images",
            "hard_lt80_images",
            "min_bit",
            "min_bit_ba",
            "weak_bits_top8",
            "weak_bits_top8_ba",
        ],
    )

    per_bit_rows = []
    per_image_rows = []
    for data in data_by_name.values():
        sample = data["sample"]["name"]
        for bit, ba in enumerate(data["per_bit_ba"]):
            per_bit_rows.append(
                {
                    "sample": sample,
                    "seed": data["sample"]["seed"],
                    "stage": data["sample"]["stage"],
                    "condition": data["sample"]["condition"],
                    "bit": bit,
                    "ba": float(ba),
                    "error_rate": float(100.0 - ba),
                }
            )
        for idx, image in enumerate(data["names"]):
            failed = np.where(data["errors"][idx])[0]
            per_image_rows.append(
                {
                    "sample": sample,
                    "seed": data["sample"]["seed"],
                    "stage": data["sample"]["stage"],
                    "condition": data["sample"]["condition"],
                    "image": image,
                    "ba": float(data["per_image_ba"][idx]),
                    "error_bits": int(failed.size),
                    "failed_bits": " ".join(str(int(bit)) for bit in failed),
                }
            )
    write_csv(
        output_dir / "per_bit_summary.csv",
        per_bit_rows,
        ["sample", "seed", "stage", "condition", "bit", "ba", "error_rate"],
    )
    write_csv(
        output_dir / "per_image_summary.csv",
        per_image_rows,
        ["sample", "seed", "stage", "condition", "image", "ba", "error_bits", "failed_bits"],
    )

    transition_rows = []
    transition_bit_rows = []
    transition_image_rows = []
    for item in transitions:
        left = data_by_name[item["left"]]
        right = data_by_name[item["right"]]
        transition_rows.append(compare_transition(item["name"], left, right, threshold))
        transition_bit_rows.extend(transition_per_bit_rows(item["name"], left, right))
        transition_image_rows.extend(transition_per_image_rows(item["name"], left, right, threshold))

    write_csv(
        output_dir / "transition_summary.csv",
        transition_rows,
        [
            "name",
            "left",
            "right",
            "common_images",
            "gt_equal_rate",
            "left_avg_ba",
            "right_avg_ba",
            "delta_avg_ba",
            "left_hard_lt90",
            "right_hard_lt90",
            "recovered_to_ge90",
            "newly_below90",
            "persistent_below90",
            "hard_jaccard",
            "bit_ba_corr",
            "image_ba_corr",
            "weak_bit_overlap_top8",
            "weak_bit_jaccard_top8",
            "weak_bit_overlap_top16",
            "weak_bit_jaccard_top16",
            "error_event_jaccard",
            "damaged_bit_events",
            "repaired_bit_events",
            "both_wrong_bit_events",
            "both_correct_bit_events",
        ],
    )
    write_csv(
        output_dir / "transition_per_bit_damage.csv",
        transition_bit_rows,
        [
            "transition",
            "bit",
            "left_ba",
            "right_ba",
            "delta_ba",
            "damaged_count",
            "repaired_count",
            "net_repair_minus_damage",
        ],
    )
    write_csv(
        output_dir / "transition_per_image_damage.csv",
        transition_image_rows,
        [
            "transition",
            "image",
            "left_ba",
            "right_ba",
            "delta_ba",
            "left_error_bits",
            "right_error_bits",
            "damaged_count",
            "repaired_count",
            "state",
            "damaged_bits",
            "repaired_bits",
        ],
    )

    payload = {"variants": variant_rows, "transitions": transition_rows}
    (output_dir / "diagnosis_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# O256 exp48 bitplane x refiner-damage tracking",
        "",
        "## What This Diagnoses",
        "",
        "- Whether weak bit positions are stable across seeds and stages.",
        "- Whether RGB refiner damage happens on clean decoding or mainly after Dropout.",
        "- Whether failed Dropout images are repaired, newly damaged, or persist after refiner changes.",
        "",
        "## Variant Summary",
        "",
        "| Sample | Seed | Stage | Condition | Images | Avg BA | Hard <90 | Min bit | Min bit BA | Weak bits top8 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in variant_rows:
        lines.append(
            f"| {row['sample']} | {row['seed']} | {row['stage']} | {row['condition']} | "
            f"{row['images']} | {row['avg_ba']:.2f} | {row['hard_lt90_images']} | "
            f"{row['min_bit']} | {row['min_bit_ba']:.2f} | {row['weak_bits_top8']} |"
        )

    lines.extend(
        [
            "",
            "## Transition Summary",
            "",
            "| Transition | Δ Avg BA | Hard <90 L/R | Recovered | New below | Weak-bit top8 overlap | Bit BA corr | Error-event J | Damaged/Repaired events |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in transition_rows:
        lines.append(
            f"| {row['name']} | {row['delta_avg_ba']:.2f} | "
            f"{row['left_hard_lt90']}/{row['right_hard_lt90']} | "
            f"{row['recovered_to_ge90']} | {row['newly_below90']} | "
            f"{row['weak_bit_overlap_top8']} | {row['bit_ba_corr']:.3f} | "
            f"{row['error_event_jaccard']:.3f} | "
            f"{row['damaged_bit_events']}/{row['repaired_bit_events']} |"
        )

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- For `base_clean_to_rgb_clean`, high damaged events mean the refiner breaks clean decoding.",
            "- For `base_dropout_to_rgb_dropout`, high damaged events mean the refiner breaks attack-robust evidence.",
            "- If weak-bit overlap across seed61/seed62 is high, bit-specific training may be a stable target.",
            "- If weak-bit overlap is low, bitplane should stay as a diagnosis tool rather than a fixed repair rule.",
            "- If clean transitions are safe but Dropout transitions are damaged, the next model change should be attack-aware or decoder-aware refiner design.",
            "",
            "## Outputs",
            "",
            "- `variant_summary.csv`",
            "- `per_bit_summary.csv`",
            "- `per_image_summary.csv`",
            "- `transition_summary.csv`",
            "- `transition_per_bit_damage.csv`",
            "- `transition_per_image_damage.csv`",
            "- `diagnosis_summary.json`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preflight(samples, transitions):
    names = {sample["name"] for sample in samples}
    for transition in transitions:
        if transition["left"] not in names or transition["right"] not in names:
            raise ValueError(f"Unknown transition endpoint: {transition}")
    for sample in samples:
        if not (ROOT / "configs" / sample["config"]).exists():
            raise FileNotFoundError(f"Missing config for {sample['name']}: {sample['config']}")
        for key in ("resume", "image_dir", "gt_message_dir"):
            if not sample[key].exists():
                raise FileNotFoundError(f"Missing {key} for {sample['name']}: {sample[key]}")
        images = image_paths(sample["image_dir"])
        if not images:
            raise RuntimeError(f"No images found for {sample['name']}: {sample['image_dir']}")
        valid_messages = sum(1 for path in images if message_path(sample["gt_message_dir"], path.stem).exists())
        if valid_messages == 0:
            raise RuntimeError(f"No matching messages for {sample['name']}: {sample['gt_message_dir']}")
        print(
            f"[check] {sample['name']}: images={len(images)} matching_messages={valid_messages}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Track bitplane stability and RGB refiner damage across clean/Dropout outputs."
    )
    parser.add_argument("--sample", action="append", required=True)
    parser.add_argument("--transition", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--threshold_ba", type=float, default=90.0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    samples = [parse_sample(spec) for spec in args.sample]
    transitions = [parse_transition(spec) for spec in args.transition]
    preflight(samples, transitions)
    if args.check_only:
        print("Preflight passed.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    decoder_cache = {}
    data_by_name = {}
    for sample in samples:
        print(f"Diagnosing {sample['name']} from {sample['image_dir']}", flush=True)
        data_by_name[sample["name"]] = diagnose_sample(
            sample, device, decoder_cache, max_images=args.max_images
        )

    output_dir = Path(args.output_dir)
    write_outputs(output_dir, data_by_name, transitions, args.threshold_ba)
    print(f"Summary md: {output_dir / 'summary.md'}")
    print(f"Transition summary: {output_dir / 'transition_summary.csv'}")


if __name__ == "__main__":
    main()
