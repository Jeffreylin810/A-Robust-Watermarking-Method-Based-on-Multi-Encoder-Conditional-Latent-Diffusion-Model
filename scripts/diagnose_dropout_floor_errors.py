#!/usr/bin/env python3
import argparse
import csv
import json
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


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key, dict2namespace(value) if isinstance(value, dict) else value)
    return namespace


def parse_variant_spec(spec):
    parts = spec.split("|")
    if len(parts) != 6:
        raise ValueError(
            "Variant spec must be name|kind|config|resume|gt_message_dir|attack_dir"
        )
    name, kind, config, resume, gt_message_dir, attack_dir = parts
    return {
        "name": name,
        "kind": kind,
        "config": config,
        "resume": Path(resume),
        "gt_message_dir": Path(gt_message_dir),
        "attack_dir": Path(attack_dir),
    }


def parse_compare_spec(spec):
    parts = spec.split("|")
    if len(parts) != 2:
        raise ValueError("Compare spec must be left|right")
    return parts[0], parts[1]


def load_config(name):
    with (ROOT / "configs" / name).open("r", encoding="utf-8") as f:
        return dict2namespace(yaml.safe_load(f))


def get_img_tensor(img_path, device, image_size):
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    return data_transform(tf(img).unsqueeze(0).to(device))


def load_decoder(cache, variant, device):
    key = (variant["config"], str(variant["resume"]))
    if key in cache:
        return cache[key]

    config = load_config(variant["config"])
    config.device = device
    dummy_args = argparse.Namespace(resume=str(variant["resume"]))
    diffusion = DenoisingDiffusion(dummy_args, config)
    diffusion.load_ddm_ckpt(str(variant["resume"]), ema=True)
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
    return float(np.percentile(values, q)) if len(values) else float("nan")


def diagnose_variant(variant, device, model_cache, max_images=0):
    diffusion, decoder, image_size = load_decoder(model_cache, variant, device)
    img_files = sorted(
        p for p in variant["attack_dir"].iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if max_images:
        img_files = img_files[:max_images]
    if not img_files:
        raise RuntimeError(f"No images found: {variant['attack_dir']}")

    image_names = []
    errors = []
    gt_messages = []
    pred_messages = []

    for idx, img_path in enumerate(img_files, start=1):
        gt_path = variant["gt_message_dir"] / f"{img_path.stem}_message.npy"
        if not gt_path.exists():
            continue
        gt = np.load(gt_path).reshape(-1).astype(np.int8)
        pred = decode_image(diffusion, decoder, img_path, device, image_size)
        if pred.shape[0] != gt.shape[0]:
            raise RuntimeError(
                f"Message length mismatch for {variant['name']} {img_path.name}: "
                f"pred={pred.shape[0]}, gt={gt.shape[0]}"
            )
        image_names.append(img_path.stem)
        gt_messages.append(gt)
        pred_messages.append(pred)
        errors.append(pred != gt)
        if idx % 100 == 0:
            print(f"[{variant['name']}] decoded {idx}/{len(img_files)} images", flush=True)

    if not errors:
        raise RuntimeError(f"No valid image/message pairs for {variant['name']}")

    error_matrix = np.stack(errors).astype(bool)
    gt_matrix = np.stack(gt_messages).astype(np.int8)
    pred_matrix = np.stack(pred_messages).astype(np.int8)
    per_image_ba = (1.0 - error_matrix.mean(axis=1)) * 100.0
    per_bit_ba = (1.0 - error_matrix.mean(axis=0)) * 100.0

    return {
        "variant": variant,
        "image_names": image_names,
        "error_matrix": error_matrix,
        "gt_matrix": gt_matrix,
        "pred_matrix": pred_matrix,
        "per_image_ba": per_image_ba,
        "per_bit_ba": per_bit_ba,
    }


def summarize_variant(data):
    per_image = data["per_image_ba"]
    per_bit = data["per_bit_ba"]
    weak_bits = np.argsort(per_bit)[:8]
    return {
        "variant": data["variant"]["name"],
        "kind": data["variant"]["kind"],
        "images": len(data["image_names"]),
        "avg_ba": float(np.mean(per_image)),
        "min_image_ba": float(np.min(per_image)),
        "p05_image_ba": percentile(per_image, 5),
        "p10_image_ba": percentile(per_image, 10),
        "hard_lt90_images": int(np.sum(per_image < 90.0)),
        "hard_lt80_images": int(np.sum(per_image < 80.0)),
        "min_bit_ba": float(np.min(per_bit)),
        "min_bit": int(np.argmin(per_bit)),
        "weak_bits_top8": " ".join(str(int(bit)) for bit in weak_bits),
        "weak_bits_top8_ba": " ".join(f"{per_bit[bit]:.2f}" for bit in weak_bits),
    }


def compare_variants(left, right):
    left_index = {name: idx for idx, name in enumerate(left["image_names"])}
    right_index = {name: idx for idx, name in enumerate(right["image_names"])}
    common = sorted(set(left_index) & set(right_index))
    if not common:
        raise RuntimeError(f"No common images for {left['variant']['name']} vs {right['variant']['name']}")

    li = np.array([left_index[name] for name in common], dtype=np.int64)
    ri = np.array([right_index[name] for name in common], dtype=np.int64)
    le = left["error_matrix"][li]
    re = right["error_matrix"][ri]
    lba = left["per_image_ba"][li]
    rba = right["per_image_ba"][ri]
    lgt = left["gt_matrix"][li]
    rgt = right["gt_matrix"][ri]

    left_hard90 = set(np.array(common)[lba < 90.0])
    right_hard90 = set(np.array(common)[rba < 90.0])
    hard_union = left_hard90 | right_hard90
    hard_inter = left_hard90 & right_hard90

    left_weak_bits = set(int(v) for v in np.argsort(left["per_bit_ba"])[:8])
    right_weak_bits = set(int(v) for v in np.argsort(right["per_bit_ba"])[:8])
    weak_union = left_weak_bits | right_weak_bits
    weak_inter = left_weak_bits & right_weak_bits

    error_inter = int(np.logical_and(le, re).sum())
    error_union = int(np.logical_or(le, re).sum())
    gt_equal_rate = float(np.mean(np.all(lgt == rgt, axis=1)) * 100.0)
    if np.std(lba) > 0 and np.std(rba) > 0:
        image_ba_corr = float(np.corrcoef(lba, rba)[0, 1])
    else:
        image_ba_corr = float("nan")
    if np.std(left["per_bit_ba"]) > 0 and np.std(right["per_bit_ba"]) > 0:
        bit_ba_corr = float(np.corrcoef(left["per_bit_ba"], right["per_bit_ba"])[0, 1])
    else:
        bit_ba_corr = float("nan")

    return {
        "comparison": f"{left['variant']['name']} -> {right['variant']['name']}",
        "common_images": len(common),
        "left_avg_ba": float(np.mean(lba)),
        "right_avg_ba": float(np.mean(rba)),
        "delta_avg_ba": float(np.mean(rba) - np.mean(lba)),
        "left_hard_lt90": len(left_hard90),
        "right_hard_lt90": len(right_hard90),
        "hard_image_overlap": len(hard_inter),
        "hard_image_jaccard": float(len(hard_inter) / len(hard_union)) if hard_union else 1.0,
        "weak_bit_overlap_top8": len(weak_inter),
        "weak_bit_jaccard_top8": float(len(weak_inter) / len(weak_union)) if weak_union else 1.0,
        "error_event_intersection": error_inter,
        "error_event_union": error_union,
        "error_event_jaccard": float(error_inter / error_union) if error_union else 1.0,
        "gt_message_equal_rate": gt_equal_rate,
        "image_ba_corr": image_ba_corr,
        "bit_ba_corr": bit_ba_corr,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir, data_by_name, comparisons):
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_rows = [summarize_variant(data) for data in data_by_name.values()]
    write_csv(
        output_dir / "variant_summary.csv",
        variant_rows,
        [
            "variant",
            "kind",
            "images",
            "avg_ba",
            "min_image_ba",
            "p05_image_ba",
            "p10_image_ba",
            "hard_lt90_images",
            "hard_lt80_images",
            "min_bit_ba",
            "min_bit",
            "weak_bits_top8",
            "weak_bits_top8_ba",
        ],
    )

    per_bit_rows = []
    per_image_rows = []
    worst_rows = []
    for data in data_by_name.values():
        name = data["variant"]["name"]
        for bit, ba in enumerate(data["per_bit_ba"]):
            per_bit_rows.append(
                {
                    "variant": name,
                    "bit": bit,
                    "ba": float(ba),
                    "error_rate": float(100.0 - ba),
                }
            )
        order = np.argsort(data["per_image_ba"])
        for idx, image_name in enumerate(data["image_names"]):
            per_image_rows.append(
                {
                    "variant": name,
                    "image": image_name,
                    "ba": float(data["per_image_ba"][idx]),
                    "error_bits": int(data["error_matrix"][idx].sum()),
                }
            )
        for rank, idx in enumerate(order[:30], start=1):
            worst_rows.append(
                {
                    "variant": name,
                    "rank": rank,
                    "image": data["image_names"][idx],
                    "ba": float(data["per_image_ba"][idx]),
                    "error_bits": int(data["error_matrix"][idx].sum()),
                    "failed_bit_ids": " ".join(
                        str(int(v)) for v in np.where(data["error_matrix"][idx])[0]
                    ),
                }
            )

    write_csv(output_dir / "per_bit_ba.csv", per_bit_rows, ["variant", "bit", "ba", "error_rate"])
    write_csv(output_dir / "per_image_ba.csv", per_image_rows, ["variant", "image", "ba", "error_bits"])
    write_csv(
        output_dir / "worst_images.csv",
        worst_rows,
        ["variant", "rank", "image", "ba", "error_bits", "failed_bit_ids"],
    )

    comparison_fields = [
        "comparison",
        "common_images",
        "left_avg_ba",
        "right_avg_ba",
        "delta_avg_ba",
        "left_hard_lt90",
        "right_hard_lt90",
        "hard_image_overlap",
        "hard_image_jaccard",
        "weak_bit_overlap_top8",
        "weak_bit_jaccard_top8",
        "error_event_intersection",
        "error_event_union",
        "error_event_jaccard",
        "gt_message_equal_rate",
        "image_ba_corr",
        "bit_ba_corr",
    ]
    write_csv(output_dir / "comparisons.csv", comparisons, comparison_fields)

    payload = {
        "variant_summary": variant_rows,
        "comparisons": comparisons,
    }
    (output_dir / "diagnosis_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# O256 Dropout Floor Error Diagnosis",
        "",
        "## Variant Summary",
        "",
        "| Variant | Kind | Images | Avg BA | P05 image BA | Hard <90 | Min bit BA | Min bit | Weak bits top8 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in variant_rows:
        lines.append(
            f"| {row['variant']} | {row['kind']} | {row['images']} | {row['avg_ba']:.2f} | "
            f"{row['p05_image_ba']:.2f} | {row['hard_lt90_images']} | {row['min_bit_ba']:.2f} | "
            f"{row['min_bit']} | {row['weak_bits_top8']} |"
        )

    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| Comparison | Δ Avg BA | Hard <90 L/R | Hard overlap J | Weak-bit overlap | Error-event J | GT msg equal | Image BA corr | Bit BA corr |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['comparison']} | {row['delta_avg_ba']:.2f} | "
            f"{row['left_hard_lt90']}/{row['right_hard_lt90']} | {row['hard_image_jaccard']:.3f} | "
            f"{row['weak_bit_overlap_top8']} | {row['error_event_jaccard']:.3f} | "
            f"{row['gt_message_equal_rate']:.1f}% | {row['image_ba_corr']:.3f} | {row['bit_ba_corr']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- High hard-image overlap means the same images are repeatedly vulnerable under Dropout r03.",
            "- High weak-bit overlap means the floor is concentrated in specific message bit positions.",
            "- High error-event Jaccard means two variants are making the same bit errors; low Jaccard means the failure is more stochastic or redistributed.",
            "- If exp41 RGB and exp42A RGB have high overlap but exp42A does not improve Min BA, continuing the same margin direction is unlikely to help.",
            "",
            "## Outputs",
            "",
            "- `variant_summary.csv`",
            "- `per_bit_ba.csv`",
            "- `per_image_ba.csv`",
            "- `worst_images.csv`",
            "- `comparisons.csv`",
            "- `diagnosis_summary.json`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Diagnose O256 Dropout r03 floor errors.")
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--compare", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images", type=int, default=0)
    args = parser.parse_args()

    variants = [parse_variant_spec(spec) for spec in args.variant]
    compares = [parse_compare_spec(spec) for spec in args.compare]
    names = {variant["name"] for variant in variants}
    for left, right in compares:
        if left not in names or right not in names:
            raise ValueError(f"Unknown compare pair: {left}|{right}")

    for variant in variants:
        for key in ("resume", "gt_message_dir", "attack_dir"):
            if not variant[key].exists():
                raise FileNotFoundError(f"Missing {key} for {variant['name']}: {variant[key]}")
        if not (ROOT / "configs" / variant["config"]).exists():
            raise FileNotFoundError(f"Missing config for {variant['name']}: {variant['config']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model_cache = {}
    data_by_name = {}
    for variant in variants:
        print(f"Diagnosing {variant['name']} from {variant['attack_dir']}", flush=True)
        data_by_name[variant["name"]] = diagnose_variant(
            variant, device, model_cache, max_images=args.max_images
        )

    comparison_rows = [
        compare_variants(data_by_name[left], data_by_name[right])
        for left, right in compares
    ]
    output_dir = Path(args.output_dir)
    write_outputs(output_dir, data_by_name, comparison_rows)
    print(f"Summary md: {output_dir / 'summary.md'}")
    print(f"Variant summary: {output_dir / 'variant_summary.csv'}")
    print(f"Comparisons: {output_dir / 'comparisons.csv'}")


if __name__ == "__main__":
    main()
