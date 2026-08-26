#!/usr/bin/env python3
import argparse
import csv
import json
import os
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


def load_config(name):
    with open(Path("configs") / name, "r", encoding="utf-8") as f:
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


def decode_image(diffusion, decoder, img_path, device, image_size):
    img_tensor = get_img_tensor(img_path, device, image_size)
    if diffusion.latent_enabled:
        img_tensor = diffusion.encode_images(img_tensor)
    with torch.no_grad():
        decoded = decoder(img_tensor)
    return decoded.detach().cpu().numpy().round().clip(0, 1).reshape(-1)


def read_attack_dirs(attack_dir, include_attacks):
    dirs = [path for path in sorted(Path(attack_dir).iterdir()) if path.is_dir()]
    if include_attacks:
        wanted = set(include_attacks)
        dirs = [path for path in dirs if path.name in wanted]
    return dirs


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_weights(bit_rows, weak_attacks, top_k, base_weight, max_weight):
    selected = [row for row in bit_rows if row["attack"] in weak_attacks] if weak_attacks else bit_rows
    if not selected:
        selected = bit_rows
    by_bit = {}
    for row in selected:
        bit = int(row["bit"])
        by_bit.setdefault(bit, []).append(float(row["ba"]))
    bit_scores = {
        bit: sum(values) / len(values)
        for bit, values in by_bit.items()
    }
    ranked = sorted(bit_scores.items(), key=lambda item: item[1])
    length = max(by_bit) + 1 if by_bit else 0
    weights = [float(base_weight)] * length
    if ranked and top_k > 0:
        selected_bits = ranked[: min(top_k, len(ranked))]
        weakest = selected_bits[0][1]
        strongest_selected = selected_bits[-1][1]
        span = max(1e-6, strongest_selected - weakest)
        for bit, score in selected_bits:
            severity = 1.0 - ((score - weakest) / span if len(selected_bits) > 1 else 0.0)
            weights[bit] = float(base_weight + severity * (max_weight - base_weight))
    return weights, ranked


def main():
    parser = argparse.ArgumentParser(description="Per-bit BA analysis for O256 attacked outputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--attack_dir", required=True)
    parser.add_argument("--gt_message_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--weak_attacks",
        nargs="*",
        default=["Paper_Dropout_r03", "Paper_GaussianNoise_arg03", "Paper_SaltPepper_r02"],
    )
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--base_weight", type=float, default=1.0)
    parser.add_argument("--max_weight", type=float, default=3.0)
    parser.add_argument("--max_images", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    config.device = device
    dummy_args = argparse.Namespace(resume=args.resume)
    diffusion = DenoisingDiffusion(dummy_args, config)
    diffusion.load_ddm_ckpt(args.resume, ema=True)
    diffusion.model.eval()
    decoder = diffusion.model.module.decoder

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attack_dirs = read_attack_dirs(args.attack_dir, None)
    if not attack_dirs:
        raise RuntimeError(f"No attack directories found: {args.attack_dir}")

    bit_rows = []
    attack_rows = []
    message_length = None
    for attack_path in attack_dirs:
        img_files = sorted(
            path for path in attack_path.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if args.max_images:
            img_files = img_files[: args.max_images]
        correct = None
        total = 0
        for img_path in img_files:
            gt_message_path = Path(args.gt_message_dir) / f"{img_path.stem}_message.npy"
            if not gt_message_path.exists():
                continue
            gt = np.load(gt_message_path).reshape(-1)
            pred = decode_image(diffusion, decoder, img_path, device, config.data.image_size)
            if message_length is None:
                message_length = int(gt.shape[0])
            if correct is None:
                correct = np.zeros_like(gt, dtype=np.float64)
            correct += (pred == gt).astype(np.float64)
            total += 1
        if total == 0 or correct is None:
            continue
        per_bit_ba = correct / total * 100.0
        attack_rows.append(
            {
                "attack": attack_path.name,
                "image_count": total,
                "avg_ba": float(np.mean(per_bit_ba)),
                "min_ba": float(np.min(per_bit_ba)),
                "min_bit": int(np.argmin(per_bit_ba)),
            }
        )
        for bit, ba in enumerate(per_bit_ba):
            bit_rows.append(
                {
                    "attack": attack_path.name,
                    "bit": bit,
                    "ba": float(ba),
                    "error_rate": float(100.0 - ba),
                    "image_count": total,
                }
            )

    if message_length is None:
        raise RuntimeError("No valid images/messages were processed.")

    write_csv(
        output_dir / "per_attack_summary.csv",
        attack_rows,
        ["attack", "image_count", "avg_ba", "min_ba", "min_bit"],
    )
    write_csv(
        output_dir / "per_attack_per_bit_ba.csv",
        bit_rows,
        ["attack", "bit", "ba", "error_rate", "image_count"],
    )

    weights, ranked = build_weights(
        bit_rows,
        args.weak_attacks,
        args.top_k,
        args.base_weight,
        args.max_weight,
    )
    weight_payload = {
        "weights": weights,
        "weak_attacks": args.weak_attacks,
        "top_k": args.top_k,
        "base_weight": args.base_weight,
        "max_weight": args.max_weight,
        "ranked_bits": [
            {"bit": int(bit), "mean_weak_attack_ba": float(score)}
            for bit, score in ranked
        ],
    }
    (output_dir / "bit_weights.json").write_text(
        json.dumps(weight_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    weakest_rows = [
        {"rank": rank + 1, "bit": int(bit), "mean_weak_attack_ba": float(score)}
        for rank, (bit, score) in enumerate(ranked[: max(args.top_k, 16)])
    ]
    write_csv(output_dir / "weakest_bits.csv", weakest_rows, ["rank", "bit", "mean_weak_attack_ba"])

    lines = [
        "# Per-bit BA analysis",
        "",
        f"- attack_dir: `{args.attack_dir}`",
        f"- gt_message_dir: `{args.gt_message_dir}`",
        f"- weak_attacks: `{', '.join(args.weak_attacks)}`",
        "",
        "## Attack summary",
        "",
        "| Attack | Images | Avg BA | Min BA | Min bit |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in attack_rows:
        lines.append(
            f"| {row['attack']} | {row['image_count']} | {row['avg_ba']:.2f} | {row['min_ba']:.2f} | {row['min_bit']} |"
        )
    lines.extend(["", "## Weakest bits from selected weak attacks", "", "| Rank | Bit | Mean weak-attack BA |", "|---:|---:|---:|"])
    for row in weakest_rows:
        lines.append(f"| {row['rank']} | {row['bit']} | {row['mean_weak_attack_ba']:.2f} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `per_attack_summary.csv`",
            "- `per_attack_per_bit_ba.csv`",
            "- `weakest_bits.csv`",
            "- `bit_weights.json`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary md: {output_dir / 'summary.md'}")
    print(f"Bit weights: {output_dir / 'bit_weights.json'}")


if __name__ == "__main__":
    main()
