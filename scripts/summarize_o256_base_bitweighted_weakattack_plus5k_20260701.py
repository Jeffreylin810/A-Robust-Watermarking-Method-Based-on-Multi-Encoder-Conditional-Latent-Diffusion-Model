#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


BASELINES = [
    {
        "variant": "plus20k_refresh500",
        "kind": "rgb",
        "psnr": 33.8299,
        "ssim": 0.9738,
        "lpips": 0.03830676269810647,
        "clean_ba": 99.39,
        "avg_ba": 95.91,
        "min_ba": 83.36,
        "min_attack": "Paper_Dropout_r03",
        "train_time": "283.14s",
        "apply_time": "61.13s",
    },
    {
        "variant": "A_weakattack_plus5k_rgb",
        "kind": "rgb",
        "psnr": 32.9388,
        "ssim": 0.9715,
        "lpips": 0.04828327928110957,
        "clean_ba": 99.89,
        "avg_ba": 98.20545454545454,
        "min_ba": 89.61,
        "min_attack": "Paper_Dropout_r03",
        "train_time": "1109.26s",
        "apply_time": "60.40s",
    },
    {
        "variant": "bitweighted_refiner_repair",
        "kind": "rgb",
        "psnr": 34.1088,
        "ssim": 0.9756,
        "lpips": 0.03636835849005729,
        "clean_ba": 99.33,
        "avg_ba": 95.76181818181817,
        "min_ba": 82.78,
        "min_attack": "Paper_Dropout_r03",
        "train_time": "287.88s",
        "apply_time": "61.81s",
    },
]


def parse_last_float(patterns, text, label):
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    raise RuntimeError(f"Could not parse {label}")


def read_attack_rows(path: Path):
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"^\s*(\S.*?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*$", line)
        if match and match.group(1).strip() != "Attack Name":
            rows[match.group(1).strip()] = float(match.group(2))
    if not rows:
        raise RuntimeError(f"No BA rows found in {path}")
    return rows


def read_real_seconds(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"real\s+([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def parse_variant(exp_root: Path, *, tag, kind, attack_run, lpips_dir, psnr_log, clean_log, train_log, apply_log=None):
    logs = exp_root / "logs"
    psnr_text = (logs / psnr_log).read_text(encoding="utf-8", errors="ignore")
    clean_text = (logs / clean_log).read_text(encoding="utf-8", errors="ignore")
    attack_rows = read_attack_rows(exp_root / "attacked_Result" / attack_run / "ba_report.txt")
    avg_ba = sum(attack_rows.values()) / len(attack_rows)
    min_attack = min(attack_rows, key=attack_rows.get)
    train_time = read_real_seconds(logs / train_log)
    apply_time = read_real_seconds(logs / apply_log) if apply_log else None
    return {
        "variant": tag,
        "kind": kind,
        "psnr": parse_last_float((r"Testing set, PSNR is ([0-9.]+)",), psnr_text, "PSNR"),
        "ssim": parse_last_float((r"SSIM is ([0-9.]+)",), psnr_text, "SSIM"),
        "lpips": json.loads((exp_root / "lpips" / lpips_dir / "summary.json").read_text(encoding="utf-8"))[
            "mean_lpips"
        ],
        "clean_ba": parse_last_float((r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9.]+)%",), clean_text, "Clean BA"),
        "avg_ba": avg_ba,
        "min_ba": attack_rows[min_attack],
        "min_attack": min_attack,
        "train_time": f"{train_time:.2f}s" if train_time is not None else "resume_plus20k_base",
        "apply_time": f"{apply_time:.2f}s" if apply_time is not None else "N/A",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    exp_root = Path(args.experiment_root)
    rows = list(BASELINES)
    rows.append(
        parse_variant(
            exp_root,
            tag="base_bitweighted_weakattack_plus5k",
            kind="base",
            attack_run="o256_base_bitweighted_weakattack_plus5k_nongeom",
            lpips_dir="o256_base_bitweighted_weakattack_plus5k",
            psnr_log="base_bitweighted_weakattack_plus5k_psnr_ssim.log",
            clean_log="base_bitweighted_weakattack_plus5k_clean_ba.log",
            train_log="base_bitweighted_weakattack_plus5k_train.log",
        )
    )
    rows.append(
        parse_variant(
            exp_root,
            tag="base_bitweighted_weakattack_plus5k_rgb_reuse_refresh500",
            kind="rgb",
            attack_run="o256_rgb_bitweighted_weakattack_plus5k_reuse_refresh500_nongeom",
            lpips_dir="o256_rgb_bitweighted_weakattack_plus5k_reuse_refresh500",
            psnr_log="rgb_bitweighted_weakattack_plus5k_reuse_psnr_ssim.log",
            clean_log="rgb_bitweighted_weakattack_plus5k_reuse_clean_ba.log",
            train_log="base_bitweighted_weakattack_plus5k_train.log",
            apply_log="rgb_bitweighted_weakattack_plus5k_reuse_apply.log",
        )
    )

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "kind",
        "psnr",
        "ssim",
        "lpips",
        "clean_ba",
        "avg_ba",
        "min_ba",
        "min_attack",
        "train_time",
        "apply_time",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# O256 base-side bit-weighted weak-attack continuation summary",
        "",
        "## Main Table",
        "",
        "| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {variant} | {kind} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {clean_ba:.2f} | {avg_ba:.2f} | {min_ba:.2f} | {min_attack} | {train_time} | {apply_time} |".format(
                **row
            )
        )
    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This experiment moves weak-bit weighting from the refiner-only stage into base-side continuation.",
            "- The main comparison is against `A_weakattack_plus5k_rgb`: if this row improves BA without a larger quality drop, bit-aware base training is a real breakthrough.",
            "- If it matches A but does not improve it, the diagnosis is useful but the simple weighting scheme is not enough; the next step should be channel/architecture-aware routing.",
        ]
    )

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Summary md: {md_path}")
    print(f"Summary csv: {csv_path}")


if __name__ == "__main__":
    main()
