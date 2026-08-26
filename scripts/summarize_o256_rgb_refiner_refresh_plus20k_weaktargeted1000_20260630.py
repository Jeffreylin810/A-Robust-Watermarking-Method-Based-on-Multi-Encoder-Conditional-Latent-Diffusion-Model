#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


BASELINES = [
    {
        "variant": "formal_rgb",
        "psnr": 32.9034,
        "ssim": 0.9589,
        "lpips": 0.0461,
        "clean_ba": 95.81,
        "avg_ba": 87.38,
        "min_ba": 72.93,
        "min_attack": "Paper_Dropout_r03",
        "train_time": "02:33:57.31",
        "apply_time": "N/A",
    },
    {
        "variant": "plus20k_refresh500",
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
        "variant": "plus20k_refresh1500",
        "psnr": 34.2602,
        "ssim": 0.9768,
        "lpips": 0.0355,
        "clean_ba": 99.33,
        "avg_ba": 95.72,
        "min_ba": 82.76,
        "min_attack": "Paper_Dropout_r03",
        "train_time": "503.79s",
        "apply_time": "61.57s",
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
        if match and match.group(1) != "Attack Name":
            rows[match.group(1).strip()] = float(match.group(2))
    if not rows:
        raise RuntimeError(f"No BA rows found in {path}")
    return rows


def read_real_seconds(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"real\s+([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def parse_variant(exp_root: Path):
    logs = exp_root / "logs"
    psnr_text = (logs / "plus20k_weaktargeted1000_psnr_ssim.log").read_text(encoding="utf-8", errors="ignore")
    clean_text = (logs / "plus20k_weaktargeted1000_clean_ba.log").read_text(encoding="utf-8", errors="ignore")
    attack_rows = read_attack_rows(exp_root / "attacked_Result" / "o256_rgb_plus20k_weaktargeted1000_nongeom" / "ba_report.txt")
    avg_ba = sum(attack_rows.values()) / len(attack_rows)
    min_attack = min(attack_rows, key=attack_rows.get)
    min_ba = attack_rows[min_attack]
    lpips_json = exp_root / "lpips" / "o256_rgb_plus20k_weaktargeted1000" / "summary.json"
    train_time = read_real_seconds(logs / "plus20k_weaktargeted1000_train.log")
    apply_time = read_real_seconds(logs / "plus20k_weaktargeted1000_apply.log")
    return {
        "variant": "plus20k_weaktargeted1000",
        "psnr": parse_last_float((r"Testing set, PSNR is ([0-9.]+)",), psnr_text, "PSNR"),
        "ssim": parse_last_float((r"SSIM is ([0-9.]+)",), psnr_text, "SSIM"),
        "lpips": json.loads(lpips_json.read_text(encoding="utf-8"))["mean_lpips"],
        "clean_ba": parse_last_float((r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9.]+)%",), clean_text, "Clean BA"),
        "avg_ba": avg_ba,
        "min_ba": min_ba,
        "min_attack": min_attack,
        "train_time": f"{train_time:.2f}s" if train_time is not None else "N/A",
        "apply_time": f"{apply_time:.2f}s" if apply_time is not None else "N/A",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    exp_root = Path(args.experiment_root)
    rows = []
    rows.extend(BASELINES)
    rows.append(parse_variant(exp_root))

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "psnr",
                "ssim",
                "lpips",
                "clean_ba",
                "avg_ba",
                "min_ba",
                "min_attack",
                "train_time",
                "apply_time",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        "# O256 +20k weak-targeted refresh1000 summary",
        "",
        "## Main Table",
        "",
        "| Variant | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {variant} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {clean_ba:.2f} | {avg_ba:.2f} | {min_ba:.2f} | {min_attack} | {train_time} | {apply_time} |".format(
                **row
            )
        )

    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `plus20k_weaktargeted1000` is the short weak-attack-targeted refresh experiment: it resumes from `plus20k_refresh500` and adds 500 more targeted refiner steps.",
            "- `plus20k_refresh1500` remains the more aggressive quality-push reference point.",
            "- The comparison is intended to reveal whether a shorter weak-targeted refresh can recover weakest attacks more efficiently than the longer quality-push branch.",
        ]
    )

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Summary md: {md_path}")
    print(f"Summary csv: {csv_path}")


if __name__ == "__main__":
    main()
