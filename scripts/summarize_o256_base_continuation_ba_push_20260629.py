#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


FORMAL_REFS = [
    {
        "variant": "formal_base30k",
        "method": "O256-Base",
        "psnr": 24.7037,
        "ssim": 0.7023,
        "lpips": 0.0934,
        "clean_ba": 99.30,
        "avg_ba": 87.33,
        "min_ba": 70.38,
        "min_attack": "Paper_Dropout_r03",
        "time": "01:41:49.74",
    },
    {
        "variant": "formal_rgb",
        "method": "O256-RGB",
        "psnr": 32.9034,
        "ssim": 0.9589,
        "lpips": 0.0461,
        "clean_ba": 95.81,
        "avg_ba": 87.38,
        "min_ba": 72.93,
        "min_attack": "Paper_Dropout_r03",
        "time": "02:33:57.31",
    },
]


def parse_last(patterns, text, label):
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


def parse_variant(exp_root: Path, tag: str, method: str):
    logs = exp_root / "logs"
    prefix = f"{tag}_{'base' if method == 'O256-Base' else 'rgb'}"
    psnr_text = (logs / f"{prefix}_psnr_ssim.log").read_text(encoding="utf-8", errors="ignore")
    clean_text = (logs / f"{prefix}_clean_ba.log").read_text(encoding="utf-8", errors="ignore")
    lpips_json = exp_root / "lpips" / f"{'o256_base' if method == 'O256-Base' else 'o256_rgb'}_{tag}" / "summary.json"
    if method == "O256-Base":
        attack_run = exp_root / "attacked_Result" / f"o256_base_{tag}_nongeom"
        time_log = logs / f"{tag}_train.log"
    else:
        attack_run = exp_root / "attacked_Result" / f"o256_rgb_{tag}_nongeom"
        time_log = logs / f"{tag}_rgb_apply.log"

    attack_rows = read_attack_rows(attack_run / "ba_report.txt")
    avg_ba = sum(attack_rows.values()) / len(attack_rows)
    min_attack = min(attack_rows, key=attack_rows.get)
    min_ba = attack_rows[min_attack]

    time_text = time_log.read_text(encoding="utf-8", errors="ignore")
    time_match = re.findall(r"real\s+([0-9.]+)", time_text)
    time_value = float(time_match[-1]) if time_match else None

    return {
        "variant": tag,
        "method": method,
        "psnr": parse_last((r"Testing set, PSNR is ([0-9.]+)",), psnr_text, "PSNR"),
        "ssim": parse_last((r"SSIM is ([0-9.]+)",), psnr_text, "SSIM"),
        "lpips": json.loads(lpips_json.read_text(encoding="utf-8"))["mean_lpips"],
        "clean_ba": parse_last(
            (
                r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9.]+)%",
                r"Clean\s+\|\s+([0-9.]+)",
            ),
            clean_text,
            "Clean BA",
        ),
        "avg_ba": avg_ba,
        "min_ba": min_ba,
        "min_attack": min_attack,
        "time": f"{time_value:.2f}s" if time_value is not None else "N/A",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    exp_root = Path(args.experiment_root)
    rows = []
    rows.extend(FORMAL_REFS)
    for tag in ("plus5k", "plus10k"):
        rows.append(parse_variant(exp_root, tag, "O256-Base"))
        rows.append(parse_variant(exp_root, tag, "O256-RGB"))

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "method", "psnr", "ssim", "lpips", "clean_ba", "avg_ba", "min_ba", "min_attack", "time"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        "# O256 base continuation BA-push summary",
        "",
        "## Main Table",
        "",
        "| Variant | Method | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Time |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {variant} | {method} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {clean_ba:.2f} | {avg_ba:.2f} | {min_ba:.2f} | {min_attack} | {time} |".format(
                **row
            )
        )

    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `plus5k` means continuing the formal base from 30k to 35k total steps.",
            "- `plus10k` means continuing from 35k to 40k total steps.",
            "- `O256-RGB` variants reuse the existing formal RGB refinement module instead of retraining a new refiner.",
        ]
    )

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(md_path)
    print(csv_path)


if __name__ == "__main__":
    main()
