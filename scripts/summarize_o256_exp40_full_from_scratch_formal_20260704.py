#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


REFERENCES = [
    {
        "variant": "exp38_seed62_continuation_reference_rgb",
        "kind": "reference",
        "psnr": 33.1252,
        "ssim": 0.9745,
        "lpips": 0.0467,
        "clean_ba": 99.93,
        "avg_ba": 98.60,
        "min_ba": 90.31,
        "min_attack": "Paper_Dropout_r03",
        "base_train_time_s": "N/A",
        "refiner_train_time_s": "N/A",
        "eval_time_s": "N/A",
    },
]


BASE_TRAIN_LOGS = [
    "exp40_identity20k_train.log",
    "exp40_base30_nongeomlight10k_train.log",
    "exp40_base50_nongeomlight_plus20k_train.log",
    "exp40_base55_weakattack_plus5k_train.log",
    "exp40_base60_dropout_focus_plus5k_train.log",
    "exp40_base65_dropout_margin_plus5k_train.log",
]


REFINER_TRAIN_LOGS = [
    "exp40_refiner_stageA.log",
    "exp40_refiner_stageB.log",
    "exp40_refiner_stageBplus.log",
    "exp40_refiner_stageC.log",
    "exp40_refiner_stageD_dropout_balanced.log",
]


def parse_last_float(patterns, text, label):
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
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
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"real\s+([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def sum_seconds(log_dir: Path, names):
    values = [read_real_seconds(log_dir / name) for name in names]
    values = [v for v in values if v is not None]
    return sum(values) if values else None


def format_seconds(value):
    return f"{value:.2f}" if value is not None else "N/A"


def parse_variant(exp_root: Path, variant: str, kind: str):
    logs = exp_root / "logs"
    attack_rows = read_attack_rows(exp_root / "attacked_Result" / f"{variant}_nongeom" / "ba_report.txt")
    min_attack = min(attack_rows, key=attack_rows.get)
    psnr_text = (logs / f"{variant}_psnr_ssim.log").read_text(encoding="utf-8", errors="ignore")
    clean_text = (logs / f"{variant}_clean_ba.log").read_text(encoding="utf-8", errors="ignore")
    lpips_json = exp_root / "lpips" / variant / "summary.json"
    base_train = sum_seconds(logs, BASE_TRAIN_LOGS)
    refiner_train = sum_seconds(logs, REFINER_TRAIN_LOGS) if kind == "rgb" else None
    eval_time = read_real_seconds(logs / f"{variant}_apply.log")
    return {
        "variant": variant,
        "kind": kind,
        "psnr": parse_last_float((r"Testing set, PSNR is ([0-9.]+)",), psnr_text, "PSNR"),
        "ssim": parse_last_float((r"SSIM is ([0-9.]+)",), psnr_text, "SSIM"),
        "lpips": json.loads(lpips_json.read_text(encoding="utf-8"))["mean_lpips"],
        "clean_ba": parse_last_float(
            (r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9.]+)%",),
            clean_text,
            "Clean BA",
        ),
        "avg_ba": sum(attack_rows.values()) / len(attack_rows),
        "min_ba": attack_rows[min_attack],
        "min_attack": min_attack,
        "base_train_time_s": format_seconds(base_train),
        "refiner_train_time_s": format_seconds(refiner_train),
        "eval_time_s": format_seconds(eval_time),
    }


def candidate_note(row):
    if row["psnr"] >= 33.0 and row["clean_ba"] >= 99.0 and row["avg_ba"] >= 98.0 and row["min_ba"] >= 90.0:
        return "Passed full from-scratch formal rerun thresholds."
    return "Did not pass all full from-scratch thresholds; compare against exp37/exp38 to identify which stage failed to transfer."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--per_attack_csv", required=True)
    args = parser.parse_args()

    exp_root = Path(args.experiment_root)
    base_row = parse_variant(exp_root, "exp40_base65_dropout_margin", "base")
    rgb_row = parse_variant(exp_root, "exp40_rgb_full_from_scratch_dropout_balanced", "rgb")
    rows = REFERENCES + [base_row, rgb_row]

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
        "base_train_time_s",
        "refiner_train_time_s",
        "eval_time_s",
    ]
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    attack_rows = read_attack_rows(
        exp_root / "attacked_Result" / "exp40_rgb_full_from_scratch_dropout_balanced_nongeom" / "ba_report.txt"
    )
    per_attack_path = Path(args.per_attack_csv)
    with per_attack_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "ba"])
        writer.writeheader()
        for attack, ba in sorted(attack_rows.items()):
            writer.writerow({"attack": attack, "ba": ba})

    md_lines = [
        "# O256 exp40 full from-scratch formal rerun summary",
        "",
        "## Main Table",
        "",
        "| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Base train s | Refiner train s | Eval/apply s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {variant} | {kind} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {clean_ba:.2f} | {avg_ba:.2f} | {min_ba:.2f} | {min_attack} | {base_train_time_s} | {refiner_train_time_s} | {eval_time_s} |".format(
                **row
            )
        )

    md_lines.extend(
        [
            "",
            "## RGB Per-Attack BA",
            "",
            "| Attack | BA |",
            "|---|---:|",
        ]
    )
    for attack, ba in sorted(attack_rows.items()):
        md_lines.append(f"| {attack} | {ba:.2f} |")

    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is the full rerun version: identity20k, nongeom-light to base50, weakattack +5k, dropout-focus +5k, dropout-margin +5k, regenerated train10000 pairs, and a freshly trained RGB refiner.",
            "- Refiner stages are A10k, B2k, B+500, targeted C500, and final dropout-balanced D500.",
            "- Success criterion: PSNR >= 33.0, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.",
            f"- Candidate status: {candidate_note(rgb_row)}",
        ]
    )

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Summary md: {md_path}")
    print(f"Summary csv: {csv_path}")
    print(f"Per-attack csv: {per_attack_path}")


if __name__ == "__main__":
    main()
