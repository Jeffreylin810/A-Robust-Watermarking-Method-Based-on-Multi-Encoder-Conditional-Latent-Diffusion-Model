#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


REFERENCES = [
    {
        "variant": "exp40_seed61_full_rerun_reference_rgb",
        "kind": "reference",
        "psnr": 33.9372,
        "ssim": 0.9819,
        "lpips": 0.0409,
        "clean_ba": 99.96,
        "avg_ba": 98.77,
        "min_ba": 91.01,
        "min_attack": "Paper_Dropout_r03",
        "base_extra_train_time_s": "N/A",
        "refiner_train_time_s": "2527.77",
        "eval_time_s": "59.65",
    },
    {
        "variant": "exp41_seed62_full_rerun_reference_rgb",
        "kind": "reference",
        "psnr": 34.0293,
        "ssim": 0.9820,
        "lpips": 0.0443,
        "clean_ba": 99.98,
        "avg_ba": 98.68,
        "min_ba": 89.94,
        "min_attack": "Paper_Dropout_r03",
        "base_extra_train_time_s": "N/A",
        "refiner_train_time_s": "2522.55",
        "eval_time_s": "59.98",
    },
]


REFINER_TRAIN_LOGS = [
    "exp42A_refiner_stageA.log",
    "exp42A_refiner_stageB.log",
    "exp42A_refiner_stageBplus.log",
    "exp42A_refiner_stageC.log",
    "exp42A_refiner_stageD_dropout_balanced.log",
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
    base_extra_train = read_real_seconds(logs / "exp42A_base70_dropout_margin_plus10k_train.log")
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
        "base_extra_train_time_s": format_seconds(base_extra_train),
        "refiner_train_time_s": format_seconds(refiner_train),
        "eval_time_s": format_seconds(eval_time),
    }


def candidate_note(row):
    if row["psnr"] >= 33.8 and row["clean_ba"] >= 99.8 and row["avg_ba"] >= 98.6 and row["min_ba"] >= 90.5:
        return "Strong exp42A probe pass: the added dropout-margin stage improves the floor while preserving quality."
    if row["psnr"] >= 33.5 and row["clean_ba"] >= 99.0 and row["avg_ba"] >= 98.0 and row["min_ba"] >= 90.0:
        return "Basic exp42A probe pass: good enough to justify a full from-scratch exp42B candidate."
    return "Probe did not pass; do not promote this fixed-margin idea to a full from-scratch rerun yet."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--per_attack_csv", required=True)
    args = parser.parse_args()

    exp_root = Path(args.experiment_root)
    base_row = parse_variant(exp_root, "exp42A_base70_dropout_margin_plus10k", "base")
    rgb_row = parse_variant(exp_root, "exp42A_rgb_base70_dropout_balanced", "rgb")
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
        "base_extra_train_time_s",
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
        exp_root / "attacked_Result" / "exp42A_rgb_base70_dropout_balanced_nongeom" / "ba_report.txt"
    )
    per_attack_path = Path(args.per_attack_csv)
    with per_attack_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "ba"])
        writer.writeheader()
        for attack, ba in sorted(attack_rows.items()):
            writer.writerow({"attack": attack, "ba": ba})

    md_lines = [
        "# O256 exp42A dropout-floor safety margin probe summary",
        "",
        "## Main Table",
        "",
        "| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Extra base train s | Refiner train s | Eval/apply s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {variant} | {kind} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {clean_ba:.2f} | {avg_ba:.2f} | {min_ba:.2f} | {min_attack} | {base_extra_train_time_s} | {refiner_train_time_s} | {eval_time_s} |".format(
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
            "- exp42A is a continuation probe, not a formal main result.",
            "- It starts from exp41 seed62 base65 and adds one extra `identity_nongeom_dropout_margin` epoch to form base70.",
            "- The RGB refiner is trained again with the fixed exp40/41 A/B/B+/C/D recipe; no seed-specific refiner tweak is introduced.",
            "- Promotion criterion for exp42B: PSNR >= 33.5, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.",
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
