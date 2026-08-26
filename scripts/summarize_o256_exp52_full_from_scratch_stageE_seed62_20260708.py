#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


REFERENCES = [
    {
        "variant": "exp40_seed61_formal_stageD_reference",
        "kind": "reference",
        "psnr": 33.9372,
        "ssim": 0.9819,
        "lpips": 0.0409,
        "clean_ba": 99.96,
        "avg_ba": 98.77,
        "min_ba": 91.01,
        "dropout_ba": 91.01,
        "min_attack": "Paper_Dropout_r03",
        "base_train_time_s": "13632.29",
        "refiner_train_time_s": "2527.77",
        "eval_time_s": "59.65",
        "note": "seed61 formal StageD reference",
    },
    {
        "variant": "exp41_seed62_formal_stageD_reference",
        "kind": "reference",
        "psnr": 34.0293,
        "ssim": 0.9820,
        "lpips": 0.0443,
        "clean_ba": 99.98,
        "avg_ba": 98.68,
        "min_ba": 89.94,
        "dropout_ba": 89.94,
        "min_attack": "Paper_Dropout_r03",
        "base_train_time_s": "13632.29",
        "refiner_train_time_s": "2522.55",
        "eval_time_s": "59.98",
        "note": "seed62 formal StageD reference; misses Dropout floor by 0.06",
    },
    {
        "variant": "exp51_seed62_stageE_continuation_reference",
        "kind": "continuation_reference",
        "psnr": 33.9524,
        "ssim": 0.9820,
        "lpips": 0.0446,
        "clean_ba": 99.98,
        "avg_ba": 98.72,
        "min_ba": 90.16,
        "dropout_ba": 90.16,
        "min_attack": "Paper_Dropout_r03",
        "base_train_time_s": "N/A",
        "refiner_train_time_s": "260.55",
        "eval_time_s": "60.51",
        "note": "seed62 StageE continuation probe reference",
    },
]


BASE_TRAIN_LOGS = [
    "exp52_identity20k_train.log",
    "exp52_base30_nongeomlight10k_train.log",
    "exp52_base50_nongeomlight_plus20k_train.log",
    "exp52_base55_weakattack_plus5k_train.log",
    "exp52_base60_dropout_focus_plus5k_train.log",
    "exp52_base65_dropout_margin_plus5k_train.log",
]


REFINER_TRAIN_LOGS = [
    "exp52_refiner_stageA.log",
    "exp52_refiner_stageB.log",
    "exp52_refiner_stageBplus.log",
    "exp52_refiner_stageC.log",
    "exp52_refiner_stageD_dropout_balanced.log",
    "exp52_refiner_stageE_decoderaware.log",
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
    refiner_train = sum_seconds(logs, REFINER_TRAIN_LOGS) if kind == "rgb_stageE" else None
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
        "dropout_ba": attack_rows.get("Paper_Dropout_r03", float("nan")),
        "min_attack": min_attack,
        "base_train_time_s": format_seconds(base_train),
        "refiner_train_time_s": format_seconds(refiner_train),
        "eval_time_s": format_seconds(eval_time),
        "note": "",
    }


def candidate_note(row):
    if row["psnr"] >= 33.8 and row["clean_ba"] >= 99.0 and row["avg_ba"] >= 98.0 and row["min_ba"] >= 90.0:
        return "Passed the locked full from-scratch StageE formal thresholds."
    if row["min_ba"] >= 90.0:
        return "Recovered the BA floor, but quality or average BA needs inspection."
    return "Did not pass the locked full from-scratch StageE formal thresholds."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--per_attack_csv", required=True)
    args = parser.parse_args()

    exp_root = Path(args.experiment_root)
    base_row = parse_variant(exp_root, "exp52_base65_dropout_margin", "base")
    rgb_row = parse_variant(exp_root, "exp52_rgb_full_from_scratch_stageE", "rgb_stageE")
    base_row["note"] = "fresh seed62 base65 rerun under locked protocol"
    rgb_row["note"] = "fresh seed62 full protocol with StageE included"
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
        "dropout_ba",
        "min_attack",
        "base_train_time_s",
        "refiner_train_time_s",
        "eval_time_s",
        "note",
    ]
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    per_attack_path = Path(args.per_attack_csv)
    with per_attack_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "attack", "ba"])
        writer.writeheader()
        for variant in ("exp52_base65_dropout_margin", "exp52_rgb_full_from_scratch_stageE"):
            attack_rows = read_attack_rows(exp_root / "attacked_Result" / f"{variant}_nongeom" / "ba_report.txt")
            for attack, ba in sorted(attack_rows.items()):
                writer.writerow({"variant": variant, "attack": attack, "ba": ba})

    md_lines = [
        "# O256 exp52 seed62 full from-scratch StageE formal rerun summary",
        "",
        "## Main Table",
        "",
        "| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Dropout BA | Min attack | Base train s | Refiner train s | Eval/apply s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {variant} | {kind} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {clean_ba:.2f} | {avg_ba:.2f} | {min_ba:.2f} | {dropout_ba:.2f} | {min_attack} | {base_train_time_s} | {refiner_train_time_s} | {eval_time_s} |".format(
                **row
            )
        )

    md_lines.extend(["", "## Exp52 Per-Attack BA", ""])
    for variant in ("exp52_base65_dropout_margin", "exp52_rgb_full_from_scratch_stageE"):
        attack_rows = read_attack_rows(exp_root / "attacked_Result" / f"{variant}_nongeom" / "ba_report.txt")
        md_lines.extend(["", f"### {variant}", "", "| Attack | BA |", "|---|---:|"])
        for attack, ba in sorted(attack_rows.items()):
            md_lines.append(f"| {attack} | {ba:.2f} |")

    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is the seed62 full from-scratch formal rerun with StageE included in the locked refiner protocol.",
            "- Base protocol remains identity20k, nongeom-light to base50, weakattack +5k, dropout-focus +5k, dropout-margin +5k.",
            "- Refiner protocol is StageA image10k, StageB message2k, StageB+ message500, StageC targeted500, StageD dropout-balanced500, then StageE decoder-aware250.",
            "- Success criterion: PSNR >= 33.8, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.",
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
