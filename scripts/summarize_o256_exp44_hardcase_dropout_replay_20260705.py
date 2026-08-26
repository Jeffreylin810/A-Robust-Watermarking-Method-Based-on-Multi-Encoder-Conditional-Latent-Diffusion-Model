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
        "train_time_s": "N/A",
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
        "train_time_s": "N/A",
        "eval_time_s": "59.98",
    },
    {
        "variant": "exp42A_dropout_margin_probe_rgb",
        "kind": "reference",
        "psnr": 34.0741,
        "ssim": 0.9824,
        "lpips": 0.0441,
        "clean_ba": 100.00,
        "avg_ba": 98.73,
        "min_ba": 89.80,
        "min_attack": "Paper_Dropout_r03",
        "train_time_s": "2533.70",
        "eval_time_s": "59.75",
    },
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


def format_seconds(value):
    return f"{value:.2f}" if value is not None else "N/A"


def parse_variant(exp_root: Path, variant: str, kind: str):
    logs = exp_root / "logs"
    attack_rows = read_attack_rows(exp_root / "attacked_Result" / f"{variant}_nongeom" / "ba_report.txt")
    min_attack = min(attack_rows, key=attack_rows.get)
    psnr_text = (logs / f"{variant}_psnr_ssim.log").read_text(encoding="utf-8", errors="ignore")
    clean_text = (logs / f"{variant}_clean_ba.log").read_text(encoding="utf-8", errors="ignore")
    lpips_json = exp_root / "lpips" / variant / "summary.json"
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
        "train_time_s": format_seconds(read_real_seconds(logs / f"{variant}_train.log")),
        "eval_time_s": format_seconds(read_real_seconds(logs / f"{variant}_apply.log")),
    }


def status_note(row):
    if row["psnr"] >= 33.5 and row["clean_ba"] >= 99.0 and row["avg_ba"] >= 98.0 and row["min_ba"] >= 90.0:
        return "Hard-case replay passed the probe threshold and may justify a formal protocol candidate."
    return "Hard-case replay did not pass the probe threshold; treat as diagnostic/negative evidence."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--per_attack_csv", required=True)
    args = parser.parse_args()

    exp_root = Path(args.experiment_root)
    row = parse_variant(exp_root, "exp44_hardcase_dropout_replay500", "rgb")
    rows = REFERENCES + [row]

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
        "train_time_s",
        "eval_time_s",
    ]
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    attack_rows = read_attack_rows(
        exp_root / "attacked_Result" / "exp44_hardcase_dropout_replay500_nongeom" / "ba_report.txt"
    )
    per_attack_path = Path(args.per_attack_csv)
    with per_attack_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "ba"])
        writer.writeheader()
        for attack, ba in sorted(attack_rows.items()):
            writer.writerow({"attack": attack, "ba": ba})

    hard_summary_path = exp_root / "hardcase_selection" / "hardcase_summary.json"
    hard_summary = {}
    if hard_summary_path.exists():
        hard_summary = json.loads(hard_summary_path.read_text(encoding="utf-8"))

    md_lines = [
        "# O256 exp44 train-side hard-case dropout replay summary",
        "",
        "## Main Table",
        "",
        "| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train s | Eval/apply s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for item in rows:
        md_lines.append(
            "| {variant} | {kind} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {clean_ba:.2f} | {avg_ba:.2f} | {min_ba:.2f} | {min_attack} | {train_time_s} | {eval_time_s} |".format(
                **item
            )
        )

    md_lines.extend(["", "## RGB Per-Attack BA", "", "| Attack | BA |", "|---|---:|"])
    for attack, ba in sorted(attack_rows.items()):
        md_lines.append(f"| {attack} | {ba:.2f} |")

    if hard_summary:
        md_lines.extend(
            [
                "",
                "## Train-Side Hardcase Selection",
                "",
                f"- processed_images: `{hard_summary.get('processed_images')}`",
                f"- below_threshold_images: `{hard_summary.get('below_threshold_images')}`",
                f"- selected_images: `{hard_summary.get('selected_images')}`",
                f"- selected_avg_ba: `{hard_summary.get('selected_avg_ba'):.2f}`",
                f"- selected_ba_range: `{hard_summary.get('selected_min_ba'):.2f}` to `{hard_summary.get('selected_max_ba'):.2f}`",
            ]
        )

    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- exp44 is a continuation probe, not a formal result.",
            "- It selects hard cases from train10000 only, not from finaltest1000.",
            "- It resumes the exp41 StageD refiner and runs a short Dropout r03 replay on the train-side hard subset.",
            "- Success criterion: PSNR >= 33.5, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.",
            f"- Candidate status: {status_note(row)}",
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
