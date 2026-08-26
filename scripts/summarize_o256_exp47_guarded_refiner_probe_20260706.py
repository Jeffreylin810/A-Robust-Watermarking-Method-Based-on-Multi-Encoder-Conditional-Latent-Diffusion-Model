#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


REFERENCES = [
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
        "train_time_s": "2522.55",
        "eval_time_s": "59.98",
        "note": "exp41 original StageD",
    },
    {
        "variant": "exp44_hardcase_dropout_replay500",
        "kind": "negative_reference",
        "psnr": 34.5561,
        "ssim": 0.9839,
        "lpips": 0.0391,
        "clean_ba": 99.98,
        "avg_ba": 98.47,
        "min_ba": 88.13,
        "min_attack": "Paper_Dropout_r03",
        "train_time_s": "124.42",
        "eval_time_s": "59.64",
        "note": "hardcase replay failed",
    },
]


VARIANTS = [
    {
        "variant": "exp47A_stageD_mild_residual_guard",
        "kind": "mild_guard",
        "note": "StageC -> guarded StageD, mild residual/frequency guard",
    },
    {
        "variant": "exp47B_stageD_strong_residual_guard",
        "kind": "strong_guard",
        "note": "StageC -> guarded StageD, stronger residual/frequency guard",
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


def read_report_arg(report_json: Path, key: str):
    if not report_json.exists():
        return ""
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    return payload.get("args", {}).get(key, "")


def parse_variant(exp_root: Path, spec: dict):
    variant = spec["variant"]
    logs = exp_root / "logs"
    attack_rows = read_attack_rows(exp_root / "attacked_Result" / f"{variant}_nongeom" / "ba_report.txt")
    min_attack = min(attack_rows, key=attack_rows.get)
    psnr_text = (logs / f"{variant}_psnr_ssim.log").read_text(encoding="utf-8", errors="ignore")
    clean_text = (logs / f"{variant}_clean_ba.log").read_text(encoding="utf-8", errors="ignore")
    lpips_json = exp_root / "lpips" / variant / "summary.json"
    report_json = exp_root / "refinement" / variant / "report.json"
    return {
        "variant": variant,
        "kind": spec["kind"],
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
        "dropout_ba": attack_rows.get("Paper_Dropout_r03", float("nan")),
        "train_time_s": format_seconds(read_real_seconds(logs / f"{variant}_train.log")),
        "eval_time_s": format_seconds(read_real_seconds(logs / f"{variant}_apply.log")),
        "residual_lap_preserve_weight": read_report_arg(report_json, "residual_lap_preserve_weight"),
        "anti_cancel_weight": read_report_arg(report_json, "anti_cancel_weight"),
        "anti_cancel_margin": read_report_arg(report_json, "anti_cancel_margin"),
        "note": spec["note"],
    }


def status_note(row):
    if row["psnr"] >= 33.8 and row["clean_ba"] >= 99.0 and row["avg_ba"] >= 98.0 and row["min_ba"] >= 90.0:
        return "passes the guarded-refiner probe threshold"
    if row["min_ba"] >= 90.0:
        return "recovers the Dropout floor but needs quality/average BA inspection"
    return "does not recover the Dropout floor"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--per_attack_csv", required=True)
    args = parser.parse_args()

    exp_root = Path(args.experiment_root)
    variant_rows = [parse_variant(exp_root, spec) for spec in VARIANTS]
    rows = REFERENCES + variant_rows

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
        "train_time_s",
        "eval_time_s",
        "residual_lap_preserve_weight",
        "anti_cancel_weight",
        "anti_cancel_margin",
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
        for spec in VARIANTS:
            attack_rows = read_attack_rows(
                exp_root / "attacked_Result" / f"{spec['variant']}_nongeom" / "ba_report.txt"
            )
            for attack, ba in sorted(attack_rows.items()):
                writer.writerow({"variant": spec["variant"], "attack": attack, "ba": ba})

    md_lines = [
        "# O256 exp47 guarded StageD refiner probe summary",
        "",
        "## Main Table",
        "",
        "| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Dropout BA | Min attack | Train s | Eval/apply s | Guard |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for item in rows:
        guard = ""
        if item.get("residual_lap_preserve_weight") not in ("", None):
            guard = (
                f"lap={item.get('residual_lap_preserve_weight')}, "
                f"anti={item.get('anti_cancel_weight')}, "
                f"margin={item.get('anti_cancel_margin')}"
            )
        display = dict(item)
        display["guard"] = guard
        display["dropout_ba"] = item.get("dropout_ba", item["min_ba"])
        md_lines.append(
            "| {variant} | {kind} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {clean_ba:.2f} | {avg_ba:.2f} | {min_ba:.2f} | {dropout_ba:.2f} | {min_attack} | {train_time_s} | {eval_time_s} | {guard} |".format(
                **display,
            )
        )

    md_lines.extend(["", "## Exp47 Per-Attack BA", ""])
    for spec in VARIANTS:
        attack_rows = read_attack_rows(
            exp_root / "attacked_Result" / f"{spec['variant']}_nongeom" / "ba_report.txt"
        )
        md_lines.extend(["", f"### {spec['variant']}", "", "| Attack | BA |", "|---|---:|"])
        for attack, ba in sorted(attack_rows.items()):
            md_lines.append(f"| {attack} | {ba:.2f} |")

    md_lines.extend(["", "## Interpretation", ""])
    for row in variant_rows:
        md_lines.append(
            f"- `{row['variant']}` {status_note(row)}: PSNR `{row['psnr']:.4f}`, "
            f"Avg BA `{row['avg_ba']:.2f}`, Dropout BA `{row['dropout_ba']:.2f}`."
        )
    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- exp47 is a continuation architecture probe, not a formal result.",
            "- Both A/B restart from exp41 StageC and retrain only the final StageD with fixed guard losses.",
            "- If either branch passes, the next step should be a full from-scratch protocol rerun using the chosen guarded StageD.",
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
