#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


REFERENCES = {
    "seed61": {
        "reference_variant": "exp40_seed61_formal_stageD_reference",
        "psnr": 33.9372,
        "ssim": 0.9819,
        "lpips": 0.0409,
        "clean_ba": 99.96,
        "avg_ba": 98.77,
        "min_ba": 91.01,
        "dropout_ba": 91.01,
        "min_attack": "Paper_Dropout_r03",
    },
    "seed62": {
        "reference_variant": "exp41_seed62_formal_stageD_reference",
        "psnr": 34.0293,
        "ssim": 0.9820,
        "lpips": 0.0443,
        "clean_ba": 99.98,
        "avg_ba": 98.68,
        "min_ba": 89.94,
        "dropout_ba": 89.94,
        "min_attack": "Paper_Dropout_r03",
    },
}


VARIANTS = [
    {
        "seed": "seed61",
        "variant": "exp51_seed61_stageE_decoderaware_refresh",
        "kind": "stageE_probe",
        "note": "apply exp50B-like decoder-aware StageE to exp40 seed61",
    },
    {
        "seed": "seed62",
        "variant": "exp51_seed62_stageE_decoderaware_refresh",
        "kind": "stageE_probe",
        "note": "apply exp50B-like decoder-aware StageE to exp41 seed62",
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


def source_refiner_stage(report_json: Path):
    raw = str(read_report_arg(report_json, "resume_refiner"))
    if not raw:
        return ""
    path = Path(raw)
    if path.name == "refiner_final.pth":
        return path.parent.name
    return path.name


def parse_variant(exp_root: Path, spec: dict):
    variant = spec["variant"]
    logs = exp_root / "logs"
    attack_rows = read_attack_rows(exp_root / "attacked_Result" / f"{variant}_nongeom" / "ba_report.txt")
    min_attack = min(attack_rows, key=attack_rows.get)
    psnr_text = (logs / f"{variant}_psnr_ssim.log").read_text(encoding="utf-8", errors="ignore")
    clean_text = (logs / f"{variant}_clean_ba.log").read_text(encoding="utf-8", errors="ignore")
    lpips_json = exp_root / "lpips" / variant / "summary.json"
    report_json = exp_root / "refinement" / variant / "report.json"
    ref = REFERENCES[spec["seed"]]
    row = {
        "seed": spec["seed"],
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
        "dropout_ba": attack_rows.get("Paper_Dropout_r03", float("nan")),
        "min_attack": min_attack,
        "train_time_s": format_seconds(read_real_seconds(logs / f"{variant}_train.log")),
        "eval_time_s": format_seconds(read_real_seconds(logs / f"{variant}_apply.log")),
        "source_refiner": source_refiner_stage(report_json),
        "message_weight": read_report_arg(report_json, "message_weight"),
        "attack_message_weight": read_report_arg(report_json, "attack_message_weight"),
        "quality_gate_psnr": read_report_arg(report_json, "quality_gate_psnr"),
        "quality_gate_weight": read_report_arg(report_json, "quality_gate_weight"),
        "note": spec["note"],
    }
    row["delta_psnr_vs_stageD"] = row["psnr"] - ref["psnr"]
    row["delta_avg_ba_vs_stageD"] = row["avg_ba"] - ref["avg_ba"]
    row["delta_min_ba_vs_stageD"] = row["min_ba"] - ref["min_ba"]
    row["delta_dropout_ba_vs_stageD"] = row["dropout_ba"] - ref["dropout_ba"]
    return row


def reference_row(seed: str):
    ref = REFERENCES[seed]
    return {
        "seed": seed,
        "variant": ref["reference_variant"],
        "kind": "reference",
        "psnr": ref["psnr"],
        "ssim": ref["ssim"],
        "lpips": ref["lpips"],
        "clean_ba": ref["clean_ba"],
        "avg_ba": ref["avg_ba"],
        "min_ba": ref["min_ba"],
        "dropout_ba": ref["dropout_ba"],
        "min_attack": ref["min_attack"],
        "train_time_s": "N/A",
        "eval_time_s": "N/A",
        "source_refiner": "formal StageD",
        "message_weight": "1.4",
        "attack_message_weight": "0.45",
        "quality_gate_psnr": "33.0",
        "quality_gate_weight": "70.0",
        "delta_psnr_vs_stageD": 0.0,
        "delta_avg_ba_vs_stageD": 0.0,
        "delta_min_ba_vs_stageD": 0.0,
        "delta_dropout_ba_vs_stageD": 0.0,
        "note": "formal full from-scratch StageD reference",
    }


def status_note(row):
    if row["psnr"] >= 33.8 and row["clean_ba"] >= 99.0 and row["avg_ba"] >= 98.0 and row["min_ba"] >= 90.0:
        return "passes the cross-seed StageE probe threshold"
    if row["min_ba"] >= 90.0:
        return "recovers the BA floor but needs PSNR/Avg BA inspection"
    return "does not pass the cross-seed StageE probe threshold"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--per_attack_csv", required=True)
    args = parser.parse_args()

    exp_root = Path(args.experiment_root)
    variant_rows = [parse_variant(exp_root, spec) for spec in VARIANTS]
    rows = [reference_row("seed61"), variant_rows[0], reference_row("seed62"), variant_rows[1]]

    fields = [
        "seed",
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
        "delta_psnr_vs_stageD",
        "delta_avg_ba_vs_stageD",
        "delta_min_ba_vs_stageD",
        "delta_dropout_ba_vs_stageD",
        "train_time_s",
        "eval_time_s",
        "source_refiner",
        "message_weight",
        "attack_message_weight",
        "quality_gate_psnr",
        "quality_gate_weight",
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
        writer = csv.DictWriter(f, fieldnames=["seed", "variant", "attack", "ba"])
        writer.writeheader()
        for spec in VARIANTS:
            attack_rows = read_attack_rows(
                exp_root / "attacked_Result" / f"{spec['variant']}_nongeom" / "ba_report.txt"
            )
            for attack, ba in sorted(attack_rows.items()):
                writer.writerow({"seed": spec["seed"], "variant": spec["variant"], "attack": attack, "ba": ba})

    md_lines = [
        "# O256 exp51 decoder-aware StageE cross-seed probe summary",
        "",
        "## Main Table",
        "",
        "| Seed | Variant | Kind | PSNR | Clean BA | Avg BA | Min BA | Dropout BA | dPSNR | dMin BA | dDropout BA | Train s | Setting |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        setting = (
            f"src={row.get('source_refiner', '')}, "
            f"msg={row.get('message_weight', '')}, "
            f"atk_msg={row.get('attack_message_weight', '')}, "
            f"qgate={row.get('quality_gate_psnr', '')}/{row.get('quality_gate_weight', '')}"
        )
        display = dict(row)
        display["setting"] = setting
        md_lines.append(
            "| {seed} | {variant} | {kind} | {psnr:.4f} | {clean_ba:.2f} | {avg_ba:.2f} | {min_ba:.2f} | {dropout_ba:.2f} | {delta_psnr_vs_stageD:.4f} | {delta_min_ba_vs_stageD:.2f} | {delta_dropout_ba_vs_stageD:.2f} | {train_time_s} | {setting} |".format(
                **display
            )
        )

    md_lines.extend(["", "## Probe Interpretation", ""])
    for row in variant_rows:
        md_lines.append(
            f"- `{row['variant']}` {status_note(row)}: PSNR `{row['psnr']:.4f}` "
            f"({row['delta_psnr_vs_stageD']:+.4f}), Min BA `{row['min_ba']:.2f}` "
            f"({row['delta_min_ba_vs_stageD']:+.2f}), Dropout BA `{row['dropout_ba']:.2f}` "
            f"({row['delta_dropout_ba_vs_stageD']:+.2f})."
        )

    md_lines.extend(["", "## Per-Attack BA", ""])
    for spec in VARIANTS:
        attack_rows = read_attack_rows(
            exp_root / "attacked_Result" / f"{spec['variant']}_nongeom" / "ba_report.txt"
        )
        md_lines.extend(["", f"### {spec['variant']}", "", "| Attack | BA |", "|---|---:|"])
        for attack, ba in sorted(attack_rows.items()):
            md_lines.append(f"| {attack} | {ba:.2f} |")

    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- exp51 is still a continuation probe, not a full from-scratch formal result.",
            "- It tests whether the exp50B decoder-aware StageE rule is stable across both seed61 and seed62 formal from-scratch checkpoints.",
            "- If both seeds pass with small PSNR loss, the next step is a full from-scratch formal rerun that includes this StageE in the locked protocol.",
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
