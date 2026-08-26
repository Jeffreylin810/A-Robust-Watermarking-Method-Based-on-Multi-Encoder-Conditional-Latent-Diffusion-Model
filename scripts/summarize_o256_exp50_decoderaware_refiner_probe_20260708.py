#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


REFERENCES = [
    {
        "variant": "exp41_seed62_full_from_scratch_reference_rgb",
        "kind": "reference",
        "psnr": 34.0293,
        "ssim": 0.9820,
        "lpips": 0.0443,
        "clean_ba": 99.98,
        "avg_ba": 98.68,
        "min_ba": 89.94,
        "dropout_ba": 89.94,
        "min_attack": "Paper_Dropout_r03",
        "train_time_s": "2522.55",
        "eval_time_s": "59.98",
        "source_refiner": "exp41_stageD_dropout_balanced500",
        "message_weight": "1.4",
        "attack_message_weight": "0.45",
        "quality_gate_psnr": "33.0",
        "quality_gate_weight": "70.0",
        "note": "current seed62 formal candidate; Dropout floor is just below 90",
    },
    {
        "variant": "exp44_hardcase_dropout_replay_negative_reference",
        "kind": "negative_reference",
        "psnr": 34.5561,
        "ssim": 0.9839,
        "lpips": 0.0391,
        "clean_ba": 99.98,
        "avg_ba": 98.47,
        "min_ba": 88.13,
        "dropout_ba": 88.13,
        "min_attack": "Paper_Dropout_r03",
        "train_time_s": "124.42",
        "eval_time_s": "59.64",
        "source_refiner": "exp41_stageD_dropout_balanced500",
        "message_weight": "N/A",
        "attack_message_weight": "N/A",
        "quality_gate_psnr": "N/A",
        "quality_gate_weight": "N/A",
        "note": "hardcase replay increased PSNR but hurt Dropout",
    },
]


VARIANTS = [
    {
        "variant": "exp50A_decoderaware_stageD_from_stageC",
        "kind": "replace_stageD",
        "note": "replace exp41 final StageD with stronger decoder-aware Dropout loss",
    },
    {
        "variant": "exp50B_decoderaware_refresh_from_stageD",
        "kind": "micro_refresh",
        "note": "tiny correction on top of exp41 StageD to test whether the 89.94 Dropout floor can be nudged above 90",
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
        "dropout_ba": attack_rows.get("Paper_Dropout_r03", float("nan")),
        "min_attack": min_attack,
        "train_time_s": format_seconds(read_real_seconds(logs / f"{variant}_train.log")),
        "eval_time_s": format_seconds(read_real_seconds(logs / f"{variant}_apply.log")),
        "source_refiner": Path(str(read_report_arg(report_json, "resume_refiner"))).name,
        "message_weight": read_report_arg(report_json, "message_weight"),
        "attack_message_weight": read_report_arg(report_json, "attack_message_weight"),
        "quality_gate_psnr": read_report_arg(report_json, "quality_gate_psnr"),
        "quality_gate_weight": read_report_arg(report_json, "quality_gate_weight"),
        "note": spec["note"],
    }


def status_note(row):
    if row["psnr"] >= 33.8 and row["clean_ba"] >= 99.0 and row["avg_ba"] >= 98.0 and row["min_ba"] >= 90.0:
        return "passes the probe threshold and is worth a full from-scratch rerun"
    if row["dropout_ba"] > 89.94 and row["psnr"] >= 33.8 and row["clean_ba"] >= 99.0:
        return "improves the exp41 Dropout floor but does not fully pass"
    if row["dropout_ba"] >= 90.0:
        return "recovers Dropout but needs quality/average BA inspection"
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
        writer = csv.DictWriter(f, fieldnames=["variant", "attack", "ba"])
        writer.writeheader()
        for spec in VARIANTS:
            attack_rows = read_attack_rows(
                exp_root / "attacked_Result" / f"{spec['variant']}_nongeom" / "ba_report.txt"
            )
            for attack, ba in sorted(attack_rows.items()):
                writer.writerow({"variant": spec["variant"], "attack": attack, "ba": ba})

    md_lines = [
        "# O256 exp50 decoder-aware refiner probe summary",
        "",
        "## Main Table",
        "",
        "| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Dropout BA | Min attack | Train s | Eval/apply s | Decoder-aware setting |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for item in rows:
        setting = (
            f"src={item.get('source_refiner', '')}, "
            f"msg={item.get('message_weight', '')}, "
            f"atk_msg={item.get('attack_message_weight', '')}, "
            f"qgate={item.get('quality_gate_psnr', '')}/{item.get('quality_gate_weight', '')}"
        )
        display = dict(item)
        display["setting"] = setting
        md_lines.append(
            "| {variant} | {kind} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {clean_ba:.2f} | {avg_ba:.2f} | {min_ba:.2f} | {dropout_ba:.2f} | {min_attack} | {train_time_s} | {eval_time_s} | {setting} |".format(
                **display,
            )
        )

    md_lines.extend(["", "## Exp50 Per-Attack BA", ""])
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
            f"Clean BA `{row['clean_ba']:.2f}`, Avg BA `{row['avg_ba']:.2f}`, "
            f"Dropout BA `{row['dropout_ba']:.2f}`."
        )
    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- exp50 is a continuation probe, not a formal result.",
            "- It follows exp48/49: evaluation is valid, clean BA is not the main damaged area, and the remaining issue is the Dropout-after-refiner floor.",
            "- Branch A replaces only the final StageD from exp41 StageC; Branch B lightly refreshes the existing exp41 StageD.",
            "- If either branch passes, the next formal step should be a full from-scratch rerun with the chosen decoder-aware StageD setting.",
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
