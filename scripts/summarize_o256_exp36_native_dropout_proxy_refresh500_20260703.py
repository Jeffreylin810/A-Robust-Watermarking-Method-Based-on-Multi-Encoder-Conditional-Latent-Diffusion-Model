#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


BASELINES = [
    ("exp33_dropout_balanced_candidate", 33.2346, 0.9744, 0.045658645013347265, 99.89, 98.41090909090912, 90.05, "309.46s", "60.04s"),
    ("exp34_full_candidate_validation_rgb", 33.2333, 0.9741, 0.045815825197845696, 99.90, 98.35454545454546, 89.72, "2325.12s", "60.40s"),
    ("exp35_dropout_compensation_refresh150", 33.3079, 0.9748, 0.04547432049550116, 99.89, 98.35363636363637, 89.74, "239.93s", "59.68s"),
]


def baseline_rows():
    return [
        {
            "variant": name,
            "kind": "rgb",
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "clean_ba": clean_ba,
            "avg_ba": avg_ba,
            "min_ba": min_ba,
            "min_attack": "Paper_Dropout_r03",
            "train_time": train_time,
            "apply_time": apply_time,
        }
        for name, psnr, ssim, lpips, clean_ba, avg_ba, min_ba, train_time, apply_time in BASELINES
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


def parse_variant(exp_root: Path):
    logs = exp_root / "logs"
    variant = "exp36_native_dropout_proxy_refresh500"
    attack_rows = read_attack_rows(exp_root / "attacked_Result" / f"{variant}_nongeom" / "ba_report.txt")
    min_attack = min(attack_rows, key=attack_rows.get)
    train_time = read_real_seconds(logs / f"{variant}_train.log")
    apply_time = read_real_seconds(logs / f"{variant}_apply.log")
    psnr_text = (logs / f"{variant}_psnr_ssim.log").read_text(encoding="utf-8", errors="ignore")
    clean_text = (logs / f"{variant}_clean_ba.log").read_text(encoding="utf-8", errors="ignore")
    lpips_json = exp_root / "lpips" / variant / "summary.json"
    return {
        "variant": variant,
        "kind": "rgb",
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
        "train_time": f"{train_time:.2f}s" if train_time is not None else "N/A",
        "apply_time": f"{apply_time:.2f}s" if apply_time is not None else "N/A",
    }


def candidate_note(row):
    if row["psnr"] >= 33.0 and row["clean_ba"] >= 99.0 and row["avg_ba"] >= 98.0 and row["min_ba"] >= 90.0:
        return "Passed formal-candidate thresholds with native-dropout proxy alignment."
    return "Proxy alignment did not pass all thresholds; stop refiner-loss micro-tuning."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    exp_root = Path(args.experiment_root)
    rows = baseline_rows()
    variant_row = parse_variant(exp_root)
    rows.append(variant_row)

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
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# O256 exp36 native dropout proxy refresh500 summary",
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
            "- This experiment tests a theory-level correction: aligning the training dropout proxy with the native formal Dropout(0.3) direction.",
            "- It starts from the exp34 regenerated base and uses exp33-style dropout-balanced refiner settings, but with `dropout_r03_native_only`.",
            "- Success criterion: PSNR >= 33.0, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.",
            f"- Candidate status: {candidate_note(variant_row)}",
        ]
    )

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Summary md: {md_path}")
    print(f"Summary csv: {csv_path}")


if __name__ == "__main__":
    main()
