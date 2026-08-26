#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


EXPERIMENTS = (
    {
        "name": "3A: Stage B refiner",
        "prefix": "exp3A_stageB_transfer_to_nongeomlight_base_seed61",
        "attack_run": "exp3A_stageB_nongeom_transfer_to_nongeomlight_base_seed61",
    },
    {
        "name": "3B: Stage C refiner",
        "prefix": "exp3B_stageC_transfer_to_nongeomlight_base_seed61",
        "attack_run": "exp3B_stageC_nongeom_transfer_to_nongeomlight_base_seed61",
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def last_float(pattern, text, label):
    matches = re.findall(pattern, text)
    if not matches:
        raise RuntimeError(f"Could not parse {label}")
    return float(matches[-1])


def parse_clean_ba(text):
    patterns = (
        r"Clean\s+\|\s+([0-9.]+)",
        r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9.]+)%",
        r"Average Bit Accuracy \(BA\):\s*([0-9.]+)%",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    raise RuntimeError("Could not parse Clean BA")


def main():
    args = parse_args()
    root = Path(args.project_root)
    rows = []

    for experiment in EXPERIMENTS:
        prefix = experiment["prefix"]
        psnr_text = (root / "logs" / f"{prefix}_psnr_ssim.log").read_text(
            encoding="utf-8", errors="ignore"
        )
        clean_text = (root / "logs" / f"{prefix}_clean_ba.log").read_text(
            encoding="utf-8", errors="ignore"
        )
        summary_path = (
            root
            / "results"
            / "attacked_Result"
            / experiment["attack_run"]
            / "ba_summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        rows.append(
            {
                "name": experiment["name"],
                "psnr": last_float(
                    r"Testing set, PSNR is ([0-9.]+)", psnr_text, "PSNR"
                ),
                "ssim": last_float(r"SSIM is ([0-9.]+)", psnr_text, "SSIM"),
                "clean_ba": parse_clean_ba(clean_text),
                "average_ba": float(summary["average_ba"]),
                "minimum_ba": float(summary["minimum_ba"]),
                "minimum_attack": summary["minimum_attack"],
            }
        )

    for row in rows:
        row["gate"] = (
            row["psnr"] >= 30.0
            and row["clean_ba"] >= 95.0
            and row["average_ba"] >= 70.0
        )

    lines = [
        "# Experiment 3: refiner transfer to attack-aware base",
        "",
        "Base: `train10000-base20k-then-nongeomlight5k`",
        "",
        "| Candidate | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack | Gate |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['psnr']:.4f} | {row['ssim']:.4f} | "
            f"{row['clean_ba']:.2f}% | {row['average_ba']:.2f}% | "
            f"{row['minimum_ba']:.2f}% | {row['minimum_attack']} | "
            f"{'PASS' if row['gate'] else 'FAIL'} |"
        )

    passing = [row["name"] for row in rows if row["gate"]]
    lines.extend(
        [
            "",
            "Gate: PSNR >= 30 dB, Clean BA >= 95%, non-geometric average BA >= 70%.",
            "",
            "Decision:",
            (
                "- Transfer gate passed: " + ", ".join(passing)
                if passing
                else "- Neither existing refiner passed. Retrain refine10000 Stage A/B on the attack-aware base."
            ),
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
