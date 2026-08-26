#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


PREFIX = "exp4_refine10000_nongeomlightbase_A10k_B2k_seed61"
ATTACK_RUN = "exp4_nongeom_refine10000_nongeomlightbase_A10k_B2k_seed61"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_last(patterns, text, label):
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    raise RuntimeError(f"Could not parse {label}")


def main():
    args = parse_args()
    root = Path(args.project_root)
    psnr_text = (root / "logs" / f"{PREFIX}_psnr_ssim.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    clean_text = (root / "logs" / f"{PREFIX}_clean_ba.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    summary = json.loads(
        (
            root
            / "results"
            / "attacked_Result"
            / ATTACK_RUN
            / "ba_summary.json"
        ).read_text(encoding="utf-8")
    )

    psnr = parse_last((r"Testing set, PSNR is ([0-9.]+)",), psnr_text, "PSNR")
    ssim = parse_last((r"SSIM is ([0-9.]+)",), psnr_text, "SSIM")
    clean_ba = parse_last(
        (
            r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9.]+)%",
            r"Clean\s+\|\s+([0-9.]+)",
        ),
        clean_text,
        "Clean BA",
    )
    average_ba = float(summary["average_ba"])
    minimum_ba = float(summary["minimum_ba"])
    gate = psnr >= 30.0 and clean_ba >= 95.0 and average_ba >= 70.0

    text = f"""# Experiment 4: refine10000 on attack-aware base

Base: `train10000-base20k-then-nongeomlight5k`

Refiner: `refine10000-nongeomlightbase-A10k-B2k`

| Metric | Result | Gate |
|---|---:|---:|
| PSNR | {psnr:.4f} dB | >= 30 dB |
| SSIM | {ssim:.4f} | - |
| Clean BA | {clean_ba:.2f}% | >= 95% |
| Non-geometric average BA | {average_ba:.2f}% | >= 70% |
| Minimum BA | {minimum_ba:.2f}% | target >= 70% |
| Minimum attack | {summary['minimum_attack']} | - |

Overall gate: {'PASS' if gate else 'FAIL'}

Decision:
{"- Use this Stage B checkpoint as the formal candidate. Only add Stage C if the minimum BA still needs improvement." if gate else "- Inspect Clean BA and the weakest attacks before deciding whether to tune Stage B or add a short targeted Stage C."}
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
