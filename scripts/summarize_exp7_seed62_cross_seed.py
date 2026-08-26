#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


REFERENCE = {
    "name": "seed61 + seed61 plugin",
    "apply_report": (
        "results/refinement/"
        "refine10000_targetedformal_C500_apply_eval1000_seed61/report.json"
    ),
    "clean_report": (
        "results/refinement/"
        "refine10000_targetedformal_C500_apply_eval1000_seed61/"
        "attacked_clean/ba_report.txt"
    ),
    "attack_summary": (
        "results/attacked_Result/"
        "exp6_nongeom_refine10000_targetedformal_C500_seed61/ba_summary.json"
    ),
}

TRANSFER = {
    "name": "seed62 + transferred seed61 plugin",
    "apply_report": (
        "results/refinement/"
        "exp7_seed62_with_seed61_C500_plugin_apply_eval1000/report.json"
    ),
    "clean_report": (
        "results/refinement/"
        "exp7_seed62_with_seed61_C500_plugin_apply_eval1000/"
        "attacked_clean/ba_report.txt"
    ),
    "attack_summary": (
        "results/attacked_Result/"
        "exp7_seed62_with_seed61_C500_plugin_nongeom/ba_summary.json"
    ),
}

BASE62 = {
    "psnr_log": "logs/exp7_seed62_base25k_psnr_ssim.log",
    "clean_report": (
        "results/attacked_Result/"
        "clean_exp7_seed62_base25k/ba_report.txt"
    ),
    "attack_summary": (
        "results/attacked_Result/"
        "exp7_seed62_base25k_nongeom/ba_summary.json"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_clean_ba(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        match = re.match(r"^\s*Clean\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*$", line)
        if match:
            return float(match.group(1))
    raise RuntimeError(f"Could not parse Clean BA from {path}")


def read_apply(root, spec):
    payload = json.loads((root / spec["apply_report"]).read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    attack = json.loads(
        (root / spec["attack_summary"]).read_text(encoding="utf-8")
    )
    return {
        "name": spec["name"],
        "psnr": float(metrics["psnr"]),
        "ssim": float(metrics["ssim"]),
        "clean_ba": read_clean_ba(root / spec["clean_report"]),
        "average_ba": float(attack["average_ba"]),
        "minimum_ba": float(attack["minimum_ba"]),
        "minimum_attack": attack["minimum_attack"],
    }


def read_base62(root):
    text = (root / BASE62["psnr_log"]).read_text(encoding="utf-8", errors="ignore")
    psnr_matches = re.findall(r"Testing set, PSNR is ([0-9.]+)", text)
    ssim_matches = re.findall(r"SSIM is ([0-9.]+)", text)
    if not psnr_matches or not ssim_matches:
        raise RuntimeError("Could not parse seed62 base PSNR/SSIM")
    attack = json.loads(
        (root / BASE62["attack_summary"]).read_text(encoding="utf-8")
    )
    return {
        "psnr": float(psnr_matches[-1]),
        "ssim": float(ssim_matches[-1]),
        "clean_ba": read_clean_ba(root / BASE62["clean_report"]),
        "average_ba": float(attack["average_ba"]),
        "minimum_ba": float(attack["minimum_ba"]),
        "minimum_attack": attack["minimum_attack"],
    }


def main():
    args = parse_args()
    root = Path(args.project_root)
    reference = read_apply(root, REFERENCE)
    transfer = read_apply(root, TRANSFER)
    base62 = read_base62(root)

    deltas = {
        "psnr": transfer["psnr"] - reference["psnr"],
        "ssim": transfer["ssim"] - reference["ssim"],
        "clean_ba": transfer["clean_ba"] - reference["clean_ba"],
        "average_ba": transfer["average_ba"] - reference["average_ba"],
        "minimum_ba": transfer["minimum_ba"] - reference["minimum_ba"],
    }
    checks = {
        "psnr": abs(deltas["psnr"]) <= 1.0,
        "clean_ba": abs(deltas["clean_ba"]) <= 3.0,
        "average_ba": abs(deltas["average_ba"]) <= 5.0,
        "minimum_ba": deltas["minimum_ba"] >= -5.0,
    }
    passed = all(checks.values())

    if passed:
        decision = (
            "PASS：seed61 訓練的插件可直接轉移到 seed62 base；目前沒有證據顯示"
            "插件只記住 seed61 checkpoint。下一步可保留同一插件進行正式比較。"
        )
    else:
        decision = (
            "FAIL：至少一項跨 seed 差異超出預先固定的容許範圍。下一步才需要"
            "使用 seed62 產生 refine10000 pairs，並重訓 seed62 專屬插件，以區分"
            "插件綁定問題與 base model 本身變異。"
        )

    def status(value):
        return "PASS" if value else "FAIL"

    text = f"""# Experiment 7: seed62 cross-seed plugin reproducibility

## Controlled design

- Dataset/split, architecture, message length, training steps, loss weights,
  VAE, attack profile, and evaluation images are unchanged.
- The only intended base-training change is `seed: 61 -> 62`.
- The RGB refinement plugin is **not retrained** in this experiment.
- Transferred plugin:
  `results/refinement/refine10000_targetedformal_C500_seed61/refiner_final.pth`

## Results

| Version | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack |
|---|---:|---:|---:|---:|---:|---|
| seed62 base, no plugin | {base62['psnr']:.4f} | {base62['ssim']:.4f} | {base62['clean_ba']:.2f}% | {base62['average_ba']:.2f}% | {base62['minimum_ba']:.2f}% | {base62['minimum_attack']} |
| {reference['name']} | {reference['psnr']:.4f} | {reference['ssim']:.4f} | {reference['clean_ba']:.2f}% | {reference['average_ba']:.2f}% | {reference['minimum_ba']:.2f}% | {reference['minimum_attack']} |
| {transfer['name']} | {transfer['psnr']:.4f} | {transfer['ssim']:.4f} | {transfer['clean_ba']:.2f}% | {transfer['average_ba']:.2f}% | {transfer['minimum_ba']:.2f}% | {transfer['minimum_attack']} |

## Pre-registered reproducibility checks

| Check | Seed62 - Seed61 | Tolerance | Result |
|---|---:|---:|---|
| PSNR | {deltas['psnr']:+.4f} dB | absolute delta <= 1.0 dB | {status(checks['psnr'])} |
| Clean BA | {deltas['clean_ba']:+.2f} pp | absolute delta <= 3.0 pp | {status(checks['clean_ba'])} |
| Nongeom Avg BA | {deltas['average_ba']:+.2f} pp | absolute delta <= 5.0 pp | {status(checks['average_ba'])} |
| Minimum BA | {deltas['minimum_ba']:+.2f} pp | may not drop more than 5.0 pp | {status(checks['minimum_ba'])} |

These thresholds test **cross-seed reproducibility**, not the final thesis quality gate.

## Decision

{decision}
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
