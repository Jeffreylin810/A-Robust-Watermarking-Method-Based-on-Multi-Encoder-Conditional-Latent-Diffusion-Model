#!/usr/bin/env python3
"""Summarize COCO subset module-reuse experiment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FORMAL_REFERENCE = {
    "name": "formal seed61 full module reference",
    "main_summary": "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_seed61_finaltest1000/summary.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_clean_ba(report: Path) -> float:
    text = report.read_text(encoding="utf-8", errors="ignore")
    patterns = (
        r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9]+(?:\.[0-9]+)?)%",
        r"^\s*Clean\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            return float(matches[-1])
    raise RuntimeError(f"Could not parse Clean BA from {report}")


def parse_psnr_ssim(log_path: Path) -> tuple[float, float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    psnr_matches = re.findall(r"Testing set,\s*PSNR is\s*([0-9.]+)", text)
    ssim_matches = re.findall(r"SSIM is\s*([0-9.]+)", text)
    if not psnr_matches or not ssim_matches:
        raise RuntimeError(f"Could not parse PSNR/SSIM from {log_path}")
    return float(psnr_matches[-1]), float(ssim_matches[-1])


def parse_main_summary(summary_path: Path) -> dict[str, dict[str, float | str]]:
    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    header = None
    rows = {}
    for i, line in enumerate(lines):
        if line.startswith("| Method | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack |"):
            header = i
            break
    if header is None:
        raise RuntimeError(f"Could not find main results table in {summary_path}")
    for line in lines[header + 2 : header + 5]:
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 7:
            continue
        rows[cols[0]] = {
            "psnr": float(cols[1]),
            "ssim": float(cols[2]),
            "clean_ba": float(cols[3].rstrip("%")),
            "average_ba": float(cols[4].rstrip("%")),
            "minimum_ba": float(cols[5].rstrip("%")),
            "minimum_attack": cols[6],
        }
    return rows


def load_variant(root: Path, tag: str) -> dict[str, float | str]:
    base_psnr_log = root / f"logs/{tag}_base30k_psnr_ssim.log"
    base_clean_report = root / f"results/attacked_Result/clean_{tag}_base30k/ba_report.txt"
    base_attack_summary = root / f"results/attacked_Result/{tag}_base30k_nongeom/ba_summary.json"

    transfer_apply = root / f"results/refinement/{tag}_transfer_apply_finaltest1000/report.json"
    transfer_clean_report = root / f"results/refinement/{tag}_transfer_apply_finaltest1000/attacked_clean/ba_report.txt"
    transfer_attack_summary = root / f"results/attacked_Result/{tag}_transfer_nongeom/ba_summary.json"

    base_psnr, base_ssim = parse_psnr_ssim(base_psnr_log)
    base_attack = json.loads(base_attack_summary.read_text(encoding="utf-8"))
    transfer_apply_payload = json.loads(transfer_apply.read_text(encoding="utf-8"))
    transfer_attack = json.loads(transfer_attack_summary.read_text(encoding="utf-8"))

    return {
        "base_psnr": base_psnr,
        "base_ssim": base_ssim,
        "base_clean_ba": parse_clean_ba(base_clean_report),
        "base_average_ba": float(base_attack["average_ba"]),
        "base_minimum_ba": float(base_attack["minimum_ba"]),
        "base_minimum_attack": base_attack["minimum_attack"],
        "transfer_psnr": float(transfer_apply_payload["metrics"]["psnr"]),
        "transfer_ssim": float(transfer_apply_payload["metrics"]["ssim"]),
        "transfer_clean_ba": parse_clean_ba(transfer_clean_report),
        "transfer_average_ba": float(transfer_attack["average_ba"]),
        "transfer_minimum_ba": float(transfer_attack["minimum_ba"]),
        "transfer_minimum_attack": transfer_attack["minimum_attack"],
    }


def evaluate_transfer(reused: dict[str, float | str], ref: dict[str, float | str]) -> dict[str, float | bool]:
    deltas = {
        "psnr": float(reused["transfer_psnr"]) - float(ref["psnr"]),
        "clean_ba": float(reused["transfer_clean_ba"]) - float(ref["clean_ba"]),
        "average_ba": float(reused["transfer_average_ba"]) - float(ref["average_ba"]),
        "minimum_ba": float(reused["transfer_minimum_ba"]) - float(ref["minimum_ba"]),
    }
    checks = {
        "base_clean_ok": float(reused["base_clean_ba"]) >= 95.0,
        "transfer_psnr_ok": deltas["psnr"] >= -1.0,
        "transfer_clean_ok": deltas["clean_ba"] >= -3.0,
        "transfer_average_ok": deltas["average_ba"] >= -3.0,
        "transfer_minimum_ok": deltas["minimum_ba"] >= -2.0,
    }
    checks["transfer_pass"] = all(checks.values())
    return {**deltas, **checks}


def main() -> None:
    args = parse_args()
    root = Path(args.project_root)
    out = Path(args.output)

    reference_rows = parse_main_summary(Path(FORMAL_REFERENCE["main_summary"]))
    formal_full = reference_rows["O256-RGBREFINE"]

    first = load_variant(root, "m07_first5000")
    second = load_variant(root, "m07_second5000")
    first_eval = evaluate_transfer(first, formal_full)
    second_eval = evaluate_transfer(second, formal_full)

    def verdict(ev: dict[str, float | bool]) -> str:
        return "PASS" if ev["transfer_pass"] else "FAIL"

    text = f"""# M07 COCO subset module reuse experiment

Classification: supporting experiment for the **speed / amortized module-cost**
claim. This is **not** a replacement for the chapter-4 main formal results.

## Goal

Test whether a single formal seed61 RGB refinement module can be reused on
COCO-trained base checkpoints that were trained from **different image subsets**
of the same COCO train pool, without retraining the module.

## Design

- Dataset family: COCO only
- Base recipe: `identity20k -> nongeom-light10k`
- Evaluation split: locked `finaltest1000`
- Frozen transferred module:
  `results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth`
- New base checkpoints:
  - first 5000 train images
  - second 5000 train images

## Formal reference

| Reference | PSNR | Clean BA | Nongeom Avg BA | Minimum BA |
|---|---:|---:|---:|---:|
| O256-RGBREFINE formal seed61 | {formal_full['psnr']:.4f} | {formal_full['clean_ba']:.2f}% | {formal_full['average_ba']:.2f}% | {formal_full['minimum_ba']:.2f}% |

## Results

| Variant | Base PSNR | Base Clean BA | Reused-module PSNR | Reused-module Clean BA | Reused-module Avg BA | Reused-module Min BA | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| first5000 subset | {first['base_psnr']:.4f} | {first['base_clean_ba']:.2f}% | {first['transfer_psnr']:.4f} | {first['transfer_clean_ba']:.2f}% | {first['transfer_average_ba']:.2f}% | {first['transfer_minimum_ba']:.2f}% | {verdict(first_eval)} |
| second5000 subset | {second['base_psnr']:.4f} | {second['base_clean_ba']:.2f}% | {second['transfer_psnr']:.4f} | {second['transfer_clean_ba']:.2f}% | {second['transfer_average_ba']:.2f}% | {second['transfer_minimum_ba']:.2f}% | {verdict(second_eval)} |

## Transfer gate

Transfer is counted as acceptable only if all of the following hold:

- base Clean BA >= 95.0%
- reused-module PSNR drop relative to formal full <= 1.0 dB
- reused-module Clean BA drop <= 3.0 pp
- reused-module Nongeom Avg BA drop <= 3.0 pp
- reused-module Minimum BA drop <= 2.0 pp

## Delta vs formal full reference

| Variant | Delta PSNR | Delta Clean BA | Delta Avg BA | Delta Min BA | Pass? |
|---|---:|---:|---:|---:|---|
| first5000 subset | {first_eval['psnr']:+.4f} | {first_eval['clean_ba']:+.2f} pp | {first_eval['average_ba']:+.2f} pp | {first_eval['minimum_ba']:+.2f} pp | {verdict(first_eval)} |
| second5000 subset | {second_eval['psnr']:+.4f} | {second_eval['clean_ba']:+.2f} pp | {second_eval['average_ba']:+.2f} pp | {second_eval['minimum_ba']:+.2f} pp | {verdict(second_eval)} |

## Interpretation

- If both subsets pass, the thesis may conservatively argue that under the same
  COCO family and the same base recipe, the refinement module shows evidence of
  **checkpoint-level reusability**, so the module cost may be discussed as a
  dataset-family amortized preparation cost rather than a mandatory per-checkpoint cost.
- If either subset fails, the thesis should not claim module reuse as a general
  assumption even within COCO; the module should then remain a checkpoint-specific
  one-time cost.
"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "formal_full_reference": formal_full,
                "first5000": {**first, **first_eval},
                "second5000": {**second, **second_eval},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Summary: {out}")


if __name__ == "__main__":
    main()
