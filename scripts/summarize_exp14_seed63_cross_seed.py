#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


REFERENCE = {
    "name": "seed61 base + seed61 Full module",
    "apply_report": "results/refinement/exp13_full_final_seed61_apply_eval1000/report.json",
    "clean_report": (
        "results/refinement/exp13_full_final_seed61_apply_eval1000/"
        "attacked_clean/ba_report.txt"
    ),
    "attack_summary": (
        "results/attacked_Result/exp13_full_final_seed61_nongeom/ba_summary.json"
    ),
}

TRANSFER = {
    "name": "seed63 base + transferred seed61 Full module",
    "apply_report": (
        "results/refinement/exp14_seed63_with_seed61_fullmodule_apply_eval1000/"
        "report.json"
    ),
    "clean_report": (
        "results/refinement/exp14_seed63_with_seed61_fullmodule_apply_eval1000/"
        "attacked_clean/ba_report.txt"
    ),
    "attack_summary": (
        "results/attacked_Result/exp14_seed63_with_seed61_fullmodule_nongeom/"
        "ba_summary.json"
    ),
}

BASE63 = {
    "psnr_log": "logs/exp14_seed63_base30k_psnr_ssim.log",
    "clean_report": "results/attacked_Result/clean_exp14_seed63_base30k/ba_report.txt",
    "attack_summary": (
        "results/attacked_Result/exp14_seed63_base30k_nongeom/ba_summary.json"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_clean_ba(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    patterns = (
        r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9]+(?:\.[0-9]+)?)%",
        r"^\s*Clean\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            return float(matches[-1])
    raise RuntimeError(f"Could not parse Clean BA from {path}")


def read_apply(root, spec):
    payload = json.loads((root / spec["apply_report"]).read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    attack = json.loads((root / spec["attack_summary"]).read_text(encoding="utf-8"))
    return {
        "name": spec["name"],
        "psnr": float(metrics["psnr"]),
        "ssim": float(metrics["ssim"]),
        "clean_ba": read_clean_ba(root / spec["clean_report"]),
        "average_ba": float(attack["average_ba"]),
        "minimum_ba": float(attack["minimum_ba"]),
        "minimum_attack": attack["minimum_attack"],
    }


def read_base63(root):
    text = (root / BASE63["psnr_log"]).read_text(encoding="utf-8", errors="ignore")
    psnr_matches = re.findall(r"Testing set,\s*PSNR is\s*([0-9.]+)", text)
    ssim_matches = re.findall(r"SSIM is\s*([0-9.]+)", text)
    if not psnr_matches or not ssim_matches:
        raise RuntimeError("Could not parse seed63 base PSNR/SSIM")
    attack = json.loads((root / BASE63["attack_summary"]).read_text(encoding="utf-8"))
    return {
        "psnr": float(psnr_matches[-1]),
        "ssim": float(ssim_matches[-1]),
        "clean_ba": read_clean_ba(root / BASE63["clean_report"]),
        "average_ba": float(attack["average_ba"]),
        "minimum_ba": float(attack["minimum_ba"]),
        "minimum_attack": attack["minimum_attack"],
    }


def status(value):
    return "PASS" if value else "FAIL"


def main():
    args = parse_args()
    root = Path(args.project_root)
    reference = read_apply(root, REFERENCE)
    transfer = read_apply(root, TRANSFER)
    base63 = read_base63(root)

    deltas = {
        "psnr": transfer["psnr"] - reference["psnr"],
        "ssim": transfer["ssim"] - reference["ssim"],
        "clean_ba": transfer["clean_ba"] - reference["clean_ba"],
        "average_ba": transfer["average_ba"] - reference["average_ba"],
        "minimum_ba": transfer["minimum_ba"] - reference["minimum_ba"],
    }
    strict_checks = {
        "base_clean_ba": base63["clean_ba"] >= 90.0,
        "psnr": abs(deltas["psnr"]) <= 1.0,
        "clean_ba": abs(deltas["clean_ba"]) <= 3.0,
        "average_ba": abs(deltas["average_ba"]) <= 5.0,
        "minimum_ba": deltas["minimum_ba"] >= -5.0,
    }
    transfer_checks = {
        "base_clean_ba": base63["clean_ba"] >= 90.0,
        "psnr": deltas["psnr"] >= -1.0,
        "clean_ba": deltas["clean_ba"] >= -3.0,
        "average_ba": deltas["average_ba"] >= -5.0,
        "minimum_ba": deltas["minimum_ba"] >= -5.0,
    }
    strict_passed = all(strict_checks.values())
    transfer_passed = all(transfer_checks.values())

    if not transfer_checks["base_clean_ba"]:
        decision = (
            "INVALID BASE：seed63 base Clean BA 未達 90%，因此不能判斷模組是否能跨 seed。"
            " 先診斷 base training，再談模組轉移。"
        )
    elif transfer_passed and strict_passed:
        decision = (
            "PASS：seed61 Full 模組可直接轉移到 seed63 base，且所有預先固定的"
            " cross-seed 容許範圍均通過。"
        )
    elif transfer_passed:
        decision = (
            "PASS TRANSFER / STRICT EQUIVALENCE FLAG：seed61 Full 模組可直接轉移到"
            " seed63。所有品質指標都沒有出現超出容許範圍的負向退化；若有嚴格重現性"
            "檢查未通過，代表不同 seed 帶來可觀變異，不代表模組轉移失敗。"
        )
    else:
        decision = (
            "FAIL TRANSFER：至少一項品質指標出現超出容許範圍的負向退化。下一步才需要"
            "考慮使用 seed63 重新產生 refine10000 pairs，並訓練 seed63 專屬模組。"
        )

    text = f"""# Experiment 14: seed63 cross-seed Full-module reproducibility

## Controlled design

- Dataset/split, architecture, message length, VAE, losses, evaluation images,
  and nongeom attack suite are unchanged.
- The only intended base-training change is `seed: 61 -> 63`.
- Base recipe is the locked 30k recipe:
  `identity 20k -> nongeom-light 10k`.
- The RGB refinement module is **not retrained** in this experiment.
- Transferred module:
  `results/refinement/exp13_full_A10000_B2000_BP500_C500/refiner_final.pth`

## Results

| Version | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack |
|---|---:|---:|---:|---:|---:|---|
| seed63 base, no module | {base63['psnr']:.4f} | {base63['ssim']:.4f} | {base63['clean_ba']:.2f}% | {base63['average_ba']:.2f}% | {base63['minimum_ba']:.2f}% | {base63['minimum_attack']} |
| {reference['name']} | {reference['psnr']:.4f} | {reference['ssim']:.4f} | {reference['clean_ba']:.2f}% | {reference['average_ba']:.2f}% | {reference['minimum_ba']:.2f}% | {reference['minimum_attack']} |
| {transfer['name']} | {transfer['psnr']:.4f} | {transfer['ssim']:.4f} | {transfer['clean_ba']:.2f}% | {transfer['average_ba']:.2f}% | {transfer['minimum_ba']:.2f}% | {transfer['minimum_attack']} |

## Pre-registered reproducibility checks

| Check | Seed63 - Seed61 | Tolerance | Result |
|---|---:|---:|---|
| Seed63 base Clean BA precondition | {base63['clean_ba']:.2f}% | >= 90.0% | {status(strict_checks['base_clean_ba'])} |
| PSNR | {deltas['psnr']:+.4f} dB | absolute delta <= 1.0 dB | {status(strict_checks['psnr'])} |
| Clean BA | {deltas['clean_ba']:+.2f} pp | absolute delta <= 3.0 pp | {status(strict_checks['clean_ba'])} |
| Nongeom Avg BA | {deltas['average_ba']:+.2f} pp | absolute delta <= 5.0 pp | {status(strict_checks['average_ba'])} |
| Minimum BA | {deltas['minimum_ba']:+.2f} pp | may not drop more than 5.0 pp | {status(strict_checks['minimum_ba'])} |

These thresholds test **cross-seed reproducibility**, not the final thesis quality gate.

## Transfer usability checks

For transfer usability, positive improvements are not treated as failures.

| Check | Seed63 - Seed61 | Allowed negative degradation | Result |
|---|---:|---:|---|
| PSNR | {deltas['psnr']:+.4f} dB | may not drop more than 1.0 dB | {status(transfer_checks['psnr'])} |
| Clean BA | {deltas['clean_ba']:+.2f} pp | may not drop more than 3.0 pp | {status(transfer_checks['clean_ba'])} |
| Nongeom Avg BA | {deltas['average_ba']:+.2f} pp | may not drop more than 5.0 pp | {status(transfer_checks['average_ba'])} |
| Minimum BA | {deltas['minimum_ba']:+.2f} pp | may not drop more than 5.0 pp | {status(transfer_checks['minimum_ba'])} |

## Decision

{decision}
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "reference": reference,
                "base63": base63,
                "transfer": transfer,
                "deltas": deltas,
                "strict_checks": strict_checks,
                "transfer_checks": transfer_checks,
                "strict_passed": strict_passed,
                "transfer_passed": transfer_passed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(text)


if __name__ == "__main__":
    main()
