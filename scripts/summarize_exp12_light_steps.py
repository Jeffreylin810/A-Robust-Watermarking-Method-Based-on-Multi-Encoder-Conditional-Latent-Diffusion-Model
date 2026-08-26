#!/usr/bin/env python3
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "results/experiment_summaries/exp12_nongeomlight_5k_vs_10k_seed61"

RUNS = [
    {
        "name": "identity20k -> light5k (25k total)",
        "prefix": "exp2b_nongeom_train10000_base20k_then_light5k_seed61",
        "clean": "clean_exp2b_train10000_base20k_then_light5k_seed61",
        "attack": "exp2b_nongeom_train10000_base20k_then_light5k_seed61",
        "extra_steps": 5000,
    },
    {
        "name": "identity20k -> light10k (30k total)",
        "prefix": "exp12_identity20k_then_nongeomlight10k_seed61",
        "clean": "clean_exp12_identity20k_then_nongeomlight10k_seed61",
        "attack": "exp12_identity20k_then_nongeomlight10k_seed61_nongeom",
        "extra_steps": 10000,
    },
]


def parse_psnr(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"Testing set,\s*PSNR is\s*([0-9.]+)\s*and SSIM is\s*([0-9.]+)", text
    )
    if not matches:
        raise RuntimeError(f"Could not parse PSNR/SSIM from {path}")
    return tuple(float(value) for value in matches[-1])


def parse_clean(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^\s*Clean\s*\|\s*([0-9.]+)\s*$", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse Clean BA from {path}")
    return float(match.group(1))


def load(spec):
    psnr, ssim = parse_psnr(ROOT / "logs" / f"{spec['prefix']}_psnr_ssim.log")
    clean = parse_clean(
        ROOT / "results/attacked_Result" / spec["clean"] / "ba_report.txt"
    )
    attack = json.loads(
        (
            ROOT
            / "results/attacked_Result"
            / spec["attack"]
            / "ba_summary.json"
        ).read_text(encoding="utf-8")
    )
    return {
        **spec,
        "psnr": psnr,
        "ssim": ssim,
        "clean_ba": clean,
        "average_ba": float(attack["average_ba"]),
        "minimum_ba": float(attack["minimum_ba"]),
        "minimum_attack": attack["minimum_attack"],
    }


def main():
    base, extended = [load(spec) for spec in RUNS]
    delta = {
        key: extended[key] - base[key]
        for key in ("psnr", "ssim", "clean_ba", "average_ba", "minimum_ba")
    }
    robustness_gain = delta["average_ba"] >= 1.0 or delta["minimum_ba"] >= 1.0
    quality_ok = delta["clean_ba"] >= -1.0 and delta["psnr"] >= -0.3
    select_extended = robustness_gain and quality_ok
    selected = extended if select_extended else base

    if select_extended:
        decision = (
            "Adopt light10k: it provides at least +1 pp robustness improvement "
            "without exceeding the PSNR/Clean BA degradation limits."
        )
    else:
        decision = (
            "Keep light5k: the extra 5k updates do not provide enough robustness "
            "gain to justify their additional cost, or they exceed a quality limit."
        )

    lines = [
        "# Experiment 12: Nongeom-Light 5k vs 10k",
        "",
        "Classification: E (exploratory; dev1000 only)",
        "",
        "| Version | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in (base, extended):
        lines.append(
            f"| {row['name']} | {row['psnr']:.4f} | {row['ssim']:.4f} | "
            f"{row['clean_ba']:.2f}% | {row['average_ba']:.2f}% | "
            f"{row['minimum_ba']:.2f}% | {row['minimum_attack']} |"
        )
    lines += [
        "",
        "## Delta: light10k - light5k",
        "",
        f"- PSNR: {delta['psnr']:+.4f} dB",
        f"- SSIM: {delta['ssim']:+.4f}",
        f"- Clean BA: {delta['clean_ba']:+.2f} pp",
        f"- Nongeom Avg BA: {delta['average_ba']:+.2f} pp",
        f"- Minimum BA: {delta['minimum_ba']:+.2f} pp",
        "",
        "## Pre-registered decision rule",
        "",
        "- Avg BA or Minimum BA must improve by at least 1.0 pp.",
        "- Clean BA may not fall by more than 1.0 pp.",
        "- PSNR may not fall by more than 0.3 dB.",
        "",
        f"Robustness gain condition: **{robustness_gain}**",
        f"Quality preservation condition: **{quality_ok}**",
        "",
        "## Decision",
        "",
        decision,
        "",
        f"Selected base curriculum: **{selected['name']}**",
        "",
    ]

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (SUMMARY_DIR / "summary.json").write_text(
        json.dumps(
            {
                "base": base,
                "extended": extended,
                "delta": delta,
                "robustness_gain_condition": robustness_gain,
                "quality_preservation_condition": quality_ok,
                "selected": selected["name"],
                "decision": decision,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(SUMMARY_DIR / "summary.md")


if __name__ == "__main__":
    main()

