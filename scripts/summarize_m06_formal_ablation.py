#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def parse_last(patterns, text, label):
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    raise RuntimeError(f"Could not parse {label}")


def read_attack_rows(path: Path):
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"^\s*(\S.*?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*$", line)
        if match and match.group(1) != "Attack Name":
            rows[match.group(1).strip()] = float(match.group(2))
    if not rows:
        raise RuntimeError(f"No BA rows found in {path}")
    return rows


def read_variant(root: Path, prefix: str, attack_run: str):
    logs = root / "logs"
    psnr_text = (logs / f"{prefix}_psnr_ssim.log").read_text(encoding="utf-8", errors="ignore")
    clean_text = (logs / f"{prefix}_clean_ba.log").read_text(encoding="utf-8", errors="ignore")
    attack_dir = root / "results" / "attacked_Result" / attack_run
    attack_summary = json.loads((attack_dir / "ba_summary.json").read_text(encoding="utf-8"))
    attacks = read_attack_rows(attack_dir / "ba_report.txt")
    return {
        "psnr": parse_last((r"Testing set, PSNR is ([0-9.]+)",), psnr_text, "PSNR"),
        "ssim": parse_last((r"SSIM is ([0-9.]+)",), psnr_text, "SSIM"),
        "clean_ba": parse_last(
            (
                r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9.]+)%",
                r"Clean\s+\|\s+([0-9.]+)",
            ),
            clean_text,
            "Clean BA",
        ),
        "avg_ba": float(attack_summary["average_ba"]),
        "min_ba": float(attack_summary["minimum_ba"]),
        "min_attack": attack_summary["minimum_attack"],
        "attacks": attacks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.project_root)
    variants = [
        ("O256 base", "m03_formal_seed61_base_finaltest1000", "m03_formal_seed61_base_finaltest1000_nongeom"),
        ("Stage A", "m06_formal_seed61_stageA_finaltest1000", "m06_formal_seed61_stageA_finaltest1000_nongeom"),
        ("Stage B", "m06_formal_seed61_stageB_finaltest1000", "m06_formal_seed61_stageB_finaltest1000_nongeom"),
        ("Stage B+", "m06_formal_seed61_stageBP_finaltest1000", "m06_formal_seed61_stageBP_finaltest1000_nongeom"),
        ("Full module", "m03_formal_seed61_fullmodule_finaltest1000", "m03_formal_seed61_fullmodule_finaltest1000_nongeom"),
    ]
    data = {label: read_variant(root, prefix, attack_run) for label, prefix, attack_run in variants}
    base = data["O256 base"]

    attack_names = sorted({name for value in data.values() for name in value["attacks"].keys()})
    metric_rows = []
    for label in ["O256 base", "Stage A", "Stage B", "Stage B+", "Full module"]:
        value = data[label]
        delta_psnr = value["psnr"] - base["psnr"]
        delta_clean = value["clean_ba"] - base["clean_ba"]
        delta_avg = value["avg_ba"] - base["avg_ba"]
        delta_min = value["min_ba"] - base["min_ba"]
        metric_rows.append(
            f"| {label} | {value['psnr']:.4f} | {value['ssim']:.4f} | {value['clean_ba']:.2f}% | "
            f"{value['avg_ba']:.2f}% | {value['min_ba']:.2f}% | {value['min_attack']} | "
            f"{delta_psnr:+.4f} | {delta_clean:+.2f} pp | {delta_avg:+.2f} pp | {delta_min:+.2f} pp |"
        )

    attack_table_lines = []
    for name in attack_names:
        row = [name]
        for label in ["O256 base", "Stage A", "Stage B", "Stage B+", "Full module"]:
            value = data[label]["attacks"].get(name)
            row.append(f"{value:.2f}" if value is not None else "N/A")
        attack_table_lines.append("| " + " | ".join(row) + " |")

    text = f"""# M06 Formal Stage-wise Ablation Summary

Classification: F (formal chapter-4 ablation)

This table reuses the frozen formal base and evaluates stage-wise RGB refinement checkpoints on the same locked `finaltest1000` split:

- O256 base
- Stage A only
- Stage B
- Stage B+
- Full module

## Main Table

| Variant | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack | dPSNR vs base | dClean BA | dAvg BA | dMin BA |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

## Per-Attack BA

| Attack | O256 base | Stage A | Stage B | Stage B+ | Full module |
|---|---:|---:|---:|---:|---:|
{chr(10).join(attack_table_lines)}

## Interpretation Template

- Stage A should show the strongest image-fidelity recovery tendency, but may reduce message robustness.
- Stage B / Stage B+ should progressively recover Clean BA and non-geometric robustness.
- Full module should represent the final trade-off after targeted weak non-geometric message preservation is added.
"""

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
