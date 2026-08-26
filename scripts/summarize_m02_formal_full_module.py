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


def read_attack_rows(path):
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"^\s*(\S.*?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*$", line)
        if match and match.group(1) != "Attack Name":
            rows[match.group(1).strip()] = float(match.group(2))
    if not rows:
        raise RuntimeError(f"No BA rows found in {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.project_root)
    logs = root / "logs"
    prefix = "m02_formal_seed61_fullmodule_dev1000"
    attack_run = "m02_formal_seed61_fullmodule_dev1000_nongeom"

    psnr_text = (logs / f"{prefix}_psnr_ssim.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    clean_text = (logs / f"{prefix}_clean_ba.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    attack_dir = root / "results" / "attacked_Result" / attack_run
    attack_summary = json.loads(
        (attack_dir / "ba_summary.json").read_text(encoding="utf-8")
    )
    attacks = read_attack_rows(attack_dir / "ba_report.txt")

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
    avg_ba = float(attack_summary["average_ba"])
    min_ba = float(attack_summary["minimum_ba"])
    min_attack = attack_summary["minimum_attack"]

    text = f"""# M02 Formal Seed61 Full Module Summary

Classification: F (formal artifact generation; dev1000 health-check only)

This step uses the formal `M01` base checkpoint and retrains the full RGB refinement
module from random initialization using the locked recipe:

```text
Stage A 10k
-> Stage B 2k
-> Stage B+ 500
-> targeted Stage C 500
```

No finaltest1000 evaluation is performed in this step.

## Outputs

- Formal refine10000 pair root:
  `results/test_result/m02_formal_seed61_refine10000_pairs`
- Formal full refiner:
  `results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth`

## dev1000 health-check

| PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack |
|---:|---:|---:|---:|---:|---|
| {psnr:.4f} | {ssim:.4f} | {clean_ba:.2f}% | {avg_ba:.2f}% | {min_ba:.2f}% | {min_attack} |

## Non-geometric attacks

| Attack | BA (%) |
|---|---:|
{chr(10).join(f"| {name} | {value:.2f} |" for name, value in sorted(attacks.items()))}

## Interpretation

If these dev1000 health-check values are normal, the next formal step is:

```text
freeze the formal base + formal full module
and evaluate once on finaltest1000
```
"""

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
