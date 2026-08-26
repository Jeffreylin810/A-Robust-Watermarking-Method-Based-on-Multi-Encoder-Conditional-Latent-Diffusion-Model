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
    base = read_variant(root, "m03_formal_seed61_base_finaltest1000", "m03_formal_seed61_base_finaltest1000_nongeom")
    refined = read_variant(root, "m03_formal_seed61_fullmodule_finaltest1000", "m03_formal_seed61_fullmodule_finaltest1000_nongeom")

    attack_names = sorted(set(base["attacks"]) | set(refined["attacks"]))
    attack_rows = []
    for name in attack_names:
        b = base["attacks"].get(name)
        r = refined["attacks"].get(name)
        delta = None if b is None or r is None else r - b
        attack_rows.append(
            "| {} | {} | {} | {} |".format(
                name,
                f"{b:.2f}" if b is not None else "N/A",
                f"{r:.2f}" if r is not None else "N/A",
                f"{delta:+.2f}" if delta is not None else "N/A",
            )
        )

    text = f"""# M03 Formal Seed61 Finaltest1000 Summary

Classification: F (formal chapter-4 main results)

This step evaluates the frozen formal artifacts on the locked `finaltest1000` split:

- Formal base from `M01`
- Formal full RGB refinement module from `M02`

## Main Results

| Method | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack |
|---|---:|---:|---:|---:|---:|---|
| O256 base, no module | {base['psnr']:.4f} | {base['ssim']:.4f} | {base['clean_ba']:.2f}% | {base['avg_ba']:.2f}% | {base['min_ba']:.2f}% | {base['min_attack']} |
| O256 + RGB refinement module | {refined['psnr']:.4f} | {refined['ssim']:.4f} | {refined['clean_ba']:.2f}% | {refined['avg_ba']:.2f}% | {refined['min_ba']:.2f}% | {refined['min_attack']} |

## Delta (module - base)

| Metric | Delta |
|---|---:|
| PSNR | {refined['psnr'] - base['psnr']:+.4f} |
| SSIM | {refined['ssim'] - base['ssim']:+.4f} |
| Clean BA | {refined['clean_ba'] - base['clean_ba']:+.2f} pp |
| Nongeom Avg BA | {refined['avg_ba'] - base['avg_ba']:+.2f} pp |
| Minimum BA | {refined['min_ba'] - base['min_ba']:+.2f} pp |

## Per-Attack BA

| Attack | Base | Base + Module | Delta |
|---|---:|---:|---:|
{chr(10).join(attack_rows)}
"""

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
