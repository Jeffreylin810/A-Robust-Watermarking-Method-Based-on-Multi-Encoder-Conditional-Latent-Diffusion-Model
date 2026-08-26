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
    parser.add_argument("--diffw_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.diffw_root)
    psnr_text = (root / "logs/m04_formal_diffw_like_finaltest1000_psnr_ssim.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    clean_text = (root / "logs/m04_formal_diffw_like_finaltest1000_clean_ba.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    attack_dir = root / "results/attacked_Result/m04_formal_diffw_like_finaltest1000_nongeom"
    attack_summary = json.loads((attack_dir / "ba_summary.json").read_text(encoding="utf-8"))
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

    text = f"""# M04 Formal DiffW-like Finaltest1000 Summary

Classification: F (formal chapter-4 baseline results)

This step trains the formal-clean DiffW-like 256x256 / 32-bit baseline from
random initialization for 20k steps, then evaluates the frozen final checkpoint
once on the locked `finaltest1000` split.

## Main Results

| Method | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack |
|---|---:|---:|---:|---:|---:|---|
| DiffW-like | {psnr:.4f} | {ssim:.4f} | {clean_ba:.2f}% | {float(attack_summary['average_ba']):.2f}% | {float(attack_summary['minimum_ba']):.2f}% | {attack_summary['minimum_attack']} |

## Per-Attack BA

| Attack | BA (%) |
|---|---:|
{chr(10).join(f"| {name} | {value:.2f} |" for name, value in sorted(attacks.items()))}
"""

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
