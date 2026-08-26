#!/usr/bin/env python3
"""Summarize message-preserving RGB refinement probe results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_clean_ba(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9.]+)%", text)
    if match:
        return float(match.group(1))
    match = re.search(r"Clean\s+\|\s+([0-9.]+)", text)
    return float(match.group(1)) if match else None


def parse_attack_report(path: Path) -> tuple[float | None, float | None, str]:
    if not path.exists():
        return None, None, ""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "|" not in line or "Attack Name" in line or "---" in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 2:
            continue
        try:
            rows.append((parts[0], float(parts[1])))
        except ValueError:
            continue
    if not rows:
        return None, None, ""
    avg = sum(value for _name, value in rows) / len(rows)
    min_name, min_value = min(rows, key=lambda item: item[1])
    return avg, min_value, min_name


def load_apply_metrics(path: Path) -> dict[str, float | str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    return {
        "psnr": metrics.get("psnr", ""),
        "ssim": metrics.get("ssim", ""),
        "apply_clean_ba": metrics.get("ba", ""),
        "elapsed_seconds": metrics.get("elapsed_seconds", ""),
    }


def fmt(value, digits=4):
    if value is None or value == "":
        return "missing"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("variants", nargs="+", help="Variant names, e.g. latent500 sensitive500")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    rows = []
    for variant in args.variants:
        apply_dir = exp_root / "apply" / variant
        attack_dir = exp_root / "attacked_Result" / f"{variant}_nongeom"
        metrics = load_apply_metrics(apply_dir / "report.json")
        clean_ba = parse_clean_ba(apply_dir / "attacked_clean" / "ba_report.txt")
        avg_ba, min_ba, min_attack = parse_attack_report(attack_dir / "ba_report.txt")
        rows.append(
            {
                "variant": variant,
                "psnr": metrics.get("psnr", ""),
                "ssim": metrics.get("ssim", ""),
                "clean_ba": clean_ba if clean_ba is not None else metrics.get("apply_clean_ba", ""),
                "nongeom_avg_ba": avg_ba if avg_ba is not None else "",
                "minimum_ba": min_ba if min_ba is not None else "",
                "minimum_attack": min_attack,
                "apply_report": str(apply_dir / "report.json"),
                "clean_ba_report": str(apply_dir / "attacked_clean" / "ba_report.txt"),
                "nongeom_ba_report": str(attack_dir / "ba_report.txt"),
            }
        )

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    table = "\n".join(
        "| {variant} | {psnr} | {ssim} | {clean_ba} | {nongeom_avg_ba} | {minimum_ba} | {minimum_attack} |".format(
            variant=row["variant"],
            psnr=fmt(row["psnr"]),
            ssim=fmt(row["ssim"]),
            clean_ba=fmt(row["clean_ba"], 2),
            nongeom_avg_ba=fmt(row["nongeom_avg_ba"], 2),
            minimum_ba=fmt(row["minimum_ba"], 2),
            minimum_attack=row["minimum_attack"] or "missing",
        )
        for row in rows
    )
    md = f"""# Message-Preserving RGB Refinement Probe Summary

## Results

| Variant | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack |
|---|---:|---:|---:|---:|---:|---|
{table}

## Notes

- This is a supplementary probe, not a replacement for the formal chapter-4 main result.
- The first gate is PSNR/SSIM and Clean BA on finaltest1000.
- Full non-geometric attacks should be interpreted only if the corresponding `nongeom_ba_report` exists.

## Source Files

- CSV: `{csv_path}`
"""
    Path(args.output_md).write_text(md, encoding="utf-8")
    print(f"saved: {csv_path}")
    print(f"saved: {args.output_md}")


if __name__ == "__main__":
    main()
