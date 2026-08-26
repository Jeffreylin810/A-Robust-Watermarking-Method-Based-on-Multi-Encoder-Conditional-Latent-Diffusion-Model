#!/usr/bin/env python3
"""Summarize DiffW-like post-hoc quality sweep metrics into CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ATTACK_RE = re.compile(r"^(?P<name>.+?)\s+\|\s+(?P<value>\d+(?:\.\d+)?)\s*$")
PSNR_RE = re.compile(r"Testing set, PSNR is (?P<psnr>\d+\.\d+) and SSIM is (?P<ssim>\d+\.\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", required=True)
    parser.add_argument("--target_psnr", type=float, default=32.9034)
    return parser.parse_args()


def parse_psnr_log(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        match = PSNR_RE.search(line)
        if match:
            return float(match.group("psnr")), float(match.group("ssim"))
    return None, None


def parse_lpips_summary(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["mean_lpips"])


def parse_ba_report(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = ATTACK_RE.match(line.strip())
        if match:
            values[match.group("name")] = float(match.group("value"))
    return values


def gamma_value_from_tag(tag: str) -> float:
    return float(tag.replace("gamma_", "").replace("p", "."))


def main() -> None:
    args = parse_args()
    exp_root = Path(args.exp_root)
    scaled_root = exp_root / "scaled_outputs"
    summary_csv = exp_root / "diffw_quality_sweep_summary.csv"
    per_attack_csv = exp_root / "diffw_quality_sweep_per_attack.csv"

    tags = sorted([p.name for p in scaled_root.iterdir() if p.is_dir() and p.name.startswith("gamma_")])
    if not tags:
        raise RuntimeError(f"No gamma directories found under: {scaled_root}")

    summary_rows: list[dict[str, str]] = []
    attack_rows: list[dict[str, str]] = []

    for tag in tags:
        gamma = gamma_value_from_tag(tag)
        psnr, ssim = parse_psnr_log(exp_root / "logs" / f"{tag}_psnr_ssim.log")
        lpips = parse_lpips_summary(exp_root / "lpips" / tag / "summary.json")
        clean_values = parse_ba_report(exp_root / "attacked_Result" / f"{tag}_clean" / "ba_report.txt")
        attack_values = parse_ba_report(exp_root / "attacked_Result" / f"{tag}_nongeom" / "ba_report.txt")

        clean_ba = clean_values.get("Clean")
        per_attack_items = [(k, v) for k, v in attack_values.items() if k != "Clean"]
        avg_ba = None
        min_ba = None
        min_attack = ""
        if per_attack_items:
            avg_ba = sum(v for _, v in per_attack_items) / len(per_attack_items)
            min_attack, min_ba = min(per_attack_items, key=lambda item: item[1])

        summary_rows.append(
            {
                "gamma": f"{gamma:.2f}",
                "psnr": "" if psnr is None else f"{psnr:.4f}",
                "ssim": "" if ssim is None else f"{ssim:.4f}",
                "lpips": "" if lpips is None else f"{lpips:.6f}",
                "clean_ba": "" if clean_ba is None else f"{clean_ba:.2f}",
                "nongeom_avg_ba": "" if avg_ba is None else f"{avg_ba:.2f}",
                "min_ba": "" if min_ba is None else f"{min_ba:.2f}",
                "min_attack": min_attack,
                "psnr_gap_to_o256_rgb": ""
                if psnr is None
                else f"{abs(psnr - args.target_psnr):.4f}",
            }
        )

        for attack_name, value in sorted(per_attack_items):
            attack_rows.append(
                {"gamma": f"{gamma:.2f}", "attack_name": attack_name, "ba": f"{value:.2f}"}
            )

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gamma",
                "psnr",
                "ssim",
                "lpips",
                "clean_ba",
                "nongeom_avg_ba",
                "min_ba",
                "min_attack",
                "psnr_gap_to_o256_rgb",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with per_attack_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gamma", "attack_name", "ba"])
        writer.writeheader()
        writer.writerows(attack_rows)

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved per-attack CSV: {per_attack_csv}")


if __name__ == "__main__":
    main()
