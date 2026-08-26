#!/usr/bin/env python3
"""Summarize formal speed/cost results for seed61 locked artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REAL_PATTERN = re.compile(r"^real\s+([0-9.]+)\s*$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffw_benchmark", required=True)
    parser.add_argument("--ours_benchmark", required=True)
    parser.add_argument("--diffw_train_log", required=True)
    parser.add_argument("--base20_log", required=True)
    parser.add_argument("--base10_log", required=True)
    parser.add_argument("--pairs_log", required=True)
    parser.add_argument("--stage_a_log", required=True)
    parser.add_argument("--stage_b_log", required=True)
    parser.add_argument("--stage_bplus_log", required=True)
    parser.add_argument("--stage_c_log", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def last_real(path: str) -> float:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    values = REAL_PATTERN.findall(text)
    if not values:
        raise RuntimeError(f"Could not parse real time from {path}")
    return float(values[-1])


def duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def benchmark_rows(payload: dict) -> dict[tuple[int, str], dict]:
    return {(row["batch_size"], row["mode"]): row for row in payload["results"]}


def main() -> None:
    args = parse_args()
    diffw = json.loads(Path(args.diffw_benchmark).read_text(encoding="utf-8"))
    ours = json.loads(Path(args.ours_benchmark).read_text(encoding="utf-8"))
    diffw_rows = benchmark_rows(diffw)
    ours_rows = benchmark_rows(ours)

    costs = {
        "DiffW-like base training (20k)": last_real(args.diffw_train_log),
        "O256 identity-only base training (20k)": last_real(args.base20_log),
        "O256 nongeom-light continuation (10k)": last_real(args.base10_log),
        "Formal refine10000 pair generation": last_real(args.pairs_log),
        "Module Stage A (10k)": last_real(args.stage_a_log),
        "Module Stage B (2k)": last_real(args.stage_b_log),
        "Module Stage B+ (500)": last_real(args.stage_bplus_log),
        "Module targeted Stage C (500)": last_real(args.stage_c_log),
    }
    o256_base_total = (
        costs["O256 identity-only base training (20k)"]
        + costs["O256 nongeom-light continuation (10k)"]
    )
    module_prep_total = sum(
        value
        for key, value in costs.items()
        if key.startswith("Formal refine10000") or key.startswith("Module ")
    )
    o256_complete_total = o256_base_total + module_prep_total
    diffw_total = costs["DiffW-like base training (20k)"]

    lines = [
        "# Formal Efficiency Summary (seed61)",
        "",
        "## Scope",
        "",
        "- Classification: **F** (chapter-4 formal efficiency experiment).",
        "- Artifacts: locked seed61 formal checkpoints only.",
        "- DiffW-like uses the formal-clean 20k baseline from `M04`.",
        "- O256 uses the formal base from `M01` and the formal RGB refinement module from `M02`.",
        "- All training/preparation cost numbers below use wall-clock `real` time parsed from the formal logs.",
        "- Inference timing is synchronized compute-only GPU timing; it excludes dataset I/O, PNG writing, PSNR/SSIM, and BA evaluation.",
        "",
        "## Compute-only inference",
        "",
        f"- GPU: {ours['gpu']}",
        f"- Sampling steps: {ours['sampling_timesteps']}",
        f"- Warm-up: {ours['warmup']} iterations",
        f"- Repeats: {ours['repeats']}",
        "",
        "| Batch | Method | ms/image | std | image/s | CV | Peak incremental MiB |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]

    for batch_size in sorted({key[0] for key in diffw_rows}):
        ordered = [
            ("DiffW-like", diffw_rows[(batch_size, "base")]),
            ("O256 base", ours_rows[(batch_size, "base")]),
            ("RGB refinement module only", ours_rows[(batch_size, "refiner_only")]),
            ("O256 + module", ours_rows[(batch_size, "base_plus_refiner")]),
        ]
        for label, row in ordered:
            lines.append(
                f"| {batch_size} | {label} | {row['mean_ms_per_image']:.4f} | "
                f"{row['std_ms_per_image']:.4f} | {row['images_per_second']:.2f} | "
                f"{row['cv_percent']:.2f}% | {row['peak_incremental_mib']:.1f} |"
            )

    lines.extend(
        [
            "",
            "## One-time training and preparation cost",
            "",
            "| Stage | Seconds | Wall-clock |",
            "|---|---:|---:|",
        ]
    )
    for key, value in costs.items():
        lines.append(f"| {key} | {value:.2f} | {duration(value)} |")
    lines.extend(
        [
            f"| **O256 base training total** | **{o256_base_total:.2f}** | **{duration(o256_base_total)}** |",
            f"| **RGB refinement module preparation total** | **{module_prep_total:.2f}** | **{duration(module_prep_total)}** |",
            f"| **O256 complete preparation total** | **{o256_complete_total:.2f}** | **{duration(o256_complete_total)}** |",
            f"| **DiffW-like complete preparation total** | **{diffw_total:.2f}** | **{duration(diffw_total)}** |",
            "",
            "## Method-level totals",
            "",
            "| Method | Base training | Extra preparation | Complete preparation |",
            "|---|---:|---:|---:|",
            f"| DiffW-like | {duration(diffw_total)} | 00:00:00.00 | {duration(diffw_total)} |",
            f"| O256 base | {duration(o256_base_total)} | 00:00:00.00 | {duration(o256_base_total)} |",
            f"| O256-RGBREFINE | {duration(o256_base_total)} | {duration(module_prep_total)} | {duration(o256_complete_total)} |",
            "",
            "## Interpretation",
            "",
            f"- O256 base training vs DiffW-like: **{diffw_total / o256_base_total:.3f}x** faster if we compare base-training time only.",
            f"- O256 complete preparation vs DiffW-like: **{o256_complete_total / diffw_total:.3f}x** of DiffW-like's one-time cost when the RGB refinement module is included.",
            f"- RGB refinement module adds **{module_prep_total:.2f} s ({duration(module_prep_total)})** of one-time preparation on top of the O256 base.",
        ]
    )

    batch1_base = ours_rows[(1, "base")]["mean_ms_per_image"]
    batch1_full = ours_rows[(1, "base_plus_refiner")]["mean_ms_per_image"]
    batch1_module = ours_rows[(1, "refiner_only")]["mean_ms_per_image"]
    lines.extend(
        [
            f"- Batch-1 module-only compute overhead: **{batch1_module:.4f} ms/image**.",
            f"- Batch-1 end-to-end increase from O256 base to O256 + module: **{batch1_full - batch1_base:.4f} ms/image ({100.0 * (batch1_full / batch1_base - 1.0):.2f}%)**.",
            "",
            "When writing chapter 4, keep three scopes separate:",
            "1. `base training cost`",
            "2. `complete preparation cost`",
            "3. `inference cost`",
            "",
            "Do not collapse them into one generic 'speed' claim.",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved summary: {output}")


if __name__ == "__main__":
    main()
