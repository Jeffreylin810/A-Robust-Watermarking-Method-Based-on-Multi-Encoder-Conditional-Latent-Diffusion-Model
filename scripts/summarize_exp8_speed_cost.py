#!/usr/bin/env python3
"""Create the Experiment 8 speed and cost summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REAL_PATTERN = re.compile(r"^real\s+([0-9.]+)\s*$", re.MULTILINE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffw_benchmark", required=True)
    parser.add_argument("--ours_benchmark", required=True)
    parser.add_argument("--diffw_train_log", required=True)
    parser.add_argument("--base20_log", required=True)
    parser.add_argument("--base5_log", required=True)
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


def benchmark_rows(payload):
    return {(row["batch_size"], row["mode"]): row for row in payload["results"]}


def main():
    args = parse_args()
    diffw = json.loads(Path(args.diffw_benchmark).read_text(encoding="utf-8"))
    ours = json.loads(Path(args.ours_benchmark).read_text(encoding="utf-8"))
    diffw_rows = benchmark_rows(diffw)
    ours_rows = benchmark_rows(ours)

    costs = {
        "DiffW-like base training": last_real(args.diffw_train_log),
        "O256 identity-only 20k": last_real(args.base20_log),
        "O256 nongeom-light continuation 5k": last_real(args.base5_log),
        "Generate refine10000 pairs": last_real(args.pairs_log),
        "Plugin Stage A": last_real(args.stage_a_log),
        "Plugin Stage B": last_real(args.stage_b_log),
        "Plugin Stage B+": last_real(args.stage_bplus_log),
        "Plugin targeted Stage C": last_real(args.stage_c_log),
    }
    ours_base = costs["O256 identity-only 20k"] + costs["O256 nongeom-light continuation 5k"]
    plugin_prep = sum(
        value
        for key, value in costs.items()
        if key.startswith("Generate") or key.startswith("Plugin")
    )
    ours_total = ours_base + plugin_prep
    diffw_total = costs["DiffW-like base training"]

    lines = [
        "# Experiment 8: formal speed and cost evaluation",
        "",
        "## Protocol",
        "",
        f"- GPU: {ours['gpu']}",
        f"- Sampling steps: {ours['sampling_timesteps']}",
        f"- Warm-up: {ours['warmup']} iterations",
        f"- Repeats: {ours['repeats']}",
        "- Synchronized compute-only timing excludes dataset I/O, PNG writing, PSNR/SSIM, and BA.",
        "- Native sampling CPU/GPU intermediate-tensor transfers remain included.",
        "- Training/preparation cost uses wall-clock `real` time from the original logs.",
        "",
        "## Compute-only inference",
        "",
        "| Batch | Method | ms/image | std | image/s | CV | Peak incremental MiB |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for batch_size in sorted({key[0] for key in diffw_rows}):
        ordered = [
            ("DiffW-like", diffw_rows[(batch_size, "base")]),
            ("O256 base", ours_rows[(batch_size, "base")]),
            ("Plugin only", ours_rows[(batch_size, "refiner_only")]),
            ("O256 base + plugin", ours_rows[(batch_size, "base_plus_refiner")]),
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
            f"| **O256 base total** | **{ours_base:.2f}** | **{duration(ours_base)}** |",
            f"| **Plugin preparation total** | **{plugin_prep:.2f}** | **{duration(plugin_prep)}** |",
            f"| **O256 complete preparation** | **{ours_total:.2f}** | **{duration(ours_total)}** |",
            "",
            "## Interpretation",
            "",
            f"- O256 base training speed ratio vs DiffW-like: **{diffw_total / ours_base:.3f}x faster**.",
            f"- Complete preparation ratio DiffW-like/O256+plugin: **{diffw_total / ours_total:.3f}x**.",
            f"- Plugin adds **{plugin_prep:.2f} s ({duration(plugin_prep)})** of one-time preparation.",
        ]
    )

    batch1_base = ours_rows[(1, "base")]["mean_ms_per_image"]
    batch1_full = ours_rows[(1, "base_plus_refiner")]["mean_ms_per_image"]
    batch1_plugin = ours_rows[(1, "refiner_only")]["mean_ms_per_image"]
    lines.extend(
        [
            f"- Batch-1 plugin-only overhead: **{batch1_plugin:.4f} ms/image**.",
            f"- Batch-1 measured end-to-end increase: **{batch1_full - batch1_base:.4f} ms/image "
            f"({100.0 * (batch1_full / batch1_base - 1.0):.2f}%)**.",
            "",
            "Do not claim a speed advantage unless the comparison scope is stated explicitly:",
            "`base training`, `complete preparation`, or `inference` answer different questions.",
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved summary: {output}")


if __name__ == "__main__":
    main()
