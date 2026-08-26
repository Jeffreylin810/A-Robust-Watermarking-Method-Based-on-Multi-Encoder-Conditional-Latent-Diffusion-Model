#!/usr/bin/env python3
"""Summarize repeat-stability inference sanity for formal O256 artifacts."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def rows_by_key(payload: dict) -> dict[tuple[int, str], dict]:
    return {(row["batch_size"], row["mode"]): row for row in payload["results"]}


def ci95(std: float, repeats: int) -> float:
    if repeats <= 0:
        return 0.0
    return 1.96 * std / sqrt(repeats)


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    rows = rows_by_key(payload)
    repeats = int(payload["repeats"])

    lines = [
        "# Formal Inference Sanity (seed61)",
        "",
        "## Scope",
        "",
        "- Classification: **F** (formal benchmark sanity / reporting support).",
        "- Purpose: verify whether `O256 base` and `O256 + RGB refinement module` differ by more than benchmark noise.",
        "- This experiment uses the same locked formal seed61 artifacts as the main chapter-4 results.",
        f"- GPU: {payload['gpu']}",
        f"- Warm-up: {payload['warmup']}",
        f"- Iterations: {payload['iterations']}",
        f"- Module-only iterations: {payload['refiner_iterations']}",
        f"- Repeats: {payload['repeats']}",
        "",
        "## Timing summary",
        "",
        "| Batch | Mode | Mean ms/image | Std | 95% CI half-width | Min run | Max run | CV |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for batch in sorted({key[0] for key in rows}):
        for mode in ("base", "refiner_only", "base_plus_refiner"):
            row = rows[(batch, mode)]
            run_values = row["run_ms_per_image"]
            lines.append(
                f"| {batch} | {mode} | {row['mean_ms_per_image']:.4f} | "
                f"{row['std_ms_per_image']:.4f} | {ci95(row['std_ms_per_image'], repeats):.4f} | "
                f"{min(run_values):.4f} | {max(run_values):.4f} | {row['cv_percent']:.2f}% |"
            )

    lines.extend(["", "## Base vs base+module comparison", ""])
    lines.extend(
        [
            "| Batch | Base mean | Base+module mean | Delta (full-base) | Relative delta | 95% CI overlap? |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )

    conclusions = []
    for batch in sorted({key[0] for key in rows}):
        base = rows[(batch, "base")]
        full = rows[(batch, "base_plus_refiner")]
        delta = full["mean_ms_per_image"] - base["mean_ms_per_image"]
        rel = 100.0 * delta / base["mean_ms_per_image"]
        base_low = base["mean_ms_per_image"] - ci95(base["std_ms_per_image"], repeats)
        base_high = base["mean_ms_per_image"] + ci95(base["std_ms_per_image"], repeats)
        full_low = full["mean_ms_per_image"] - ci95(full["std_ms_per_image"], repeats)
        full_high = full["mean_ms_per_image"] + ci95(full["std_ms_per_image"], repeats)
        overlap = not (base_high < full_low or full_high < base_low)
        lines.append(
            f"| {batch} | {base['mean_ms_per_image']:.4f} | {full['mean_ms_per_image']:.4f} | "
            f"{delta:+.4f} | {rel:+.2f}% | {'Yes' if overlap else 'No'} |"
        )
        if overlap:
            conclusions.append(
                f"- Batch {batch}: base and base+module 95% confidence intervals overlap, so the measured delta should be treated as benchmark noise rather than a real speed difference."
            )
        else:
            conclusions.append(
                f"- Batch {batch}: base and base+module do not overlap under this benchmark; report the delta carefully and do not overgeneralize beyond this GPU/protocol."
            )

    module_only_batch1 = rows[(1, "refiner_only")]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Module-only compute is tiny relative to base sampling: batch-1 module-only mean = **{module_only_batch1['mean_ms_per_image']:.4f} ms/image**.",
            *conclusions,
            "- Use this sanity check to justify the wording `module overhead is negligible relative to base sampling`, not to claim that the module makes sampling faster.",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved summary: {output}")


if __name__ == "__main__":
    main()
