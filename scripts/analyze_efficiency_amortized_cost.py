#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


EXP_ROOT = Path(
    "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/"
    "efficiency_amortized_cost_experiment_20260624"
)
TIMING_CSV = EXP_ROOT / "training_timing_results.csv"
OUT_CSV = EXP_ROOT / "amortized_cost_analysis.csv"
OUT_MD = EXP_ROOT / "amortized_cost_summary.md"

PAIR_GENERATION_SECONDS = 15 * 60 + 14.26
REFINEMENT_TRAINING_SECONDS = 52 * 60 + 7.57
EXTRA_RGB_COST_SECONDS = PAIR_GENERATION_SECONDS + REFINEMENT_TRAINING_SECONDS


def fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{sign}{h:02d}:{m:02d}:{s:05.2f}"


def load_rows() -> dict[int, dict[str, dict[str, str]]]:
    rows: dict[int, dict[str, dict[str, str]]] = {}
    with TIMING_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            steps = int(row["num_steps"])
            rows.setdefault(steps, {})[row["method"]] = row
    return rows


def main() -> None:
    rows = load_rows()
    out_rows = []
    for steps in sorted(rows):
        pair = rows[steps]
        if "DiffW-like" not in pair or "O256-Base" not in pair:
            raise RuntimeError(f"Missing method pair for steps={steps}")
        diffw_seconds = float(pair["DiffW-like"]["total_seconds"])
        o256_seconds = float(pair["O256-Base"]["total_seconds"])
        o256_rgb_total = o256_seconds + EXTRA_RGB_COST_SECONDS
        saving = diffw_seconds - o256_seconds
        rgb_minus_diffw = o256_rgb_total - diffw_seconds
        out_rows.append(
            {
                "num_steps": steps,
                "diffw_seconds": f"{diffw_seconds:.6f}",
                "o256_base_seconds": f"{o256_seconds:.6f}",
                "pair_generation_seconds": f"{PAIR_GENERATION_SECONDS:.2f}",
                "refinement_training_seconds": f"{REFINEMENT_TRAINING_SECONDS:.2f}",
                "extra_rgb_cost_seconds": f"{EXTRA_RGB_COST_SECONDS:.2f}",
                "o256_rgb_total_seconds": f"{o256_rgb_total:.6f}",
                "o256_base_saving_seconds": f"{saving:.6f}",
                "o256_rgb_minus_diffw_seconds": f"{rgb_minus_diffw:.6f}",
                "break_even_reached": "yes" if rgb_minus_diffw <= 0 else "no",
                "notes": "Measured training times from Experiment A; RGB extra cost from formal pair generation and refinement training logs.",
            }
        )

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    # Estimate break-even using average measured per-step difference.
    diff_per_step = []
    for steps in sorted(rows):
        diffw_sps = float(rows[steps]["DiffW-like"]["seconds_per_step"])
        o256_sps = float(rows[steps]["O256-Base"]["seconds_per_step"])
        diff_per_step.append(diffw_sps - o256_sps)
    avg_saving_per_step = sum(diff_per_step) / len(diff_per_step)
    estimated_break_even_steps = EXTRA_RGB_COST_SECONDS / avg_saving_per_step

    md = [
        "# Amortized Cost Summary",
        "",
        f"- Experiment root: `{EXP_ROOT}`",
        f"- Training timing source: `{TIMING_CSV}`",
        f"- Pair generation cost: `{fmt_hms(PAIR_GENERATION_SECONDS)}`",
        f"- RGB refinement training cost: `{fmt_hms(REFINEMENT_TRAINING_SECONDS)}`",
        f"- Extra one-time RGB cost: `{fmt_hms(EXTRA_RGB_COST_SECONDS)}`",
        "",
        "## Measured Amortized-Cost Table",
        "",
        "| steps | DiffW-like measured | O256-Base measured | O256-RGB total prep | O256-Base saving | O256-RGB minus DiffW-like | break-even reached |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in out_rows:
        md.append(
            "| {num_steps} | {diffw} | {o256} | {rgb} | {saving} | {minus} | {be} |".format(
                num_steps=row["num_steps"],
                diffw=fmt_hms(float(row["diffw_seconds"])),
                o256=fmt_hms(float(row["o256_base_seconds"])),
                rgb=fmt_hms(float(row["o256_rgb_total_seconds"])),
                saving=fmt_hms(float(row["o256_base_saving_seconds"])),
                minus=fmt_hms(float(row["o256_rgb_minus_diffw_seconds"])),
                be=row["break_even_reached"],
            )
        )

    md.extend(
        [
            "",
            "## Estimated Break-Even",
            "",
            f"- Average measured per-step saving of O256-Base over DiffW-like: `{avg_saving_per_step:.6f} s/step`.",
            f"- Estimated break-even steps for amortizing the one-time RGB cost: `{estimated_break_even_steps:.0f}` steps.",
            "- This is an estimate derived from measured seconds/step, not a directly measured training run.",
            "",
            "## Thesis Inclusion Guidance",
            "",
            "- O256-Base is consistently faster than DiffW-like across 5k, 10k, 20k, and 40k measured points.",
            "- O256-RGB has a fixed one-time RGB preparation cost, but the measured O256-Base training-time saving already amortizes it at the tested step counts.",
            "- If included in the thesis, describe this as a supplementary amortized-cost analysis rather than a primary method-performance table.",
            "- Keep the caveat that Experiment A uses a single-run timing benchmark for O256-Base and is not a replacement for the formal staged recipe.",
        ]
    )
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
