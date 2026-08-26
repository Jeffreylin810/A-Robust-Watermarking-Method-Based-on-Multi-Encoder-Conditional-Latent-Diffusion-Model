#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(
    "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/efficiency_amortized_cost_experiment_20260624"
)
OUT = ROOT / "figures"
EXTRA_RGB_SECONDS = 914.26 + 3127.57


def hours(seconds: float) -> float:
    return seconds / 3600.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(ROOT / "training_timing_results.csv")
    measured_steps = [5000, 10000, 20000, 40000]
    projected_steps = [40000, 60000, 80000, 100000]

    pivot = training.pivot(index="num_steps", columns="method", values="total_seconds").sort_index()
    per_step = training.groupby("method")["seconds_per_step"].mean().to_dict()

    diffw_measured = [pivot.loc[s, "DiffW-like"] for s in measured_steps]
    base_measured = [pivot.loc[s, "O256-Base"] for s in measured_steps]
    rgb_measured = [pivot.loc[s, "O256-Base"] + EXTRA_RGB_SECONDS for s in measured_steps]

    # Anchor projection at the measured 40k point and extend using mean measured seconds/step.
    anchor = 40000
    diffw_anchor = pivot.loc[anchor, "DiffW-like"]
    base_anchor = pivot.loc[anchor, "O256-Base"]
    diffw_projected = [
        diffw_anchor + (s - anchor) * per_step["DiffW-like"] for s in projected_steps
    ]
    base_projected = [
        base_anchor + (s - anchor) * per_step["O256-Base"] for s in projected_steps
    ]
    rgb_projected = [v + EXTRA_RGB_SECONDS for v in base_projected]

    plt.figure(figsize=(8.0, 4.8))
    plt.plot(
        measured_steps,
        [hours(v) for v in diffw_measured],
        marker="o",
        linewidth=2.3,
        color="#3b4cc0",
        label="DiffW-like measured",
    )
    plt.plot(
        measured_steps,
        [hours(v) for v in base_measured],
        marker="o",
        linewidth=2.3,
        color="#2f9e44",
        label="O256-Base measured",
    )
    plt.plot(
        measured_steps,
        [hours(v) for v in rgb_measured],
        marker="o",
        linewidth=2.3,
        color="#d9480f",
        label="O256-RGB total prep. measured-based",
    )

    plt.plot(
        projected_steps,
        [hours(v) for v in diffw_projected],
        marker="o",
        linewidth=2.0,
        linestyle="--",
        color="#3b4cc0",
        alpha=0.82,
        label="DiffW-like projection",
    )
    plt.plot(
        projected_steps,
        [hours(v) for v in base_projected],
        marker="o",
        linewidth=2.0,
        linestyle="--",
        color="#2f9e44",
        alpha=0.82,
        label="O256-Base projection",
    )
    plt.plot(
        projected_steps,
        [hours(v) for v in rgb_projected],
        marker="o",
        linewidth=2.0,
        linestyle="--",
        color="#d9480f",
        alpha=0.82,
        label="O256-RGB total prep. projection",
    )

    plt.axvline(40000, color="#777777", linestyle=":", linewidth=1.2)
    plt.text(
        41000,
        0.35,
        "Projection after 40k\n(not measured)",
        fontsize=9,
        color="#555555",
        va="bottom",
    )
    plt.xlabel("Training steps")
    plt.ylabel("Training / preparation cost (hours)")
    plt.title("Projected Preparation Cost up to 100k Steps")
    plt.grid(True, alpha=0.28)
    plt.ylim(bottom=0)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    png = OUT / "fig_B_projection_to_100k.png"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()

    rows = []
    for s, d, b, r, source in [
        *zip(measured_steps, diffw_measured, base_measured, rgb_measured, ["measured"] * 4),
        *zip(projected_steps[1:], diffw_projected[1:], base_projected[1:], rgb_projected[1:], ["projected"] * 3),
    ]:
        rows.append(
            {
                "steps": s,
                "source": source,
                "diffw_hours": hours(d),
                "o256_base_hours": hours(b),
                "o256_rgb_total_prep_hours": hours(r),
                "o256_rgb_minus_diffw_hours": hours(r - d),
            }
        )
    csv_path = OUT / "fig_B_projection_to_100k_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    md = OUT / "fig_B_projection_to_100k_summary.md"
    md.write_text(
        "\n".join(
            [
                "# Projection to 100k Steps",
                "",
                f"- figure: `{png}`",
                f"- data: `{csv_path}`",
                "- measured range: 5k, 10k, 20k, 40k steps",
                "- projected range: 60k, 80k, 100k steps",
                "- projection method: anchored at measured 40k point and extended using mean measured seconds/step from Experiment A.",
                f"- RGB one-time extra cost: `{EXTRA_RGB_SECONDS:.2f}` seconds = `{hours(EXTRA_RGB_SECONDS):.4f}` hours.",
                "",
                "Important caption note:",
                "",
                "> Solid lines show measured timing benchmark results from 5k to 40k steps. Dashed lines show a projection to 100k steps based on the measured mean seconds/step and are not measured 100k training results. The O256-RGB curve adds the one-time pair generation and RGB refinement training cost to the O256-Base timing benchmark.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {png}")
    print(f"saved {csv_path}")
    print(f"saved {md}")


if __name__ == "__main__":
    main()
