#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "diffw": "#3b4cc0",
    "base": "#2f9e44",
    "rgb": "#d9480f",
    "gray": "#666666",
}


def hours(seconds: pd.Series | float) -> pd.Series | float:
    return seconds / 3600.0


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_a_training_time(training: pd.DataFrame, out: Path) -> None:
    pivot = training.pivot(index="num_steps", columns="method", values="total_seconds").sort_index()
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(pivot.index, hours(pivot["DiffW-like"]), marker="o", linewidth=2.2, color=COLORS["diffw"], label="DiffW-like")
    plt.plot(pivot.index, hours(pivot["O256-Base"]), marker="o", linewidth=2.2, color=COLORS["base"], label="O256-Base")
    plt.xlabel("Training steps")
    plt.ylabel("Measured training time (hours)")
    plt.title("Experiment A: Training-Time Scaling")
    plt.grid(True, alpha=0.28)
    plt.ylim(bottom=0)
    plt.legend(frameon=False)
    savefig(out / "fig_A_training_time_vs_steps.png")


def plot_a_seconds_per_step(training: pd.DataFrame, out: Path) -> None:
    pivot = training.pivot(index="num_steps", columns="method", values="seconds_per_step").sort_index()
    x = range(len(pivot.index))
    width = 0.36
    plt.figure(figsize=(7.2, 4.2))
    plt.bar([i - width / 2 for i in x], pivot["DiffW-like"], width=width, color=COLORS["diffw"], label="DiffW-like")
    plt.bar([i + width / 2 for i in x], pivot["O256-Base"], width=width, color=COLORS["base"], label="O256-Base")
    plt.xticks(list(x), [f"{int(s/1000)}k" for s in pivot.index])
    plt.xlabel("Training steps")
    plt.ylabel("Seconds per step")
    plt.title("Experiment A: Per-Step Training Cost")
    plt.grid(axis="y", alpha=0.28)
    plt.ylim(bottom=0)
    plt.legend(frameon=False)
    savefig(out / "fig_A_seconds_per_step.png")


def plot_b_preparation_cost(amortized: pd.DataFrame, out: Path) -> None:
    df = amortized.sort_values("num_steps")
    plt.figure(figsize=(7.6, 4.7))
    plt.plot(df["num_steps"], hours(df["diffw_seconds"]), marker="o", linewidth=2.2, color=COLORS["diffw"], label="DiffW-like")
    plt.plot(df["num_steps"], hours(df["o256_base_seconds"]), marker="o", linewidth=2.2, color=COLORS["base"], label="O256-Base")
    plt.plot(df["num_steps"], hours(df["o256_rgb_total_seconds"]), marker="o", linewidth=2.2, color=COLORS["rgb"], label="O256-RGB total prep.")
    plt.xlabel("Training steps")
    plt.ylabel("Preparation / training cost (hours)")
    plt.title("Experiment B: Amortized Preparation Cost")
    plt.grid(True, alpha=0.28)
    plt.ylim(bottom=0)
    plt.legend(frameon=False)
    savefig(out / "fig_B_amortized_preparation_cost.png")


def plot_b_difference(amortized: pd.DataFrame, out: Path) -> None:
    df = amortized.sort_values("num_steps")
    diff_hours = hours(df["o256_rgb_minus_diffw_seconds"])
    plt.figure(figsize=(7.4, 4.4))
    plt.plot(df["num_steps"], diff_hours, marker="o", linewidth=2.4, color=COLORS["rgb"], label="O256-RGB total prep. - DiffW-like")
    plt.axhline(0, color="black", linewidth=1.1, linestyle="--", label="Break-even")
    for x, y in zip(df["num_steps"], diff_hours):
        plt.annotate(f"{y:+.2f} h", (x, y), textcoords="offset points", xytext=(0, 8 if y >= 0 else -16), ha="center", fontsize=8)
    plt.xlabel("Training steps")
    plt.ylabel("Cost difference vs DiffW-like (hours)")
    plt.title("Experiment B: Break-Even View")
    plt.grid(True, alpha=0.28)
    plt.legend(frameon=False)
    savefig(out / "fig_B_break_even_difference.png")


def plot_b_ratio(amortized: pd.DataFrame, out: Path) -> None:
    df = amortized.sort_values("num_steps").copy()
    df["base_ratio"] = df["o256_base_seconds"] / df["diffw_seconds"]
    df["rgb_ratio"] = df["o256_rgb_total_seconds"] / df["diffw_seconds"]
    plt.figure(figsize=(7.4, 4.4))
    plt.plot(df["num_steps"], df["base_ratio"], marker="o", linewidth=2.2, color=COLORS["base"], label="O256-Base / DiffW-like")
    plt.plot(df["num_steps"], df["rgb_ratio"], marker="o", linewidth=2.2, color=COLORS["rgb"], label="O256-RGB total prep. / DiffW-like")
    plt.axhline(1.0, color="black", linewidth=1.1, linestyle="--", label="DiffW-like = 1.0")
    plt.xlabel("Training steps")
    plt.ylabel("Cost ratio relative to DiffW-like")
    plt.title("Experiment B: Normalized Cost Ratio")
    plt.grid(True, alpha=0.28)
    plt.ylim(bottom=0)
    plt.legend(frameon=False)
    savefig(out / "fig_B_cost_ratio.png")


def plot_c_reuse_quality(reuse: pd.DataFrame, out: Path) -> None:
    df = reuse.copy()
    df["subset_n"] = df["num_images"]
    for metric, ylabel, filename in [
        ("psnr", "PSNR (dB)", "fig_C_reuse_psnr.png"),
        ("clean_ba", "Clean BA (%)", "fig_C_reuse_clean_ba.png"),
        ("nongeom_avg_ba", "Nongeom Avg BA (%)", "fig_C_reuse_nongeom_avg_ba.png"),
        ("min_ba", "Minimum BA (%)", "fig_C_reuse_min_ba.png"),
    ]:
        pivot = df.pivot(index="subset_n", columns="method", values=metric).sort_index()
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(pivot.index, pivot["O256-Base"], marker="o", linewidth=2.2, color=COLORS["base"], label="O256-Base")
        plt.plot(pivot.index, pivot["O256-RGB"], marker="o", linewidth=2.2, color=COLORS["rgb"], label="O256-RGB")
        plt.xlabel("Unseen COCO subset size")
        plt.ylabel(ylabel)
        plt.title(f"Experiment C: Reuse Stability ({ylabel})")
        plt.grid(True, alpha=0.28)
        plt.legend(frameon=False)
        savefig(out / filename)


def plot_c_overhead(reuse: pd.DataFrame, out: Path) -> None:
    rgb = reuse[reuse["method"] == "O256-RGB"].sort_values("num_images")
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(rgb["num_images"], rgb["refinement_overhead_percent"], marker="o", linewidth=2.2, color=COLORS["rgb"])
    plt.xlabel("Unseen COCO subset size")
    plt.ylabel("RGB refinement inference overhead (%)")
    plt.title("Experiment C: Inference Overhead")
    plt.grid(True, alpha=0.28)
    plt.ylim(bottom=0)
    savefig(out / "fig_C_reuse_inference_overhead.png")


def write_summary(out: Path) -> None:
    lines = [
        "# ABC Efficiency Figures",
        "",
        "Generated figures for discussion with the advisor.",
        "",
        "## Figure Files",
        "",
        "| Experiment | Figure | Purpose |",
        "|---|---|---|",
        "| A | `fig_A_training_time_vs_steps.png` | Measured DiffW-like vs O256-Base training time as training steps increase. |",
        "| A | `fig_A_seconds_per_step.png` | Per-step training cost comparison. |",
        "| B | `fig_B_amortized_preparation_cost.png` | DiffW-like, O256-Base, and O256-RGB total preparation cost. |",
        "| B | `fig_B_break_even_difference.png` | O256-RGB total preparation cost minus DiffW-like; zero line marks break-even. |",
        "| B | `fig_B_cost_ratio.png` | Normalized cost ratio relative to DiffW-like. |",
        "| C | `fig_C_reuse_psnr.png` | O256-RGB reuse image-quality stability. |",
        "| C | `fig_C_reuse_clean_ba.png` | Clean BA stability on unseen COCO subsets. |",
        "| C | `fig_C_reuse_nongeom_avg_ba.png` | Non-geometric average BA stability. |",
        "| C | `fig_C_reuse_min_ba.png` | Minimum BA stability. |",
        "| C | `fig_C_reuse_inference_overhead.png` | RGB refinement inference overhead. |",
        "",
        "## Interpretation Guardrail",
        "",
        "- Experiment A varies training steps, not training-set size. It supports that O256-Base has a stable per-step speed advantage over DiffW-like under the same benchmark protocol.",
        "- Experiment B supports that the one-time RGB refinement preparation cost is amortized around the 20k-step setting and becomes favorable at longer training schedules.",
        "- Experiment C supports that the trained RGB Refinement Module can be reused on unseen COCO subsets without retraining, with high PSNR and small inference overhead.",
        "- These figures should use full, clearly labeled axes. Do not visually exaggerate the gap by truncating axes without explicit annotation.",
    ]
    (out / "ABC_efficiency_figures_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/efficiency_amortized_cost_experiment_20260624",
    )
    args = parser.parse_args()
    root = Path(args.root)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(root / "training_timing_results.csv")
    amortized = pd.read_csv(root / "amortized_cost_analysis.csv")
    reuse = pd.read_csv(root / "rgb_refinement_reuse_eval.csv")

    plot_a_training_time(training, out)
    plot_a_seconds_per_step(training, out)
    plot_b_preparation_cost(amortized, out)
    plot_b_difference(amortized, out)
    plot_b_ratio(amortized, out)
    plot_c_reuse_quality(reuse, out)
    plot_c_overhead(reuse, out)
    write_summary(out)
    print(f"saved figures to {out}")


if __name__ == "__main__":
    main()
