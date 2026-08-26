#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


DEFAULT_SUMMARIES = [
    "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/17_supplementary_o256_rgb_refiner_refresh_plus10k_seed61_20260630/summary.csv",
    "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/18_supplementary_o256_base_plus15k_then_refresh500_seed61_20260630/summary.csv",
    "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/19_supplementary_o256_rgb_refiner_refresh_plus15k_refresh1000_seed61_20260630/summary.csv",
    "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/20_supplementary_o256_rgb_refiner_refresh_plus15k_refresh1500_seed61_20260630/summary.csv",
    "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/21_supplementary_o256_base_plus20k_then_refresh500_seed61_20260630/summary.csv",
    "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/22_supplementary_o256_rgb_refiner_refresh_plus20k_refresh1500_seed61_20260630/summary.csv",
]

OUTPUT_DIR = Path("/home/jeffrey/projects/實驗結果/14_mainline_formal_protocol/o256_tradeoff_figures_20260630")

LABELS = {
    "formal_rgb": "Formal RGB",
    "plus10k_rgb_reuse": "+10k reuse",
    "refresh500_plus10k": "+10k refresh500",
    "plus15k_rgb_reuse": "+15k reuse",
    "plus15k_refresh500": "+15k refresh500",
    "plus15k_refresh1000": "+15k refresh1000",
    "plus15k_refresh1500": "+15k refresh1500",
    "plus20k_rgb_reuse": "+20k reuse",
    "plus20k_refresh500": "+20k refresh500",
    "plus20k_refresh1500": "+20k refresh1500",
}

COLORS = {
    "formal_rgb": "#6b7280",
    "plus10k_rgb_reuse": "#2563eb",
    "refresh500_plus10k": "#3b82f6",
    "plus15k_rgb_reuse": "#059669",
    "plus15k_refresh500": "#10b981",
    "plus15k_refresh1000": "#34d399",
    "plus15k_refresh1500": "#6ee7b7",
    "plus20k_rgb_reuse": "#b45309",
    "plus20k_refresh500": "#f59e0b",
    "plus20k_refresh1500": "#fbbf24",
}

PREFERRED_ORDER = [
    "formal_rgb",
    "plus10k_rgb_reuse",
    "refresh500_plus10k",
    "plus15k_rgb_reuse",
    "plus15k_refresh500",
    "plus15k_refresh1000",
    "plus15k_refresh1500",
    "plus20k_rgb_reuse",
    "plus20k_refresh500",
    "plus20k_refresh1500",
]


def setup_font():
    preferred = [
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Noto Sans",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in preferred:
        if font_name in available:
            plt.rcParams["font.family"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_rows(summary_paths: list[Path]):
    by_variant = {}
    for path in summary_paths:
      if not path.exists():
        continue
      with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
          variant = row["variant"]
          if variant not in by_variant:
            by_variant[variant] = {
              "variant": variant,
              "psnr": float(row["psnr"]),
              "ssim": float(row["ssim"]),
              "lpips": float(row["lpips"]),
              "clean_ba": float(row["clean_ba"]),
              "avg_ba": float(row["avg_ba"]),
              "min_ba": float(row["min_ba"]),
            }
    return [by_variant[v] for v in PREFERRED_ORDER if v in by_variant]


def annotate_points(ax, rows, x_key, y_key):
    for row in rows:
        ax.annotate(
            LABELS.get(row["variant"], row["variant"]),
            (row[x_key], row[y_key]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )


def scatter_tradeoff(rows, y_key, y_label, title, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for row in rows:
        ax.scatter(
            row["psnr"],
            row[y_key],
            s=70,
            color=COLORS.get(row["variant"], "#333333"),
            edgecolor="white",
            linewidth=0.8,
        )
    annotate_points(ax, rows, "psnr", y_key)
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def line_family(rows, out_path: Path):
    ordered = [r for r in rows if r["variant"] in PREFERRED_ORDER]
    x = list(range(len(ordered)))
    labels = [LABELS.get(r["variant"], r["variant"]) for r in ordered]
    psnr = [r["psnr"] for r in ordered]
    avg = [r["avg_ba"] for r in ordered]
    min_ba = [r["min_ba"] for r in ordered]

    fig, ax1 = plt.subplots(figsize=(10.0, 5.2))
    ax1.plot(x, psnr, marker="o", linewidth=2.0, color="#2563eb", label="PSNR")
    ax1.set_ylabel("PSNR (dB)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, avg, marker="s", linewidth=1.8, color="#059669", label="Avg BA")
    ax2.plot(x, min_ba, marker="^", linewidth=1.8, color="#b45309", label="Min BA")
    ax2.set_ylabel("BA (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=28, ha="right")
    ax1.set_title("O256 operating-point sweep")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--summary_csv", action="append", default=[])
    args = parser.parse_args()

    setup_font()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = [Path(p) for p in (args.summary_csv or DEFAULT_SUMMARIES)]
    rows = load_rows(summary_paths)
    if not rows:
        raise SystemExit("No rows loaded from summary CSV files.")

    scatter_tradeoff(
        rows,
        "avg_ba",
        "Average BA (%)",
        "O256 trade-off: PSNR vs Avg BA",
        output_dir / "o256_tradeoff_psnr_vs_avg_ba.png",
    )
    scatter_tradeoff(
        rows,
        "min_ba",
        "Minimum BA (%)",
        "O256 trade-off: PSNR vs Min BA",
        output_dir / "o256_tradeoff_psnr_vs_min_ba.png",
    )
    line_family(rows, output_dir / "o256_tradeoff_family_curve.png")

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# O256 Trade-off Figures",
                "",
                "Generated figures:",
                "- `o256_tradeoff_psnr_vs_avg_ba.png`",
                "- `o256_tradeoff_psnr_vs_min_ba.png`",
                "- `o256_tradeoff_family_curve.png`",
                "",
                "Input summaries:",
                *[f"- `{p}`" for p in summary_paths if p.exists()],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
