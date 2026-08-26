#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_csv", required=True)
    parser.add_argument("--actual10k_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def setup_font() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False


def load_data(small_csv: Path, actual10k_csv: Path) -> pd.DataFrame:
    small = pd.read_csv(small_csv)
    actual = pd.read_csv(actual10k_csv)

    small = small[["num_images", "method", "total_seconds", "ms_per_image", "speedup_vs_diffw_like"]].copy()
    actual = actual[["num_images", "method", "total_seconds", "ms_per_image", "speedup_vs_diffw_like"]].copy()
    df = pd.concat([small, actual], ignore_index=True)
    df["num_images"] = df["num_images"].astype(int)
    df["total_seconds"] = df["total_seconds"].astype(float)
    df["ms_per_image"] = df["ms_per_image"].astype(float)
    df["speedup_vs_diffw_like"] = df["speedup_vs_diffw_like"].astype(float)
    return df.sort_values(["method", "num_images"])


def fmt_seconds(seconds: float) -> str:
    return f"{seconds/60:.1f} min" if seconds >= 60 else f"{seconds:.1f} s"


def plot_total_time(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.6))
    colors = {"DiffW-like": "#c0392b", "O256-Base": "#2e86de", "O256-RGB": "#16a085"}
    markers = {"DiffW-like": "o", "O256-Base": "s", "O256-RGB": "^"}

    for method in ["DiffW-like", "O256-Base", "O256-RGB"]:
        sub = df[df["method"] == method].sort_values("num_images")
        ax.plot(
            sub["num_images"],
            sub["total_seconds"],
            label=method,
            color=colors[method],
            marker=markers[method],
            linewidth=2.2,
            markersize=7,
        )
        last = sub.iloc[-1]
        ax.scatter(last["num_images"], last["total_seconds"], s=110, facecolors="none", edgecolors=colors[method], linewidths=2.0)
        ax.annotate(
            "10k deploy",
            (last["num_images"], last["total_seconds"]),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=9,
            color=colors[method],
        )

    ax.set_title("End-to-End Deployment Time vs Number of Images")
    ax.set_xlabel("Number of images")
    ax.set_ylabel("Total wall-clock time (seconds)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_xticks([100, 300, 1000, 10000])
    fig.tight_layout()
    out = output_dir / "deployment_time_vs_num_images.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ms_per_image(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.6))
    colors = {"DiffW-like": "#c0392b", "O256-Base": "#2e86de", "O256-RGB": "#16a085"}
    markers = {"DiffW-like": "o", "O256-Base": "s", "O256-RGB": "^"}

    for method in ["DiffW-like", "O256-Base", "O256-RGB"]:
        sub = df[df["method"] == method].sort_values("num_images")
        ax.plot(
            sub["num_images"],
            sub["ms_per_image"],
            label=method,
            color=colors[method],
            marker=markers[method],
            linewidth=2.2,
            markersize=7,
        )
        last = sub.iloc[-1]
        ax.scatter(last["num_images"], last["ms_per_image"], s=110, facecolors="none", edgecolors=colors[method], linewidths=2.0)

    ax.set_title("Per-Image Deployment Cost vs Number of Images")
    ax.set_xlabel("Number of images")
    ax.set_ylabel("ms / image")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_xticks([100, 300, 1000, 10000])
    fig.tight_layout()
    out = output_dir / "deployment_ms_per_image_vs_num_images.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_speedup(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.6))
    colors = {"O256-Base": "#2e86de", "O256-RGB": "#16a085"}
    markers = {"O256-Base": "s", "O256-RGB": "^"}

    for method in ["O256-Base", "O256-RGB"]:
        sub = df[df["method"] == method].sort_values("num_images")
        ax.plot(
            sub["num_images"],
            sub["speedup_vs_diffw_like"],
            label=method,
            color=colors[method],
            marker=markers[method],
            linewidth=2.2,
            markersize=7,
        )
        last = sub.iloc[-1]
        ax.scatter(last["num_images"], last["speedup_vs_diffw_like"], s=110, facecolors="none", edgecolors=colors[method], linewidths=2.0)

    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title("Deployment Speedup over DiffW-like")
    ax.set_xlabel("Number of images")
    ax.set_ylabel("Speedup (x)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_xticks([100, 300, 1000, 10000])
    fig.tight_layout()
    out = output_dir / "deployment_speedup_vs_num_images.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_note(df: pd.DataFrame, output_dir: Path, figure_paths: list[Path]) -> Path:
    note_path = output_dir / "deployment_throughput_figures_note.md"
    with note_path.open("w", encoding="utf-8") as f:
        f.write("# Deployment Throughput Figures Note\n\n")
        f.write("- `100 / 300 / 1000` points come from locked `finaltest1000` subsets.\n")
        f.write("- `10000` points come from the actual deployment benchmark on `train10000`.\n")
        f.write("- The hollow marker at `10000` is used to distinguish the deployment benchmark from the smaller locked-subset points.\n\n")
        f.write("## Figure Files\n")
        for path in figure_paths:
            f.write(f"- `{path}`\n")
    return note_path


def main() -> None:
    args = parse_args()
    setup_font()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(Path(args.small_csv), Path(args.actual10k_csv))

    figs = [
        plot_total_time(df, output_dir),
        plot_ms_per_image(df, output_dir),
        plot_speedup(df, output_dir),
    ]
    note = write_note(df, output_dir, figs)
    print("Saved figures:")
    for fig in figs:
        print(fig)
    print("Saved note:")
    print(note)


if __name__ == "__main__":
    main()
