#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


PROJECTION_N = 111000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", required=True)
    return parser.parse_args()


def setup_font() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_hhmmss(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def parse_time_log(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"^real\s+([0-9.]+)\s*$", text, flags=re.MULTILINE)
    if not matches:
        return None
    return float(matches[-1])


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compute_ci95(row: dict) -> float:
    runs = row.get("run_ms_per_image") or []
    std = float(row.get("std_ms_per_image", 0.0))
    n = max(1, len(runs))
    return 1.96 * std / math.sqrt(n)


def summarize_compute(exp_root: Path) -> list[dict]:
    raw_dir = exp_root / "raw"
    paths = [
        raw_dir / "compute_diffw_like.json",
        raw_dir / "compute_o256_rgb.json",
    ]
    if not all(path.exists() for path in paths):
        return []

    rows: list[dict] = []
    for path in paths:
        payload = read_json(path)
        gpu = payload.get("gpu", "unknown")
        warmup = payload.get("warmup", "")
        iterations = payload.get("iterations", "")
        repeats = payload.get("repeats", "")
        for result in payload["results"]:
            batch = int(result["batch_size"])
            mode = result["mode"]
            if "DiffW-like" in payload["method_name"]:
                if mode != "base":
                    continue
                method = "DiffW-like"
            else:
                if mode == "base":
                    method = "O256-Base"
                elif mode == "base_plus_refiner":
                    method = "O256-RGB"
                elif mode == "refiner_only":
                    method = "RGB refiner only"
                else:
                    continue
            mean = float(result["mean_ms_per_image"])
            rows.append(
                {
                    "timing_type": "compute-only",
                    "batch_size": batch,
                    "method": method,
                    "mean_ms_per_image": f"{mean:.6f}",
                    "std_ms_per_image": f"{float(result['std_ms_per_image']):.6f}",
                    "ci95_half_width": f"{compute_ci95(result):.6f}",
                    "cv_percent": f"{float(result['cv_percent']):.4f}",
                    "images_per_second": f"{float(result['images_per_second']):.6f}",
                    "relative_to_diffw_like": "",
                    "speedup_vs_diffw_like": "",
                    "peak_incremental_mib": f"{float(result['peak_incremental_mib']):.4f}",
                    "gpu": gpu,
                    "warmup": warmup,
                    "iterations": iterations,
                    "repeats": repeats,
                    "notes": payload.get("timing_scope", ""),
                }
            )

    diffw_by_batch = {
        row["batch_size"]: float(row["mean_ms_per_image"])
        for row in rows
        if row["method"] == "DiffW-like"
    }
    for row in rows:
        diffw = diffw_by_batch.get(row["batch_size"])
        if diffw:
            mean = float(row["mean_ms_per_image"])
            row["relative_to_diffw_like"] = f"{mean / diffw:.6f}"
            row["speedup_vs_diffw_like"] = f"{diffw / mean:.6f}"
    fields = [
        "timing_type",
        "batch_size",
        "method",
        "mean_ms_per_image",
        "std_ms_per_image",
        "ci95_half_width",
        "cv_percent",
        "images_per_second",
        "relative_to_diffw_like",
        "speedup_vs_diffw_like",
        "peak_incremental_mib",
        "gpu",
        "warmup",
        "iterations",
        "repeats",
        "notes",
    ]
    write_csv(exp_root / "compute_only_inference_timing.csv", rows, fields)
    return rows


def summarize_end_to_end(exp_root: Path) -> list[dict]:
    logs = exp_root / "logs"
    rows: list[dict] = []
    for n in (100, 300, 1000):
        diffw_real = parse_time_log(logs / f"end2end_eval{n}_diffw_like_eval.log")
        base_real = parse_time_log(logs / f"end2end_eval{n}_o256_base_eval.log")
        refine_real = parse_time_log(logs / f"end2end_eval{n}_o256_rgb_refine.log")
        records = []
        if diffw_real is not None:
            records.append(("DiffW-like", diffw_real, "end-to-end DiffW-like eval_diffusion"))
        if base_real is not None:
            records.append(("O256-Base", base_real, "end-to-end O256 base eval_diffusion"))
        if base_real is not None and refine_real is not None:
            records.append(
                (
                    "O256-RGB",
                    base_real + refine_real,
                    "O256 base eval_diffusion + RGB refiner apply",
                )
            )
        diffw = next((seconds for method, seconds, _note in records if method == "DiffW-like"), None)
        for method, seconds, note in records:
            ms = seconds * 1000.0 / n
            row = {
                "timing_type": "end-to-end",
                "num_images": n,
                "method": method,
                "total_seconds": f"{seconds:.6f}",
                "total_hhmmss": fmt_hhmmss(seconds),
                "seconds_per_image": f"{seconds / n:.8f}",
                "ms_per_image": f"{ms:.6f}",
                "images_per_second": f"{n / seconds:.6f}" if seconds else "0",
                "relative_to_diffw_like": "",
                "speedup_vs_diffw_like": "",
                "includes_io": "yes",
                "includes_png_writing": "yes",
                "includes_refiner_apply": "yes" if method == "O256-RGB" else "no",
                "gpu": "NVIDIA RTX 4090",
                "notes": note,
            }
            if diffw:
                row["relative_to_diffw_like"] = f"{seconds / diffw:.6f}"
                row["speedup_vs_diffw_like"] = f"{diffw / seconds:.6f}"
            rows.append(row)
    if rows:
        fields = [
            "timing_type",
            "num_images",
            "method",
            "total_seconds",
            "total_hhmmss",
            "seconds_per_image",
            "ms_per_image",
            "images_per_second",
            "relative_to_diffw_like",
            "speedup_vs_diffw_like",
            "includes_io",
            "includes_png_writing",
            "includes_refiner_apply",
            "gpu",
            "notes",
        ]
        write_csv(exp_root / "end_to_end_inference_timing.csv", rows, fields)
    return rows


def summarize_projection(exp_root: Path, compute_rows: list[dict], e2e_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []

    def add_projection(source: str, source_id: str, method_rows: list[dict], ms_key: str) -> None:
        diffw_ms = next(
            (float(row[ms_key]) for row in method_rows if row["method"] == "DiffW-like"), None
        )
        for row in method_rows:
            method = row["method"]
            if method == "RGB refiner only":
                continue
            ms = float(row[ms_key])
            total_seconds = ms * PROJECTION_N / 1000.0
            out = {
                "projection_source": source,
                "num_images": PROJECTION_N,
                "method": method,
                "source_subset_or_batch": source_id,
                "source_ms_per_image": f"{ms:.6f}",
                "projected_total_seconds": f"{total_seconds:.6f}",
                "projected_total_hhmmss": fmt_hhmmss(total_seconds),
                "relative_to_diffw_like": "",
                "speedup_vs_diffw_like": "",
                "measured_or_projected": "projected",
                "notes": "Projected from measured per-image timing; 111k images were not directly measured.",
            }
            if diffw_ms:
                out["relative_to_diffw_like"] = f"{ms / diffw_ms:.6f}"
                out["speedup_vs_diffw_like"] = f"{diffw_ms / ms:.6f}"
            rows.append(out)

    compute_bs4 = [row for row in compute_rows if row["batch_size"] == 4 and row["method"] != "RGB refiner only"]
    if compute_bs4:
        add_projection("compute-only", "batch_size=4", compute_bs4, "mean_ms_per_image")

    e2e_1000 = [row for row in e2e_rows if row["num_images"] == 1000]
    if e2e_1000:
        add_projection("end-to-end", "N=1000", e2e_1000, "ms_per_image")

    if rows:
        fields = [
            "projection_source",
            "num_images",
            "method",
            "source_subset_or_batch",
            "source_ms_per_image",
            "projected_total_seconds",
            "projected_total_hhmmss",
            "relative_to_diffw_like",
            "speedup_vs_diffw_like",
            "measured_or_projected",
            "notes",
        ]
        write_csv(exp_root / "inference_cost_projection_111k.csv", rows, fields)
    return rows


def plot_compute(exp_root: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fig_dir = exp_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["batch_size"] = df["batch_size"].astype(int)
    df["mean_ms_per_image"] = df["mean_ms_per_image"].astype(float)
    plt.figure(figsize=(8.0, 4.8))
    colors = {"DiffW-like": "#3b4cc0", "O256-Base": "#2f9e44", "O256-RGB": "#d9480f"}
    for method in ("DiffW-like", "O256-Base", "O256-RGB"):
        sub = df[df["method"] == method].sort_values("batch_size")
        if sub.empty:
            continue
        plt.plot(
            sub["batch_size"],
            sub["mean_ms_per_image"],
            marker="o",
            linewidth=2.3,
            label=method,
            color=colors.get(method),
        )
    plt.xlabel("Batch size")
    plt.ylabel("平均推論時間 (ms/image)")
    plt.title("Compute-only 推論時間比較")
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.28)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_compute_only_ms_per_image_vs_batch.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_end_to_end(exp_root: Path, rows: list[dict], projection_rows: list[dict]) -> None:
    if not rows:
        return
    fig_dir = exp_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["num_images"] = df["num_images"].astype(int)
    df["total_seconds"] = df["total_seconds"].astype(float)
    plt.figure(figsize=(8.0, 4.8))
    colors = {"DiffW-like": "#3b4cc0", "O256-Base": "#2f9e44", "O256-RGB": "#d9480f"}
    for method in ("DiffW-like", "O256-Base", "O256-RGB"):
        sub = df[df["method"] == method].sort_values("num_images")
        if sub.empty:
            continue
        plt.plot(
            sub["num_images"],
            sub["total_seconds"] / 60.0,
            marker="o",
            linewidth=2.3,
            label=f"{method} measured",
            color=colors.get(method),
        )
    proj = pd.DataFrame([row for row in projection_rows if row["projection_source"] == "end-to-end"])
    if not proj.empty:
        proj["projected_total_seconds"] = proj["projected_total_seconds"].astype(float)
        for method in ("DiffW-like", "O256-RGB"):
            measured = df[df["method"] == method].sort_values("num_images")
            projected = proj[proj["method"] == method]
            if measured.empty or projected.empty:
                continue
            x = [int(measured.iloc[-1]["num_images"]), PROJECTION_N]
            y = [
                float(measured.iloc[-1]["total_seconds"]) / 60.0,
                float(projected.iloc[0]["projected_total_seconds"]) / 60.0,
            ]
            plt.plot(x, y, linestyle="--", linewidth=1.8, color=colors.get(method), alpha=0.75)
            plt.scatter([PROJECTION_N], [y[-1]], color=colors.get(method), marker="x", s=60)
    plt.xlabel("影像數量")
    plt.ylabel("總推論時間 (minutes)")
    plt.title("End-to-end 推論時間與 111k 推估")
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.28)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_end_to_end_total_time_vs_num_images.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_projection(exp_root: Path, projection_rows: list[dict]) -> None:
    if not projection_rows:
        return
    fig_dir = exp_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(projection_rows)
    df["projected_total_seconds"] = df["projected_total_seconds"].astype(float)
    df = df[df["method"].isin(["DiffW-like", "O256-RGB"])]
    if df.empty:
        return
    labels = []
    values = []
    colors = []
    for _, row in df.iterrows():
        labels.append(f"{row['projection_source']}\n{row['method']}")
        values.append(row["projected_total_seconds"] / 3600.0)
        colors.append("#3b4cc0" if row["method"] == "DiffW-like" else "#d9480f")
    plt.figure(figsize=(8.2, 4.8))
    plt.bar(labels, values, color=colors)
    plt.ylabel("111k 推估總時間 (hours)")
    plt.title("111k 影像推論成本推估")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_projection_111k_total_time.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_summary(exp_root: Path, compute_rows: list[dict], e2e_rows: list[dict], projection_rows: list[dict]) -> None:
    lines = [
        "# Formal Inference Cost Comparison Summary",
        "",
        "Classification: formal efficiency support / supplementary deployment estimate.",
        "",
        "## Definitions",
        "",
        "- Compute-only inference excludes dataset I/O, PNG writing, PSNR, SSIM, BA, and attacks.",
        "- End-to-end inference includes reading the fixed subset, model inference, and writing output images/message files.",
        "- 111k results are projections from measured per-image time, not directly measured 111k runs.",
        "",
    ]
    if compute_rows:
        lines.extend(
            [
                "## Compute-only Results",
                "",
                "| Batch | Method | Mean ms/image | Std | 95% CI | Image/s | Speedup vs DiffW-like |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(compute_rows, key=lambda r: (int(r["batch_size"]), r["method"])):
            if row["method"] == "RGB refiner only":
                continue
            lines.append(
                f"| {row['batch_size']} | {row['method']} | {float(row['mean_ms_per_image']):.4f} | "
                f"{float(row['std_ms_per_image']):.4f} | {float(row['ci95_half_width']):.4f} | "
                f"{float(row['images_per_second']):.2f} | {float(row['speedup_vs_diffw_like'] or 0):.2f}x |"
            )
        lines.append("")
    if e2e_rows:
        lines.extend(
            [
                "## End-to-end Results",
                "",
                "| N | Method | Total time | ms/image | Image/s | Speedup vs DiffW-like | Notes |",
                "|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(e2e_rows, key=lambda r: (int(r["num_images"]), r["method"])):
            lines.append(
                f"| {row['num_images']} | {row['method']} | {row['total_hhmmss']} | "
                f"{float(row['ms_per_image']):.4f} | {float(row['images_per_second']):.2f} | "
                f"{float(row['speedup_vs_diffw_like'] or 0):.2f}x | {row['notes']} |"
            )
        lines.append("")
    if projection_rows:
        lines.extend(
            [
                "## 111k Projection",
                "",
                "| Source | Method | Source ms/image | Projected total time | Speedup vs DiffW-like |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in projection_rows:
            if row["method"] == "O256-Base":
                continue
            lines.append(
                f"| {row['projection_source']} ({row['source_subset_or_batch']}) | {row['method']} | "
                f"{float(row['source_ms_per_image']):.4f} | {row['projected_total_hhmmss']} | "
                f"{float(row['speedup_vs_diffw_like'] or 0):.2f}x |"
            )
        lines.append("")
    lines.extend(
        [
            "## Writing Guardrail",
            "",
            "- Do not write that RGB refinement accelerates inference. It adds a small overhead relative to O256-Base.",
            "- The overall O256-RGB pipeline can be faster than DiffW-like under this protocol.",
            "- Do not describe 111k projection as a directly measured 111k experiment.",
        ]
    )
    (exp_root / "inference_cost_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_font()
    exp_root = Path(args.exp_root)
    compute_rows = summarize_compute(exp_root)
    e2e_rows = summarize_end_to_end(exp_root)
    projection_rows = summarize_projection(exp_root, compute_rows, e2e_rows)
    plot_compute(exp_root, compute_rows)
    plot_end_to_end(exp_root, e2e_rows, projection_rows)
    plot_projection(exp_root, projection_rows)
    write_summary(exp_root, compute_rows, e2e_rows, projection_rows)
    print(f"summary: {exp_root / 'inference_cost_summary.md'}")


if __name__ == "__main__":
    main()
