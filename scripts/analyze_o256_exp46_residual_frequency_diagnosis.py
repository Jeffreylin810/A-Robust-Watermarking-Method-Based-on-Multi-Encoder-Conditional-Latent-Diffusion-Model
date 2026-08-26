#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether RGB refinement erases residual/high-frequency watermark evidence."
    )
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--exp41_rgb_dir", required=True)
    parser.add_argument("--exp44_rgb_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--delta_csv", required=True, help="exp45 delta/per_image_delta.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--fft_cutoff", type=float, default=0.35)
    return parser.parse_args()


def image_map(directory: str | Path) -> dict[str, Path]:
    root = Path(directory)
    return {
        path.stem: path
        for path in sorted(root.iterdir())
        if path.suffix.lower() in IMAGE_EXTS and path.exists()
    }


def load_image(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def read_exp45_delta(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stem = row["image"]
            parsed = dict(row)
            for key in (
                "left_ba",
                "right_ba",
                "delta_ba",
                "left_psnr",
                "right_psnr",
                "delta_psnr",
                "baseline_ba",
                "left_vs_baseline_delta_ba",
                "right_vs_baseline_delta_ba",
            ):
                if key in parsed and parsed[key] != "":
                    parsed[key] = float(parsed[key])
            for key in ("left_error_bits", "right_error_bits", "delta_error_bits", "baseline_error_bits"):
                if key in parsed and parsed[key] != "":
                    parsed[key] = int(parsed[key])
            rows[stem] = parsed
    return rows


def flatten(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(-1).astype(np.float64)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = flatten(a)
    bf = flatten(b)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom < EPS:
        return float("nan")
    return float(np.dot(af, bf) / denom)


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) < EPS or float(np.std(y)) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    return arr.mean(axis=2)


def gradient_energy(arr: np.ndarray) -> float:
    g = gray(arr)
    dx = np.diff(g, axis=1)
    dy = np.diff(g, axis=0)
    return float((np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / 2.0)


def laplacian_abs(arr: np.ndarray) -> float:
    g = gray(arr)
    lap = (
        -4.0 * g
        + np.roll(g, 1, axis=0)
        + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1)
        + np.roll(g, -1, axis=1)
    )
    return float(np.mean(np.abs(lap)))


def fft_high_metrics(arr: np.ndarray, cutoff: float) -> tuple[float, float, float]:
    g = gray(arr).astype(np.float64)
    g = g - float(np.mean(g))
    spectrum = np.fft.fftshift(np.fft.fft2(g))
    power = np.abs(spectrum) ** 2
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = radius / (float(radius.max()) + EPS)
    high = radius >= cutoff
    total_power = float(power.sum())
    high_power = float(power[high].sum())
    return high_power / (total_power + EPS), high_power / float(power.size), total_power / float(power.size)


def residual_metrics(residual: np.ndarray, fft_cutoff: float) -> dict[str, float]:
    abs_res = np.abs(residual)
    high_ratio, high_power, total_power = fft_high_metrics(residual, fft_cutoff)
    return {
        "l1": float(np.mean(abs_res)),
        "l2": float(np.sqrt(np.mean(residual ** 2))),
        "std": float(np.std(residual)),
        "p95_abs": float(np.percentile(abs_res, 95)),
        "max_abs": float(np.max(abs_res)),
        "grad_energy": gradient_energy(residual),
        "lap_abs": laplacian_abs(residual),
        "fft_high_ratio": high_ratio,
        "fft_high_power": high_power,
        "fft_total_power": total_power,
    }


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def safe_ratio(a: float, b: float) -> float:
    return float(a / (b + EPS))


def add_prefixed(row: dict[str, object], prefix: str, values: dict[str, float]) -> None:
    for key, value in values.items():
        row[f"{prefix}_{key}"] = value


def analyze_one(stem: str, paths: dict[str, dict[str, Path]], delta_info: dict[str, object], fft_cutoff: float) -> dict[str, object]:
    gt_img = Image.open(paths["gt"][stem]).convert("RGB")
    size = gt_img.size
    gt = np.asarray(gt_img, dtype=np.float32) / 255.0
    base = load_image(paths["base"][stem], size=size)
    exp41 = load_image(paths["exp41"][stem], size=size)
    exp44 = load_image(paths["exp44"][stem], size=size)

    base_res = base - gt
    exp41_res = exp41 - gt
    exp44_res = exp44 - gt
    exp41_delta = exp41 - base
    exp44_delta = exp44 - base
    replay_delta = exp44 - exp41

    base_m = residual_metrics(base_res, fft_cutoff)
    exp41_m = residual_metrics(exp41_res, fft_cutoff)
    exp44_m = residual_metrics(exp44_res, fft_cutoff)
    exp41_d_m = residual_metrics(exp41_delta, fft_cutoff)
    exp44_d_m = residual_metrics(exp44_delta, fft_cutoff)
    replay_d_m = residual_metrics(replay_delta, fft_cutoff)

    row: dict[str, object] = {
        "image": stem,
        "transition": delta_info.get("transition", ""),
        "delta_ba": delta_info.get("delta_ba", ""),
        "left_ba_exp41_rgb": delta_info.get("left_ba", ""),
        "right_ba_exp44_rgb": delta_info.get("right_ba", ""),
        "left_error_bits": delta_info.get("left_error_bits", ""),
        "right_error_bits": delta_info.get("right_error_bits", ""),
        "delta_error_bits": delta_info.get("delta_error_bits", ""),
        "delta_psnr_exp45": delta_info.get("delta_psnr", ""),
        "exp41_psnr": psnr(exp41, gt),
        "exp44_psnr": psnr(exp44, gt),
        "delta_psnr": psnr(exp44, gt) - psnr(exp41, gt),
    }
    add_prefixed(row, "base_res", base_m)
    add_prefixed(row, "exp41_res", exp41_m)
    add_prefixed(row, "exp44_res", exp44_m)
    add_prefixed(row, "exp41_delta", exp41_d_m)
    add_prefixed(row, "exp44_delta", exp44_d_m)
    add_prefixed(row, "replay_delta", replay_d_m)

    row.update(
        {
            "exp41_res_cos_base_res": cosine(exp41_res, base_res),
            "exp44_res_cos_base_res": cosine(exp44_res, base_res),
            "delta_res_cos_base_res": cosine(exp44_res, base_res) - cosine(exp41_res, base_res),
            "exp41_delta_cancel_cos": cosine(exp41_delta, -base_res),
            "exp44_delta_cancel_cos": cosine(exp44_delta, -base_res),
            "delta_cancel_cos": cosine(exp44_delta, -base_res) - cosine(exp41_delta, -base_res),
            "exp41_res_l2_ratio_to_base": safe_ratio(exp41_m["l2"], base_m["l2"]),
            "exp44_res_l2_ratio_to_base": safe_ratio(exp44_m["l2"], base_m["l2"]),
            "delta_res_l2_ratio_to_base": safe_ratio(exp44_m["l2"], base_m["l2"]) - safe_ratio(exp41_m["l2"], base_m["l2"]),
            "exp41_lap_ratio_to_base": safe_ratio(exp41_m["lap_abs"], base_m["lap_abs"]),
            "exp44_lap_ratio_to_base": safe_ratio(exp44_m["lap_abs"], base_m["lap_abs"]),
            "delta_lap_ratio_to_base": safe_ratio(exp44_m["lap_abs"], base_m["lap_abs"]) - safe_ratio(exp41_m["lap_abs"], base_m["lap_abs"]),
            "exp41_fft_high_power_ratio_to_base": safe_ratio(exp41_m["fft_high_power"], base_m["fft_high_power"]),
            "exp44_fft_high_power_ratio_to_base": safe_ratio(exp44_m["fft_high_power"], base_m["fft_high_power"]),
            "delta_fft_high_power_ratio_to_base": safe_ratio(exp44_m["fft_high_power"], base_m["fft_high_power"]) - safe_ratio(exp41_m["fft_high_power"], base_m["fft_high_power"]),
            "delta_refiner_l1": exp44_d_m["l1"] - exp41_d_m["l1"],
            "delta_refiner_l2": exp44_d_m["l2"] - exp41_d_m["l2"],
        }
    )
    return row


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric(row: dict[str, object], key: str) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return float("nan")
    return float(value)


def summarize_group(name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    keys = [
        "delta_ba",
        "delta_psnr",
        "delta_res_cos_base_res",
        "delta_cancel_cos",
        "delta_res_l2_ratio_to_base",
        "delta_lap_ratio_to_base",
        "delta_fft_high_power_ratio_to_base",
        "delta_refiner_l1",
        "delta_refiner_l2",
        "replay_delta_l1",
        "replay_delta_l2",
        "exp41_res_cos_base_res",
        "exp44_res_cos_base_res",
        "exp41_delta_cancel_cos",
        "exp44_delta_cancel_cos",
    ]
    out: dict[str, object] = {"group": name, "count": len(rows)}
    for key in keys:
        values = [numeric(row, key) for row in rows]
        values = [value for value in values if np.isfinite(value)]
        out[f"mean_{key}"] = float(np.mean(values)) if values else float("nan")
    return out


def build_correlations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    target = "delta_ba"
    features = [
        "delta_psnr",
        "exp44_psnr",
        "delta_res_cos_base_res",
        "delta_cancel_cos",
        "delta_res_l2_ratio_to_base",
        "delta_lap_ratio_to_base",
        "delta_fft_high_power_ratio_to_base",
        "delta_refiner_l1",
        "delta_refiner_l2",
        "replay_delta_l1",
        "replay_delta_l2",
        "exp41_res_cos_base_res",
        "exp44_res_cos_base_res",
        "exp41_delta_cancel_cos",
        "exp44_delta_cancel_cos",
        "exp44_lap_ratio_to_base",
        "exp44_fft_high_power_ratio_to_base",
    ]
    out = []
    y = [numeric(row, target) for row in rows]
    for feature in features:
        x = [numeric(row, feature) for row in rows]
        out.append(
            {
                "target": target,
                "feature": feature,
                "pearson": corr(x, y),
            }
        )
    return sorted(out, key=lambda item: abs(float(item["pearson"])) if np.isfinite(float(item["pearson"])) else -1, reverse=True)


def make_plots(rows: list[dict[str, object]], group_rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plot_warning.txt").write_text(f"Skipped plots: {exc}\n", encoding="utf-8")
        return []

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "newly_below_threshold": "#c2410c",
        "persistent_below_threshold": "#b91c1c",
        "both_ge_threshold": "#2563eb",
        "recovered_to_ge_threshold": "#059669",
    }
    paths: list[Path] = []

    def scatter(x_key: str, y_key: str, xlabel: str, ylabel: str, title: str, filename: str):
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        for transition in sorted({str(row["transition"]) for row in rows}):
            subset = [row for row in rows if row["transition"] == transition]
            ax.scatter(
                [numeric(row, x_key) for row in subset],
                [numeric(row, y_key) for row in subset],
                s=14,
                alpha=0.55,
                label=transition,
                color=colors.get(transition, "#6b7280"),
            )
        ax.axhline(0, color="black", linewidth=1.0)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        path = fig_dir / filename
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    scatter(
        "delta_lap_ratio_to_base",
        "delta_ba",
        "Delta Laplacian residual ratio (exp44 - exp41)",
        "Delta Dropout BA (exp44 - exp41)",
        "High-frequency residual change vs BA change",
        "delta_lap_ratio_vs_delta_ba.png",
    )
    scatter(
        "delta_cancel_cos",
        "delta_ba",
        "Delta cancellation cosine (exp44 - exp41)",
        "Delta Dropout BA (exp44 - exp41)",
        "Residual cancellation change vs BA change",
        "delta_cancel_cos_vs_delta_ba.png",
    )
    scatter(
        "replay_delta_l1",
        "delta_ba",
        "Mean absolute exp44-exp41 image delta",
        "Delta Dropout BA (exp44 - exp41)",
        "Refiner output change vs BA change",
        "replay_delta_l1_vs_delta_ba.png",
    )
    scatter(
        "delta_psnr",
        "delta_ba",
        "Delta PSNR (exp44 - exp41)",
        "Delta Dropout BA (exp44 - exp41)",
        "PSNR gain vs BA change",
        "delta_psnr_vs_delta_ba.png",
    )

    groups = [row for row in group_rows if row["group"] in ("both_ge_threshold", "persistent_below_threshold", "newly_below_threshold")]
    labels = [str(row["group"]).replace("_", "\n") for row in groups]
    x = np.arange(len(groups))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - width / 2, [numeric(row, "mean_delta_ba") for row in groups], width, label="Delta BA")
    ax.bar(x + width / 2, [numeric(row, "mean_delta_psnr") for row in groups], width, label="Delta PSNR")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Group-level BA and PSNR changes")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    path = fig_dir / "group_delta_ba_psnr_bar.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    maps = {
        "base": image_map(args.base_dir),
        "exp41": image_map(args.exp41_rgb_dir),
        "exp44": image_map(args.exp44_rgb_dir),
        "gt": image_map(args.gt_dir),
    }
    delta_info = read_exp45_delta(Path(args.delta_csv))
    stems = sorted(set(delta_info) & set(maps["base"]) & set(maps["exp41"]) & set(maps["exp44"]) & set(maps["gt"]))
    if args.max_images:
        stems = stems[: args.max_images]
    if not stems:
        raise RuntimeError("No matched images across base/exp41/exp44/GT/exp45 delta CSV.")

    rows = []
    for index, stem in enumerate(stems, start=1):
        rows.append(analyze_one(stem, maps, delta_info[stem], args.fft_cutoff))
        if index % 100 == 0:
            print(f"processed {index}/{len(stems)}", flush=True)

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["transition"])].append(row)
    group_rows = [summarize_group(group, group_rows) for group, group_rows in sorted(groups.items())]
    group_rows.append(summarize_group("all", rows))
    group_rows.append(summarize_group("improved_1bitplus", [row for row in rows if numeric(row, "delta_error_bits") < 0]))
    group_rows.append(summarize_group("degraded_1bitplus", [row for row in rows if numeric(row, "delta_error_bits") > 0]))

    correlation_rows = build_correlations(rows)
    write_csv(output_dir / "per_image_residual_frequency.csv", rows)
    write_csv(output_dir / "group_summary.csv", group_rows)
    write_csv(output_dir / "correlations.csv", correlation_rows)
    plot_paths = make_plots(rows, group_rows, output_dir)

    summary_payload = {
        "count": len(rows),
        "fft_cutoff": args.fft_cutoff,
        "groups": group_rows,
        "top_correlations": correlation_rows[:8],
        "plots": [str(path) for path in plot_paths],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    group_by_name = {str(row["group"]): row for row in group_rows}
    corr_lines = "\n".join(
        f"| {row['feature']} | {float(row['pearson']):.4f} |"
        for row in correlation_rows[:10]
    )
    group_lines = "\n".join(
        f"| {row['group']} | {row['count']} | {float(row['mean_delta_ba']):.4f} | "
        f"{float(row['mean_delta_psnr']):.4f} | {float(row['mean_delta_lap_ratio_to_base']):.6f} | "
        f"{float(row['mean_delta_cancel_cos']):.6f} | {float(row['mean_replay_delta_l1']):.6f} |"
        for row in group_rows
    )
    all_row = group_by_name["all"]
    md = [
        "# O256 exp46 residual/frequency diagnosis",
        "",
        "## Main Question",
        "",
        "Does exp44 improve image quality by smoothing or cancelling residual evidence that exp41 preserved under Dropout?",
        "",
        "## Overall",
        "",
        "| Count | Mean delta BA | Mean delta PSNR | Mean delta Lap ratio | Mean delta cancellation cosine | Mean exp44-exp41 delta L1 |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {all_row['count']} | {float(all_row['mean_delta_ba']):.4f} | {float(all_row['mean_delta_psnr']):.4f} | "
        f"{float(all_row['mean_delta_lap_ratio_to_base']):.6f} | {float(all_row['mean_delta_cancel_cos']):.6f} | "
        f"{float(all_row['mean_replay_delta_l1']):.6f} |",
        "",
        "## Group Summary",
        "",
        "| Group | Count | Mean delta BA | Mean delta PSNR | Mean delta Lap ratio | Mean delta cancellation cosine | Mean exp44-exp41 delta L1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        group_lines,
        "",
        "## Top Correlations With Delta BA",
        "",
        "| Feature | Pearson r |",
        "|---|---:|",
        corr_lines,
        "",
        "## Reading Guide",
        "",
        "- `delta_ba` is exp44 Dropout BA minus exp41 Dropout BA; negative means exp44 is worse.",
        "- `delta_lap_ratio_to_base` measures whether exp44 keeps more or less high-frequency residual than exp41, normalized by base residual.",
        "- `delta_cancel_cos` measures whether exp44's refiner update is more aligned with cancelling base residual than exp41.",
        "- `replay_delta_l1` is the direct exp44-vs-exp41 image change magnitude.",
        "- If newly failed images show BA loss while PSNR rises and residual/frequency preservation drops, the refiner is likely smoothing away watermark evidence.",
        "",
        "## Outputs",
        "",
        "- `per_image_residual_frequency.csv`",
        "- `group_summary.csv`",
        "- `correlations.csv`",
        "- `figures/delta_lap_ratio_vs_delta_ba.png`",
        "- `figures/delta_cancel_cos_vs_delta_ba.png`",
        "- `figures/replay_delta_l1_vs_delta_ba.png`",
        "- `figures/delta_psnr_vs_delta_ba.png`",
        "- `figures/group_delta_ba_psnr_bar.png`",
    ]
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Summary md: {output_dir / 'summary.md'}")
    print(f"Per-image csv: {output_dir / 'per_image_residual_frequency.csv'}")


if __name__ == "__main__":
    main()
