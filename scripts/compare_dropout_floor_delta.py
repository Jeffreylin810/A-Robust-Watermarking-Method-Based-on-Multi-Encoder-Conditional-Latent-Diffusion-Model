#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
BIT_BA_STEP = 100.0 / 32.0


def read_per_image_csv(path):
    by_variant = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variant = row["variant"]
            by_variant.setdefault(variant, {})[row["image"]] = {
                "ba": float(row["ba"]),
                "error_bits": int(row["error_bits"]),
            }
    return by_variant


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def image_map(directory):
    if not directory:
        return {}
    root = Path(directory)
    return {
        path.stem: path
        for path in sorted(root.iterdir())
        if path.suffix.lower() in IMAGE_EXTS and path.exists()
    }


def psnr(a_path, b_path):
    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        b_img = Image.open(b_path).convert("RGB").resize((a.shape[1], a.shape[0]))
        b = np.asarray(b_img, dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def percentile(values, q):
    values = list(values)
    return float(np.percentile(values, q)) if values else float("nan")


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def classify(left_ba, right_ba, threshold):
    left_hard = left_ba < threshold
    right_hard = right_ba < threshold
    if left_hard and not right_hard:
        return "recovered_to_ge_threshold"
    if not left_hard and right_hard:
        return "newly_below_threshold"
    if left_hard and right_hard:
        return "persistent_below_threshold"
    return "both_ge_threshold"


def build_rows(args):
    data = read_per_image_csv(args.per_image_csv)
    for variant in (args.left, args.right):
        if variant not in data:
            raise ValueError(f"Variant not found in per-image CSV: {variant}")
    if args.baseline and args.baseline not in data:
        raise ValueError(f"Baseline variant not found in per-image CSV: {args.baseline}")

    left = data[args.left]
    right = data[args.right]
    baseline = data.get(args.baseline, {})
    common = sorted(set(left) & set(right))
    if not common:
        raise RuntimeError(f"No common images for {args.left} vs {args.right}")

    left_images = image_map(args.left_image_dir)
    right_images = image_map(args.right_image_dir)
    gt_images = image_map(args.gt_dir)
    have_psnr = bool(left_images and right_images and gt_images)

    rows = []
    for image in common:
        left_ba = left[image]["ba"]
        right_ba = right[image]["ba"]
        left_err = left[image]["error_bits"]
        right_err = right[image]["error_bits"]
        row = {
            "image": image,
            "left_variant": args.left,
            "right_variant": args.right,
            "left_ba": left_ba,
            "right_ba": right_ba,
            "delta_ba": right_ba - left_ba,
            "left_error_bits": left_err,
            "right_error_bits": right_err,
            "delta_error_bits": right_err - left_err,
            "left_hard": left_ba < args.threshold_ba,
            "right_hard": right_ba < args.threshold_ba,
            "transition": classify(left_ba, right_ba, args.threshold_ba),
            "changed_at_least_one_bit": abs(right_err - left_err) >= 1,
            "improved_1bitplus": right_err <= left_err - 1,
            "degraded_1bitplus": right_err >= left_err + 1,
        }
        if args.baseline and image in baseline:
            base_ba = baseline[image]["ba"]
            base_err = baseline[image]["error_bits"]
            row.update(
                {
                    "baseline_variant": args.baseline,
                    "baseline_ba": base_ba,
                    "baseline_error_bits": base_err,
                    "left_vs_baseline_delta_ba": left_ba - base_ba,
                    "right_vs_baseline_delta_ba": right_ba - base_ba,
                }
            )
        if have_psnr and image in left_images and image in right_images and image in gt_images:
            left_psnr = psnr(left_images[image], gt_images[image])
            right_psnr = psnr(right_images[image], gt_images[image])
            row.update(
                {
                    "left_psnr": left_psnr,
                    "right_psnr": right_psnr,
                    "delta_psnr": right_psnr - left_psnr,
                }
            )
        rows.append(row)
    return rows


def summarize(rows, threshold):
    transitions = Counter(row["transition"] for row in rows)
    delta_ba = [float(row["delta_ba"]) for row in rows]
    delta_bits = [int(row["delta_error_bits"]) for row in rows]
    left_ba = [float(row["left_ba"]) for row in rows]
    right_ba = [float(row["right_ba"]) for row in rows]
    left_hard = [row for row in rows if row["left_hard"]]
    right_hard = [row for row in rows if row["right_hard"]]
    recovered = [row for row in rows if row["transition"] == "recovered_to_ge_threshold"]
    newly_failed = [row for row in rows if row["transition"] == "newly_below_threshold"]
    persistent = [row for row in rows if row["transition"] == "persistent_below_threshold"]

    summary = {
        "common_images": len(rows),
        "threshold_ba": threshold,
        "left_avg_ba": mean(left_ba),
        "right_avg_ba": mean(right_ba),
        "delta_avg_ba": mean(right_ba) - mean(left_ba),
        "left_min_ba": float(np.min(left_ba)),
        "right_min_ba": float(np.min(right_ba)),
        "left_hard_lt_threshold": len(left_hard),
        "right_hard_lt_threshold": len(right_hard),
        "recovered_to_ge_threshold": len(recovered),
        "newly_below_threshold": len(newly_failed),
        "persistent_below_threshold": len(persistent),
        "both_ge_threshold": transitions["both_ge_threshold"],
        "improved_1bitplus": sum(1 for row in rows if row["improved_1bitplus"]),
        "degraded_1bitplus": sum(1 for row in rows if row["degraded_1bitplus"]),
        "unchanged_error_bits": sum(1 for bits in delta_bits if bits == 0),
        "delta_ba_mean": mean(delta_ba),
        "delta_ba_median": float(np.median(delta_ba)),
        "delta_ba_p05": percentile(delta_ba, 5),
        "delta_ba_p95": percentile(delta_ba, 95),
        "delta_error_bits_mean": mean(delta_bits),
        "net_new_hard_images": len(right_hard) - len(left_hard),
        "hard_set_jaccard": (
            len({r["image"] for r in left_hard} & {r["image"] for r in right_hard})
            / len({r["image"] for r in left_hard} | {r["image"] for r in right_hard})
            if ({r["image"] for r in left_hard} | {r["image"] for r in right_hard})
            else 1.0
        ),
    }
    if rows and "left_psnr" in rows[0]:
        left_psnr = [float(row["left_psnr"]) for row in rows if "left_psnr" in row]
        right_psnr = [float(row["right_psnr"]) for row in rows if "right_psnr" in row]
        summary.update(
            {
                "left_avg_psnr": mean(left_psnr),
                "right_avg_psnr": mean(right_psnr),
                "delta_avg_psnr": mean(right_psnr) - mean(left_psnr),
            }
        )
    return summary


def make_plots(rows, output_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plot_warning.txt").write_text(
            f"Matplotlib unavailable; skipped plots: {exc}\n", encoding="utf-8"
        )
        return []

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    left_ba = np.array([row["left_ba"] for row in rows], dtype=float)
    right_ba = np.array([row["right_ba"] for row in rows], dtype=float)
    delta_ba = right_ba - left_ba

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.scatter(left_ba, right_ba, s=14, alpha=0.55)
    lo = min(left_ba.min(), right_ba.min()) - 1
    hi = max(left_ba.max(), right_ba.max()) + 1
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
    ax.axhline(90, color="#b23b3b", linewidth=1.0, linestyle=":")
    ax.axvline(90, color="#b23b3b", linewidth=1.0, linestyle=":")
    ax.set_xlabel("exp41 Dropout r03 per-image BA")
    ax.set_ylabel("exp44 Dropout r03 per-image BA")
    ax.set_title("Dropout floor per-image BA: exp41 vs exp44")
    ax.grid(alpha=0.25)
    path = fig_dir / "exp41_vs_exp44_dropout_ba_scatter.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.hist(delta_ba, bins=np.arange(delta_ba.min() - BIT_BA_STEP, delta_ba.max() + 2 * BIT_BA_STEP, BIT_BA_STEP), color="#4e79a7", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Delta BA (exp44 - exp41)")
    ax.set_ylabel("Image count")
    ax.set_title("Dropout r03 per-image BA delta")
    ax.grid(axis="y", alpha=0.25)
    path = fig_dir / "exp44_minus_exp41_delta_ba_hist.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    if rows and "delta_psnr" in rows[0]:
        delta_psnr = np.array([row["delta_psnr"] for row in rows if "delta_psnr" in row], dtype=float)
        delta_ba_psnr = np.array([row["delta_ba"] for row in rows if "delta_psnr" in row], dtype=float)
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        ax.scatter(delta_psnr, delta_ba_psnr, s=14, alpha=0.55)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_xlabel("Delta PSNR (exp44 - exp41)")
        ax.set_ylabel("Delta BA (exp44 - exp41)")
        ax.set_title("Quality gain vs Dropout BA change")
        ax.grid(alpha=0.25)
        path = fig_dir / "delta_psnr_vs_delta_ba.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def format_bool(value):
    return "yes" if value else "no"


def write_outputs(rows, summary, args, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    dynamic_fields = []
    for key in (
        "baseline_variant",
        "baseline_ba",
        "baseline_error_bits",
        "left_vs_baseline_delta_ba",
        "right_vs_baseline_delta_ba",
        "left_psnr",
        "right_psnr",
        "delta_psnr",
    ):
        if any(key in row for row in rows):
            dynamic_fields.append(key)
    fields = [
        "image",
        "left_variant",
        "right_variant",
        "left_ba",
        "right_ba",
        "delta_ba",
        "left_error_bits",
        "right_error_bits",
        "delta_error_bits",
        "left_hard",
        "right_hard",
        "transition",
        "changed_at_least_one_bit",
        "improved_1bitplus",
        "degraded_1bitplus",
    ] + dynamic_fields
    write_csv(output_dir / "per_image_delta.csv", rows, fields)

    summary_rows = [{k: v for k, v in summary.items()}]
    write_csv(output_dir / "transition_summary.csv", summary_rows, list(summary_rows[0].keys()))

    recovered = sorted(
        [row for row in rows if row["transition"] == "recovered_to_ge_threshold"],
        key=lambda row: row["delta_ba"],
        reverse=True,
    )
    newly_failed = sorted(
        [row for row in rows if row["transition"] == "newly_below_threshold"],
        key=lambda row: row["delta_ba"],
    )
    persistent = sorted(
        [row for row in rows if row["transition"] == "persistent_below_threshold"],
        key=lambda row: row["right_ba"],
    )
    improved = sorted(rows, key=lambda row: row["delta_ba"], reverse=True)
    degraded = sorted(rows, key=lambda row: row["delta_ba"])

    for filename, subset in (
        ("top_recovered.csv", recovered[:100]),
        ("top_newly_failed.csv", newly_failed[:100]),
        ("persistent_hard.csv", persistent[:200]),
        ("top_improved.csv", improved[:100]),
        ("top_degraded.csv", degraded[:100]),
    ):
        write_csv(output_dir / filename, subset, fields)

    plot_paths = make_plots(rows, output_dir)

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "left": args.left,
                "right": args.right,
                "baseline": args.baseline,
                "summary": summary,
                "plots": [str(path) for path in plot_paths],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# O256 exp45 Dropout Floor Delta Diagnosis",
        "",
        "## Main Question",
        "",
        "Does exp44 hardcase replay recover the exp41 seed62 Dropout floor, or does it trade robustness for image quality?",
        "",
        "## Delta Summary",
        "",
        "| Left | Right | Common | Avg BA L/R | Δ Avg BA | Min BA L/R | Hard<90 L/R | Recovered | Newly <90 | Persistent <90 | Improved >=1 bit | Degraded >=1 bit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {args.left} | {args.right} | {summary['common_images']} | "
            f"{summary['left_avg_ba']:.2f}/{summary['right_avg_ba']:.2f} | "
            f"{summary['delta_avg_ba']:.2f} | "
            f"{summary['left_min_ba']:.2f}/{summary['right_min_ba']:.2f} | "
            f"{summary['left_hard_lt_threshold']}/{summary['right_hard_lt_threshold']} | "
            f"{summary['recovered_to_ge_threshold']} | "
            f"{summary['newly_below_threshold']} | "
            f"{summary['persistent_below_threshold']} | "
            f"{summary['improved_1bitplus']} | {summary['degraded_1bitplus']} |"
        ),
        "",
    ]
    if "left_avg_psnr" in summary:
        lines.extend(
            [
                "## Quality Delta",
                "",
                "| Left PSNR | Right PSNR | Δ PSNR |",
                "|---:|---:|---:|",
                f"| {summary['left_avg_psnr']:.4f} | {summary['right_avg_psnr']:.4f} | {summary['delta_avg_psnr']:.4f} |",
                "",
            ]
        )

    def top_table(title, subset):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Image | BA L | BA R | Δ BA | Err L | Err R | Transition |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in subset[:10]:
            lines.append(
                f"| {row['image']} | {row['left_ba']:.2f} | {row['right_ba']:.2f} | "
                f"{row['delta_ba']:.2f} | {row['left_error_bits']} | {row['right_error_bits']} | {row['transition']} |"
            )
        if not subset:
            lines.append("| N/A |  |  |  |  |  |  |")
        lines.append("")

    top_table("Top Recovered Images", recovered)
    top_table("Top Newly Failed Images", newly_failed)
    top_table("Top Degraded Images", degraded)

    lines.extend(
        [
            "## Interpretation Guide",
            "",
            "- If `newly_below_threshold` is larger than `recovered_to_ge_threshold`, hardcase replay is hurting the finaltest Dropout floor more than it helps.",
            "- If PSNR improves while BA delta is negative, the replay is acting like quality recovery rather than robustness recovery.",
            "- If many images are persistent below 90, the bottleneck is not a small set of easy-to-replay cases.",
            "",
            "## Outputs",
            "",
            "- `per_image_delta.csv`",
            "- `transition_summary.csv`",
            "- `top_recovered.csv`",
            "- `top_newly_failed.csv`",
            "- `persistent_hard.csv`",
            "- `top_improved.csv`",
            "- `top_degraded.csv`",
            "- `figures/exp41_vs_exp44_dropout_ba_scatter.png`",
            "- `figures/exp44_minus_exp41_delta_ba_hist.png`",
            "- `figures/delta_psnr_vs_delta_ba.png`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare per-image Dropout BA deltas between two O256 variants.")
    parser.add_argument("--per_image_csv", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--baseline", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold_ba", type=float, default=90.0)
    parser.add_argument("--left_image_dir", default="")
    parser.add_argument("--right_image_dir", default="")
    parser.add_argument("--gt_dir", default="")
    args = parser.parse_args()

    rows = build_rows(args)
    summary = summarize(rows, args.threshold_ba)
    write_outputs(rows, summary, args, Path(args.output_dir))
    print(f"Summary md: {Path(args.output_dir) / 'summary.md'}")
    print(f"Per-image delta csv: {Path(args.output_dir) / 'per_image_delta.csv'}")


if __name__ == "__main__":
    main()
