#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num_images", type=int, default=1000)
    return parser.parse_args()


def parse_real_seconds(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"^real\s+([0-9.]+)\s*$", text, flags=re.MULTILINE)
    if not matches:
      raise RuntimeError(f"Missing real time in log: {path}")
    return float(matches[-1])


def fmt_hhmmss(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def main() -> None:
    args = parse_args()
    exp_root = Path(args.exp_root).resolve()
    logs = exp_root / "logs"

    per_run_rows: list[dict[str, str]] = []
    aggregate_map: dict[str, list[float]] = {"DiffW-like": [], "O256-Base": [], "O256-RGB": []}

    for repeat in range(1, args.repeats + 1):
        diffw_seconds = parse_real_seconds(logs / f"repeat{repeat}_diffw_like_eval.log")
        base_seconds = parse_real_seconds(logs / f"repeat{repeat}_o256_base_eval.log")
        refine_seconds = parse_real_seconds(logs / f"repeat{repeat}_o256_rgb_refine.log")
        method_rows = [
            ("DiffW-like", diffw_seconds),
            ("O256-Base", base_seconds),
            ("O256-RGB", base_seconds + refine_seconds),
        ]
        diffw_ref = diffw_seconds
        for method, total_seconds in method_rows:
            aggregate_map[method].append(total_seconds)
            per_run_rows.append(
                {
                    "repeat": str(repeat),
                    "num_images": str(args.num_images),
                    "method": method,
                    "total_seconds": f"{total_seconds:.6f}",
                    "total_hhmmss": fmt_hhmmss(total_seconds),
                    "ms_per_image": f"{(total_seconds * 1000.0) / args.num_images:.6f}",
                    "images_per_second": f"{args.num_images / total_seconds:.6f}",
                    "speedup_vs_diffw_like": f"{diffw_ref / total_seconds:.6f}",
                }
            )

    per_run_csv = exp_root / "repeatability_per_run.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)

    summary_rows: list[dict[str, str]] = []
    for method, values in aggregate_map.items():
        mean_seconds = statistics.fmean(values)
        std_seconds = statistics.stdev(values) if len(values) > 1 else 0.0
        mean_ms = mean_seconds * 1000.0 / args.num_images
        std_ms = std_seconds * 1000.0 / args.num_images
        ci95 = 1.96 * std_ms / math.sqrt(len(values)) if values else 0.0
        summary_rows.append(
            {
                "num_images": str(args.num_images),
                "method": method,
                "repeats": str(args.repeats),
                "mean_total_seconds": f"{mean_seconds:.6f}",
                "mean_total_hhmmss": fmt_hhmmss(mean_seconds),
                "std_total_seconds": f"{std_seconds:.6f}",
                "mean_ms_per_image": f"{mean_ms:.6f}",
                "std_ms_per_image": f"{std_ms:.6f}",
                "ci95_half_width_ms": f"{ci95:.6f}",
                "cv_percent": f"{(100.0 * std_ms / mean_ms) if mean_ms else 0.0:.4f}",
                "mean_images_per_second": f"{args.num_images / mean_seconds:.6f}",
            }
        )

    summary_csv = exp_root / "repeatability_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    md_path = exp_root / "repeatability_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# End-to-End Repeatability Sanity\n\n")
        handle.write("- Classification: supplementary timing stability benchmark\n")
        handle.write(f"- repeats: `{args.repeats}`\n")
        handle.write(f"- num_images per repeat: `{args.num_images}`\n\n")
        handle.write("| Method | Mean total time | Mean ms/image | Std ms/image | CV (%) |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in summary_rows:
            handle.write(
                f"| {row['method']} | {row['mean_total_hhmmss']} | {row['mean_ms_per_image']} | "
                f"{row['std_ms_per_image']} | {row['cv_percent']} |\n"
            )
    payload = {
        "num_images": args.num_images,
        "repeats": args.repeats,
        "per_run_rows": per_run_rows,
        "summary_rows": summary_rows,
    }
    (exp_root / "repeatability_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved per-run CSV: {per_run_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
