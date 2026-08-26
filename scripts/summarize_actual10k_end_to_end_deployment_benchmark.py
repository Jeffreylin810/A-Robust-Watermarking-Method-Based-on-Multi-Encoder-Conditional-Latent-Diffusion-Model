#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", required=True)
    parser.add_argument("--num_images", type=int, default=10000)
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


def count_images(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg"})


def main() -> None:
    args = parse_args()
    exp_root = Path(args.exp_root).resolve()
    logs = exp_root / "logs"

    diffw_seconds = parse_real_seconds(logs / "deploy10k_diffw_like_eval.log")
    base_seconds = parse_real_seconds(logs / "deploy10k_o256_base_eval.log")
    refine_seconds = parse_real_seconds(logs / "deploy10k_o256_rgb_refine.log")
    o256_rgb_seconds = base_seconds + refine_seconds

    diffw_dir = exp_root / "end_to_end_outputs" / "diffw_like" / "deploy10k" / "deploy10k" / "CoCo" / "CoCo"
    base_dir = exp_root / "end_to_end_outputs" / "o256_base" / "deploy10k" / "deploy10k" / "CoCo" / "CoCo"
    rgb_dir = exp_root / "end_to_end_outputs" / "o256_rgb" / "deploy10k" / "apply" / "refined_images"

    methods = [
        {
            "method": "DiffW-like",
            "total_seconds": diffw_seconds,
            "notes": "actual 10k end-to-end eval_diffusion deployment run",
            "results_dir": str(diffw_dir),
            "num_output_images": count_images(diffw_dir),
        },
        {
            "method": "O256-Base",
            "total_seconds": base_seconds,
            "notes": "actual 10k end-to-end O256 base eval_diffusion deployment run",
            "results_dir": str(base_dir),
            "num_output_images": count_images(base_dir),
        },
        {
            "method": "O256-RGB",
            "total_seconds": o256_rgb_seconds,
            "notes": "actual 10k end-to-end O256 base eval_diffusion plus RGB refiner apply",
            "results_dir": str(rgb_dir),
            "num_output_images": count_images(rgb_dir),
        },
    ]

    diffw_ref = diffw_seconds
    rows: list[dict[str, str]] = []
    for entry in methods:
        total_seconds = float(entry["total_seconds"])
        rows.append(
            {
                "num_images": str(args.num_images),
                "method": entry["method"],
                "total_seconds": f"{total_seconds:.6f}",
                "total_hhmmss": fmt_hhmmss(total_seconds),
                "seconds_per_image": f"{total_seconds / args.num_images:.8f}",
                "ms_per_image": f"{(total_seconds * 1000.0) / args.num_images:.6f}",
                "images_per_second": f"{args.num_images / total_seconds:.6f}",
                "relative_to_diffw_like": f"{total_seconds / diffw_ref:.6f}",
                "speedup_vs_diffw_like": f"{diffw_ref / total_seconds:.6f}",
                "num_output_images": str(entry["num_output_images"]),
                "results_dir": entry["results_dir"],
                "notes": entry["notes"],
            }
        )

    csv_path = exp_root / "actual10k_end_to_end_deployment_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = exp_root / "actual10k_end_to_end_deployment_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Actual 10k End-to-End Deployment Benchmark\n\n")
        handle.write(f"- `num_images`: `{args.num_images}`\n")
        handle.write("- Classification: supplementary deployment throughput benchmark\n")
        handle.write("- Scope: model-prepared deployment only; does not include training cost\n\n")
        handle.write("| Method | Total time | ms/image | images/sec | vs DiffW-like |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['method']} | {row['total_hhmmss']} | {row['ms_per_image']} | "
                f"{row['images_per_second']} | {row['speedup_vs_diffw_like']}x |\n"
            )
        handle.write("\n")
        handle.write("## Notes\n")
        for row in rows:
            handle.write(f"- `{row['method']}`: {row['notes']}\n")
            handle.write(f"  - output images: `{row['num_output_images']}`\n")
            handle.write(f"  - results dir: `{row['results_dir']}`\n")

    payload = {
        "num_images": args.num_images,
        "rows": rows,
    }
    (exp_root / "actual10k_end_to_end_deployment_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
