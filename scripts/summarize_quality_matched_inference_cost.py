#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize quality-matched inference cost comparison.")
    parser.add_argument("--exp_root", required=True)
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.5)
    return parser.parse_args()


def read_real_seconds(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"^real\s+([0-9.]+)\s*$", text, flags=re.MULTILINE)
    if not matches:
        raise RuntimeError(f"Could not parse real seconds from {log_path}")
    return float(matches[-1])


def read_psnr_ssim(log_path: Path) -> tuple[float, float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Testing set, PSNR is ([0-9.]+) and SSIM is ([0-9.]+)", text)
    if not m:
        raise RuntimeError(f"Could not parse PSNR/SSIM from {log_path}")
    return float(m.group(1)), float(m.group(2))


def read_lpips(summary_path: Path) -> float:
    return float(json.loads(summary_path.read_text(encoding="utf-8"))["mean_lpips"])


def fmt_hms(total_sec: float) -> str:
    total_sec = int(round(total_sec))
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    args = parse_args()
    exp_root = Path(args.exp_root)
    n = args.num_images
    gamma_tag = f"gamma_{args.gamma:.2f}".replace(".", "p")

    diffw_eval_sec = read_real_seconds(exp_root / "logs" / "diffw_raw_eval.log")
    gamma_apply_sec = read_real_seconds(exp_root / "logs" / f"{gamma_tag}_apply.log")
    o256_base_sec = read_real_seconds(exp_root / "logs" / "o256_base_eval.log")
    o256_refine_sec = read_real_seconds(exp_root / "logs" / "o256_rgb_refine.log")

    diffw_psnr, diffw_ssim = read_psnr_ssim(exp_root / "logs" / f"{gamma_tag}_psnr_ssim.log")
    diffw_lpips = read_lpips(exp_root / "lpips" / gamma_tag / "summary.json")
    o256_psnr, o256_ssim = read_psnr_ssim(exp_root / "logs" / "o256_rgb_psnr_ssim.log")
    o256_lpips = read_lpips(exp_root / "lpips" / "o256_rgb" / "summary.json")

    rows = [
        {
            "method": "DiffW-like raw",
            "pipeline": "eval_diffusion only",
            "total_seconds": f"{diffw_eval_sec:.2f}",
            "total_time_hms": fmt_hms(diffw_eval_sec),
            "ms_per_image": f"{1000.0 * diffw_eval_sec / n:.4f}",
            "psnr": "",
            "ssim": "",
            "lpips": "",
            "notes": "Reference raw formal DiffW-like timing before quality matching.",
        },
        {
            "method": f"DiffW-like + post-hoc {gamma_tag}",
            "pipeline": "eval_diffusion + gamma scaling",
            "total_seconds": f"{(diffw_eval_sec + gamma_apply_sec):.2f}",
            "total_time_hms": fmt_hms(diffw_eval_sec + gamma_apply_sec),
            "ms_per_image": f"{1000.0 * (diffw_eval_sec + gamma_apply_sec) / n:.4f}",
            "psnr": f"{diffw_psnr:.4f}",
            "ssim": f"{diffw_ssim:.4f}",
            "lpips": f"{diffw_lpips:.6f}",
            "notes": "Quality-matched DiffW-like deployment path.",
        },
        {
            "method": "O256-RGB",
            "pipeline": "O256 base eval + RGB refiner apply",
            "total_seconds": f"{(o256_base_sec + o256_refine_sec):.2f}",
            "total_time_hms": fmt_hms(o256_base_sec + o256_refine_sec),
            "ms_per_image": f"{1000.0 * (o256_base_sec + o256_refine_sec) / n:.4f}",
            "psnr": f"{o256_psnr:.4f}",
            "ssim": f"{o256_ssim:.4f}",
            "lpips": f"{o256_lpips:.6f}",
            "notes": "Formal high-quality O256-RGB pipeline.",
        },
    ]

    out_csv = exp_root / "quality_matched_inference_cost_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "pipeline",
                "total_seconds",
                "total_time_hms",
                "ms_per_image",
                "psnr",
                "ssim",
                "lpips",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    diffw_matched_ms = float(rows[1]["ms_per_image"])
    o256_ms = float(rows[2]["ms_per_image"])
    speedup = diffw_matched_ms / o256_ms if o256_ms else 0.0

    lines = [
        "# Quality-Matched Inference Cost Summary",
        "",
        f"- Matched-quality DiffW-like variant: post-hoc `{gamma_tag}`",
        f"- Number of images: `{n}`",
        "",
        "## Table",
        "",
        "| Method | Pipeline | Total time | ms/image | PSNR | SSIM | LPIPS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['pipeline']} | {row['total_time_hms']} | "
            f"{row['ms_per_image']} | {row['psnr']} | {row['ssim']} | {row['lpips']} |"
        )

    lines.extend(
        [
            "",
            "## Key readout",
            "",
            f"- Quality-matched DiffW-like pipeline ms/image: **{diffw_matched_ms:.4f}**",
            f"- O256-RGB pipeline ms/image: **{o256_ms:.4f}**",
            f"- O256-RGB speedup vs quality-matched DiffW-like: **{speedup:.2f}x**",
            "",
            "## Guardrail",
            "",
            "- This is a supplementary deployment-oriented timing comparison under the unified protocol.",
            "- The DiffW-like matched-quality point is achieved via post-hoc gamma scaling, not retraining.",
        ]
    )
    out_md = exp_root / "quality_matched_inference_cost_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"summary csv: {out_csv}")
    print(f"summary md: {out_md}")


if __name__ == "__main__":
    main()
