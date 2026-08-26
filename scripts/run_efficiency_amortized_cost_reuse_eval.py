#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path("/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER")
EXP_ROOT = Path(
    "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/"
    "efficiency_amortized_cost_experiment_20260624"
)
PY = Path("/home/jeffrey/miniconda3/envs/diffw/bin/python")

GT_DIR = Path(
    "/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/"
    "train10000_finaltest1000_locked/data/CoCo/val/gt"
)
BASE_DIR = ROOT / "results/test_result/m03_formal_seed61_base_finaltest1000/finaltest1000/CoCo"
REFINED_DIR = ROOT / "results/refinement/m03_formal_seed61_fullmodule_finaltest1000_apply/refined_images"
BASE_ATTACK_DIR = ROOT / "results/attacked_Result/m03_formal_seed61_base_finaltest1000_nongeom"
REFINED_ATTACK_DIR = ROOT / "results/attacked_Result/m03_formal_seed61_fullmodule_finaltest1000_nongeom"
BASE_CLEAN_DIR = ROOT / "results/attacked_Result/clean_m03_formal_seed61_base_finaltest1000/Clean"
REFINED_CLEAN_DIR = ROOT / "results/refinement/m03_formal_seed61_fullmodule_finaltest1000_apply/attacked_clean/Clean"
BASE_MESSAGE_DIR = ROOT / "results/test_result/m03_formal_seed61_base_finaltest1000/finaltest1000/CoCo"

CONFIG = "CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
RESUME = ROOT / "Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"

INFER_JSON = Path(
    "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/"
    "formal_inference_sanity_seed61/o256_formal_inference_sanity.json"
)

SUBSETS = (
    ("eval100", 100),
    ("eval300", 300),
    ("eval1000", 1000),
)

ATTACKS = (
    "Paper_Dropout_r03",
    "Paper_GaussianBlur_sigma2",
    "Paper_GaussianBlur_sigma3",
    "Paper_GaussianBlur_sigma4",
    "Paper_GaussianNoise_arg03",
    "Paper_GaussianNoise_sigma03_var009",
    "Paper_JPEG_Q30",
    "Paper_JPEG_Q40",
    "Paper_JPEG_Q50",
    "Paper_MedianFilter_w5",
    "Paper_SaltPepper_r02",
)


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND:\n")
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)


def list_gt_stems() -> list[str]:
    return sorted(path.stem for path in GT_DIR.iterdir() if path.is_file() and path.suffix.lower() in (".jpg", ".png"))


def ensure_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)


def build_subset_dirs(subset_name: str, stems: list[str]) -> dict[str, Path]:
    subset_root = EXP_ROOT / "subsets" / subset_name
    dirs = {
        "gt": subset_root / "gt",
        "base": subset_root / "base",
        "refined": subset_root / "refined",
        "messages": subset_root / "gt_messages",
        "base_clean_root": subset_root / "base_clean",
        "base_clean": subset_root / "base_clean" / "Clean",
        "refined_clean_root": subset_root / "refined_clean",
        "refined_clean": subset_root / "refined_clean" / "Clean",
        "base_nongeom": subset_root / "base_nongeom",
        "refined_nongeom": subset_root / "refined_nongeom",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        ensure_link(GT_DIR / f"{stem}.jpg", dirs["gt"] / f"{stem}.jpg")
        ensure_link(BASE_DIR / f"{stem}.png", dirs["base"] / f"{stem}.png")
        ensure_link(REFINED_DIR / f"{stem}.png", dirs["refined"] / f"{stem}.png")
        ensure_link(BASE_MESSAGE_DIR / f"{stem}_message.npy", dirs["messages"] / f"{stem}_message.npy")
        ensure_link(BASE_CLEAN_DIR / f"{stem}.png", dirs["base_clean"] / f"{stem}.png")
        ensure_link(REFINED_CLEAN_DIR / f"{stem}.png", dirs["refined_clean"] / f"{stem}.png")
        for attack in ATTACKS:
            ensure_link(BASE_ATTACK_DIR / attack / f"{stem}.png", dirs["base_nongeom"] / attack / f"{stem}.png")
            ensure_link(REFINED_ATTACK_DIR / attack / f"{stem}.png", dirs["refined_nongeom"] / attack / f"{stem}.png")
    return dirs


def parse_psnr_ssim(log_path: Path) -> tuple[float, float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    psnr_matches = re.findall(r"Testing set,\s*PSNR is\s*([0-9.]+)", text)
    ssim_matches = re.findall(r"SSIM is\s*([0-9.]+)", text)
    if not psnr_matches or not ssim_matches:
        raise RuntimeError(f"Could not parse PSNR/SSIM from {log_path}")
    return float(psnr_matches[-1]), float(ssim_matches[-1])


def parse_clean_ba(report: Path) -> float:
    text = report.read_text(encoding="utf-8", errors="ignore")
    patterns = (
        r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9]+(?:\.[0-9]+)?)%",
        r"^\s*Clean\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*$",
        r"^\s*Clean\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            return float(matches[-1])
    table_match = re.search(r"Clean\s*\|\s*([0-9]+(?:\.[0-9]+)?)", text)
    if table_match:
        return float(table_match.group(1))
    raise RuntimeError(f"Could not parse Clean BA from {report}")


def parse_ba_summary(json_path: Path) -> tuple[float, float]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return float(payload["average_ba"]), float(payload["minimum_ba"])


def load_inference_rows() -> dict[str, float]:
    payload = json.loads(INFER_JSON.read_text(encoding="utf-8"))
    rows = {(row["batch_size"], row["mode"]): row for row in payload["results"]}
    base = float(rows[(4, "base")]["mean_ms_per_image"]) / 1000.0
    full = float(rows[(4, "base_plus_refiner")]["mean_ms_per_image"]) / 1000.0
    return {
        "base_sec_per_image_bs4": base,
        "rgb_sec_per_image_bs4": full,
        "overhead_sec_per_image_bs4": full - base,
        "overhead_percent_bs4": ((full - base) / base) * 100.0,
        "gpu": payload["gpu"],
    }


def write_run_config(infer_rows: dict[str, float]) -> None:
    payload = {
        "experiment_name": "efficiency_amortized_cost_experiment_20260624",
        "date": "2026-06-24",
        "executed_now": ["Experiment C only"],
        "not_executed_now": ["Experiment A", "Experiment B"],
        "sources": {
            "gt_dir": str(GT_DIR),
            "base_dir": str(BASE_DIR),
            "refined_dir": str(REFINED_DIR),
            "base_attack_dir": str(BASE_ATTACK_DIR),
            "refined_attack_dir": str(REFINED_ATTACK_DIR),
            "base_clean_dir": str(BASE_CLEAN_DIR),
            "refined_clean_dir": str(REFINED_CLEAN_DIR),
            "message_dir": str(BASE_MESSAGE_DIR),
            "config": CONFIG,
            "resume": str(RESUME),
            "formal_inference_sanity_json": str(INFER_JSON),
        },
        "subset_policy": "Use the first N stems in lexicographic order from locked finaltest1000 GT as eval100/eval300/eval1000.",
        "subset_sizes": {name: count for name, count in SUBSETS},
        "timing_policy": {
            "seconds_per_image_source": "formal inference sanity batch_size=4",
            "note": "Subset inference total seconds are derived from previously measured formal bs=4 per-image timing, not newly benchmarked here.",
        },
        "attack_list": list(ATTACKS),
        "gpu_name": infer_rows["gpu"],
    }
    (EXP_ROOT / "run_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    stems_all = list_gt_stems()
    if len(stems_all) < 1000:
        raise RuntimeError("locked finaltest1000 does not contain 1000 GT images")

    infer = load_inference_rows()
    write_run_config(infer)

    csv_rows: list[dict[str, str]] = []
    summary_lines = [
        "# RGB Refinement Reuse Eval Summary",
        "",
        "- Experiment scope: **C only** for `efficiency_amortized_cost_experiment_20260624`.",
        "- No model retraining was performed.",
        "- No formal checkpoint or formal eval output was overwritten.",
        "- Metrics were recomputed on fixed subsets derived from locked `finaltest1000` using symlinked subset directories.",
        f"- Inference total seconds are derived from the confirmed formal bs=4 compute-only timing on `{infer['gpu']}`.",
        "",
        "| subset | method | psnr | ssim | lpips | clean_ba | nongeom_avg_ba | min_ba | inference_total_seconds | seconds_per_image | overhead_sec_per_image | overhead_percent |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for subset_name, count in SUBSETS:
        stems = stems_all[:count]
        dirs = build_subset_dirs(subset_name, stems)

        # Base metrics
        base_psnr_log = EXP_ROOT / "logs" / f"{subset_name}_base_psnr_ssim.log"
        run_logged(
            [str(PY), "calculate_psnr_ssim.py", "--gt_path", str(dirs["gt"]), "--results_path", str(dirs["base"])],
            base_psnr_log,
        )
        base_lpips_dir = EXP_ROOT / "lpips" / subset_name / "base"
        run_logged(
            [
                str(PY),
                "scripts/calculate_lpips_dir.py",
                "--gt_dir",
                str(dirs["gt"]),
                "--results_dir",
                str(dirs["base"]),
                "--output_dir",
                str(base_lpips_dir),
                "--batch_size",
                "8",
                "--net",
                "alex",
                "--device",
                "auto",
            ],
            EXP_ROOT / "logs" / f"{subset_name}_base_lpips.log",
        )
        run_logged(
            [
                str(PY),
                "calculate_ba.py",
                "--config",
                CONFIG,
                "--resume",
                str(RESUME),
                "--attack_dir",
                str(dirs["base_clean_root"]),
                "--gt_message_dir",
                str(dirs["messages"]),
            ],
            EXP_ROOT / "logs" / f"{subset_name}_base_clean_ba.log",
        )
        run_logged(
            [
                str(PY),
                "calculate_ba.py",
                "--config",
                CONFIG,
                "--resume",
                str(RESUME),
                "--attack_dir",
                str(dirs["base_nongeom"]),
                "--gt_message_dir",
                str(dirs["messages"]),
            ],
            EXP_ROOT / "logs" / f"{subset_name}_base_nongeom_ba.log",
        )
        run_logged(
            [
                str(PY),
                "scripts/summarize_ba_report.py",
                str(dirs["base_nongeom"] / "ba_report.txt"),
                "--output",
                str(dirs["base_nongeom"] / "ba_summary.txt"),
            ],
            EXP_ROOT / "logs" / f"{subset_name}_base_nongeom_summary.log",
        )

        # Refined metrics
        refined_psnr_log = EXP_ROOT / "logs" / f"{subset_name}_rgb_psnr_ssim.log"
        run_logged(
            [str(PY), "calculate_psnr_ssim.py", "--gt_path", str(dirs["gt"]), "--results_path", str(dirs["refined"])],
            refined_psnr_log,
        )
        refined_lpips_dir = EXP_ROOT / "lpips" / subset_name / "rgb"
        run_logged(
            [
                str(PY),
                "scripts/calculate_lpips_dir.py",
                "--gt_dir",
                str(dirs["gt"]),
                "--results_dir",
                str(dirs["refined"]),
                "--output_dir",
                str(refined_lpips_dir),
                "--batch_size",
                "8",
                "--net",
                "alex",
                "--device",
                "auto",
            ],
            EXP_ROOT / "logs" / f"{subset_name}_rgb_lpips.log",
        )
        run_logged(
            [
                str(PY),
                "calculate_ba.py",
                "--config",
                CONFIG,
                "--resume",
                str(RESUME),
                "--attack_dir",
                str(dirs["refined_clean_root"]),
                "--gt_message_dir",
                str(dirs["messages"]),
            ],
            EXP_ROOT / "logs" / f"{subset_name}_rgb_clean_ba.log",
        )
        run_logged(
            [
                str(PY),
                "calculate_ba.py",
                "--config",
                CONFIG,
                "--resume",
                str(RESUME),
                "--attack_dir",
                str(dirs["refined_nongeom"]),
                "--gt_message_dir",
                str(dirs["messages"]),
            ],
            EXP_ROOT / "logs" / f"{subset_name}_rgb_nongeom_ba.log",
        )
        run_logged(
            [
                str(PY),
                "scripts/summarize_ba_report.py",
                str(dirs["refined_nongeom"] / "ba_report.txt"),
                "--output",
                str(dirs["refined_nongeom"] / "ba_summary.txt"),
            ],
            EXP_ROOT / "logs" / f"{subset_name}_rgb_nongeom_summary.log",
        )

        base_psnr, base_ssim = parse_psnr_ssim(base_psnr_log)
        refined_psnr, refined_ssim = parse_psnr_ssim(refined_psnr_log)
        base_lpips = json.loads((base_lpips_dir / "summary.json").read_text(encoding="utf-8"))["mean_lpips"]
        refined_lpips = json.loads((refined_lpips_dir / "summary.json").read_text(encoding="utf-8"))["mean_lpips"]
        base_clean_ba = parse_clean_ba(dirs["base_clean_root"] / "ba_report.txt")
        refined_clean_ba = parse_clean_ba(dirs["refined_clean_root"] / "ba_report.txt")
        base_avg_ba, base_min_ba = parse_ba_summary(dirs["base_nongeom"] / "ba_summary.json")
        refined_avg_ba, refined_min_ba = parse_ba_summary(dirs["refined_nongeom"] / "ba_summary.json")

        for method, psnr, ssim, lpips, clean_ba, avg_ba, min_ba, sec_per_image in (
            ("O256-Base", base_psnr, base_ssim, base_lpips, base_clean_ba, base_avg_ba, base_min_ba, infer["base_sec_per_image_bs4"]),
            ("O256-RGB", refined_psnr, refined_ssim, refined_lpips, refined_clean_ba, refined_avg_ba, refined_min_ba, infer["rgb_sec_per_image_bs4"]),
        ):
            total_sec = sec_per_image * count
            csv_rows.append(
                {
                    "subset_name": subset_name,
                    "num_images": str(count),
                    "method": method,
                    "psnr": f"{psnr:.4f}",
                    "ssim": f"{ssim:.4f}",
                    "lpips": f"{lpips:.6f}",
                    "clean_ba": f"{clean_ba:.2f}",
                    "nongeom_avg_ba": f"{avg_ba:.2f}",
                    "min_ba": f"{min_ba:.2f}",
                    "inference_total_seconds": f"{total_sec:.4f}",
                    "seconds_per_image": f"{sec_per_image:.6f}",
                    "refinement_overhead_seconds_per_image": (
                        f"{infer['overhead_sec_per_image_bs4']:.6f}" if method == "O256-RGB" else "0.000000"
                    ),
                    "refinement_overhead_percent": (
                        f"{infer['overhead_percent_bs4']:.2f}" if method == "O256-RGB" else "0.00"
                    ),
                    "notes": (
                        "Timing derived from formal inference sanity bs=4; image-quality and BA recomputed on subset symlinks."
                    ),
                }
            )
            summary_lines.append(
                f"| {subset_name} | {method} | {psnr:.4f} | {ssim:.4f} | {lpips:.6f} | {clean_ba:.2f}% | {avg_ba:.2f}% | {min_ba:.2f}% | {total_sec:.4f} | {sec_per_image:.6f} | "
                + (
                    f"{infer['overhead_sec_per_image_bs4']:.6f} | {infer['overhead_percent_bs4']:.2f}% |"
                    if method == "O256-RGB"
                    else "0.000000 | 0.00% |"
                )
            )

    csv_path = EXP_ROOT / "rgb_refinement_reuse_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subset_name",
                "num_images",
                "method",
                "psnr",
                "ssim",
                "lpips",
                "clean_ba",
                "nongeom_avg_ba",
                "min_ba",
                "inference_total_seconds",
                "seconds_per_image",
                "refinement_overhead_seconds_per_image",
                "refinement_overhead_percent",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    # Summary / recommendation
    summary_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `O256-RGB` maintains PSNR > 30 dB on all evaluated unseen COCO subsets if all subset rows above remain above 30 dB.",
            "- Clean BA / Nongeom Avg BA / Minimum BA should be compared against the formal full-result reference (`95.81% / 87.38% / 72.93%`).",
            "- Inference overhead is kept small because the per-image overhead is inherited from the confirmed formal inference sanity result (+1.08% at bs=4).",
            "- This experiment supports the practical claim that the RGB refinement module can be repeatedly applied to new images from the same COCO distribution without retraining, provided the subset metrics remain close to the formal full-result reference.",
            "",
            "## Recommendation",
            "",
            "- If the subset metrics remain stable across eval100 / eval300 / eval1000, this can be considered for a short supporting paragraph in Chapter 4.5 or moved to Chapter 5 discussion if the mainline already feels crowded.",
            "- Because subset inference totals are derived from the confirmed formal bs=4 timing rather than newly benchmarked here, use conservative wording and avoid presenting them as a new hardware benchmark.",
        ]
    )
    (EXP_ROOT / "rgb_refinement_reuse_eval_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"saved csv: {csv_path}")
    print(f"saved summary: {EXP_ROOT / 'rgb_refinement_reuse_eval_summary.md'}")


if __name__ == "__main__":
    main()
