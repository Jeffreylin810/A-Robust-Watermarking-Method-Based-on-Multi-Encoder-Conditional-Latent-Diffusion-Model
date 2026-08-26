#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DATASETS = [
    ("NaturalMix_DIV2K_BSDS_eval1000", 1000, "non_coco_natural_large"),
    ("Kodak_eval24", 24, "classic_quality_sanity"),
    ("ABO_demo_renders_eval273", 273, "cg_rendered_object_stress"),
]

METHODS = [
    ("base", "O256 base"),
    ("rgbrefine", "O256-RGB"),
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def parse_psnr_ssim(path: Path) -> tuple[float | None, float | None]:
    text = read_text(path)
    match = re.search(r"Testing set, PSNR is ([0-9.]+) and SSIM is ([0-9.]+)", text)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def parse_clean_ba(path: Path) -> float | None:
    text = read_text(path)
    match = re.search(r"\[Clean\] Average Bit Accuracy \(BA\): ([0-9.]+)%", text)
    return float(match.group(1)) if match else None


def parse_lpips(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["mean_lpips"])


def parse_attack_report(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    text = read_text(path)
    for line in text.splitlines():
        match = re.match(r"(Paper_.*?)\s+\|\s+([0-9.]+)", line)
        if match:
            out[match.group(1).strip()] = float(match.group(2))
    return out


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "missing"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", required=True)
    parser.add_argument(
        "--repo_root",
        default="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER",
    )
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    repo_root = Path(args.repo_root)
    logs = exp_root / "logs"

    rows = []
    attack_rows = []
    for dataset, count, role in DATASETS:
        for method_key, method_name in METHODS:
            log_prefix = "base" if method_key == "base" else "rgb"
            psnr, ssim = parse_psnr_ssim(logs / f"{dataset}_{log_prefix}_psnr_ssim.log")
            clean_ba = parse_clean_ba(logs / f"{dataset}_{log_prefix}_clean_ba.log")
            lpips = parse_lpips(exp_root / "lpips" / dataset / method_key / "summary.json")
            attack_name = f"non_coco_v2_{dataset}_{method_key if method_key == 'base' else 'rgbrefine'}_nongeom"
            attack_report = repo_root / "results" / "attacked_Result" / attack_name / "ba_report.txt"
            attacks = parse_attack_report(attack_report)
            nongeom_avg = sum(attacks.values()) / len(attacks) if attacks else None
            min_attack = min(attacks, key=attacks.get) if attacks else "missing"
            min_ba = attacks[min_attack] if attacks else None

            rows.append(
                {
                    "dataset": dataset,
                    "role": role,
                    "sample_count": count,
                    "method": method_name,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips,
                    "clean_ba": clean_ba,
                    "nongeom_avg_ba": nongeom_avg,
                    "min_ba": min_ba,
                    "min_attack": min_attack,
                    "attack_report": str(attack_report) if attack_report.exists() else "missing",
                }
            )
            for attack, ba in sorted(attacks.items()):
                attack_rows.append(
                    {
                        "dataset": dataset,
                        "sample_count": count,
                        "method": method_name,
                        "attack": attack,
                        "ba": ba,
                    }
                )

    summary_csv = exp_root / "domain_shift_v2_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "role",
                "sample_count",
                "method",
                "psnr",
                "ssim",
                "lpips",
                "clean_ba",
                "nongeom_avg_ba",
                "min_ba",
                "min_attack",
                "attack_report",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    attack_csv = exp_root / "domain_shift_v2_per_attack_ba.csv"
    with attack_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "sample_count", "method", "attack", "ba"],
        )
        writer.writeheader()
        writer.writerows(attack_rows)

    md_lines = [
        "# Non-COCO Domain Shift v2 Extended Summary",
        "",
        "This supplementary experiment evaluates the trained O256 base checkpoint and the trained RGB Refinement Module on larger non-COCO natural images and rendered object images without retraining.",
        "",
        "## Main Table",
        "",
        "| Dataset | Role | N | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Clean BA ↑ | Nongeom Avg BA ↑ | Minimum BA ↑ | Minimum attack |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {dataset} | {role} | {sample_count} | {method} | {psnr} | {ssim} | {lpips} | {clean_ba} | {avg} | {min_ba} | {min_attack} |".format(
                dataset=row["dataset"],
                role=row["role"],
                sample_count=row["sample_count"],
                method=row["method"],
                psnr=fmt(row["psnr"]),
                ssim=fmt(row["ssim"]),
                lpips=fmt(row["lpips"]),
                clean_ba=fmt_pct(row["clean_ba"]),
                avg=fmt_pct(row["nongeom_avg_ba"]),
                min_ba=fmt_pct(row["min_ba"]),
                min_attack=row["min_attack"],
            )
        )

    md_lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- summary CSV: `{summary_csv}`",
            f"- per-attack BA CSV: `{attack_csv}`",
            f"- run config: `{exp_root / 'run_config.md'}`",
            "",
            "## Interpretation Guardrail",
            "",
            "- This is a supplementary cross-domain/generalization check, not a replacement for the locked COCO finaltest1000 main protocol.",
            "- NaturalMix_DIV2K_BSDS_eval1000 is a mixed non-COCO natural-image set; Kodak remains a small classic quality sanity set.",
            "- ABO_demo_renders_eval273 should be described as rendered object images, not all possible CG domains.",
        ]
    )
    (exp_root / "domain_shift_v2_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"saved summary: {summary_csv}")
    print(f"saved per-attack: {attack_csv}")
    print(f"saved markdown: {exp_root / 'domain_shift_v2_summary.md'}")


if __name__ == "__main__":
    main()
