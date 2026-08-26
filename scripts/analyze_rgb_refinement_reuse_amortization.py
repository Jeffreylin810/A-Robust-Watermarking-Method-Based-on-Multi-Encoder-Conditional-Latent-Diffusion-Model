#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PAIR_SECONDS = 15 * 60 + 14.26
REFINER_SECONDS = 52 * 60 + 7.57
TOTAL_EXTRA_SECONDS = PAIR_SECONDS + REFINER_SECONDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze potential amortization of RGB refinement reuse.")
    parser.add_argument(
        "--output_root",
        default="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/rgb_refinement_reuse_amortization_20260627",
    )
    parser.add_argument(
        "--reuse_counts",
        nargs="+",
        type=int,
        default=[1, 2, 3, 5, 10],
    )
    return parser.parse_args()


def hhmmss(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for reuse_count in args.reuse_counts:
        per_use = TOTAL_EXTRA_SECONDS / reuse_count
        rows.append(
            {
                "reuse_count": reuse_count,
                "pair_generation_seconds": f"{PAIR_SECONDS:.2f}",
                "refiner_training_seconds": f"{REFINER_SECONDS:.2f}",
                "total_fixed_module_seconds": f"{TOTAL_EXTRA_SECONDS:.2f}",
                "amortized_extra_seconds_per_use": f"{per_use:.2f}",
                "amortized_extra_hhmmss_per_use": hhmmss(per_use),
                "notes": (
                    "Supplementary accounting only; this does not prove formal total cost superiority "
                    "over DiffW-like."
                ),
            }
        )

    csv_path = output_root / "amortized_module_reuse_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "pair_generation_seconds": PAIR_SECONDS,
        "refiner_training_seconds": REFINER_SECONDS,
        "total_fixed_module_seconds": TOTAL_EXTRA_SECONDS,
        "rows": rows,
        "guardrail": (
            "Use this as repeated-use amortization accounting for RGB refinement module reuse. "
            "Do not reinterpret it as formal crossover against DiffW-like complete preparation cost."
        ),
        "evidence_context": [
            "Same-COCO Module Reuse (alternate train10000 base)",
            "unseen COCO subset reuse eval100/eval300/eval1000",
        ],
    }
    (output_root / "amortized_module_reuse_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_lines = [
        "# RGB Refinement Reuse Amortization",
        "",
        f"- pair generation: `{hhmmss(PAIR_SECONDS)}`",
        f"- refiner training: `{hhmmss(REFINER_SECONDS)}`",
        f"- fixed module preparation total: `{hhmmss(TOTAL_EXTRA_SECONDS)}`",
        "",
        "| reuse count K | amortized extra cost per use | note |",
        "|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['reuse_count']} | {row['amortized_extra_hhmmss_per_use']} | potential repeated-use accounting only |"
        )
    md_lines += [
        "",
        "## Guardrail",
        "",
        "- This table is suitable for discussing `potential amortization via repeated reuse`.",
        "- It is **not** a replacement for the formal complete-preparation cost table.",
        "- It should be paired with actual reuse evidence such as Same-COCO Module Reuse and unseen COCO subset reuse.",
    ]
    (output_root / "amortized_module_reuse_summary.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
