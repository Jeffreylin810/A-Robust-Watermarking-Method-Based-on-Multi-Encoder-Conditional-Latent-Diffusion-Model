#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


TARGETS = {
    "psnr": 32.9034,
    "ssim": 0.9589,
    "lpips": 0.0461,
    "clean_ba": 95.81,
    "nongeom_avg_ba": 87.38,
    "minimum_ba": 72.93,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--mode", choices=["quick", "full"], required=True)
    return parser.parse_args()


def parse_last_float(patterns: tuple[str, ...], text: str, label: str) -> float:
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    raise RuntimeError(f"Could not parse {label}")


def parse_training_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"total_elapsed_sec=([0-9.]+)", text)
    if not matches:
        return None
    return float(matches[-1])


def seconds_to_hhmmss(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: int(row["steps"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def upsert_row(path: Path, row: dict[str, str], fields: list[str]) -> None:
    rows = read_csv_rows(path)
    rows = [existing for existing in rows if existing.get("steps") != row["steps"]]
    rows.append(row)
    write_csv_rows(path, rows, fields)


def yesno(value: bool) -> str:
    return "yes" if value else "no"


def decide_quick(psnr: float, ssim: float, lpips: float, clean_ba: float) -> str:
    if (
        psnr >= TARGETS["psnr"]
        and ssim >= TARGETS["ssim"]
        and lpips <= TARGETS["lpips"]
        and clean_ba >= TARGETS["clean_ba"]
    ):
        return "quick target passed; run full 11-attack gate"
    if psnr >= 32.0 and clean_ba >= 95.0:
        return "close to target; run full 11-attack gate and consider next checkpoint"
    if 30.0 <= psnr < 32.0 or 0.05 <= lpips <= 0.08:
        return "partially close; consider next checkpoint if time allows"
    if psnr < 30.0 and lpips > 0.08:
        return "stop recommended; fidelity target is still far"
    return "review manually"


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "unknown"
    return f"{value:.{digits}f}"


def summarize_markdown(exp_root: Path) -> None:
    quick_rows = read_csv_rows(exp_root / "quick_gate_results.csv")
    full_rows = read_csv_rows(exp_root / "full_gate_results.csv")

    lines = [
        "# DiffW-like Cost-to-Target Summary",
        "",
        "Classification: supplementary cost-to-target analysis, not a formal main-result replacement.",
        "",
        "## Target Thresholds",
        "",
        "| Metric | O256-RGB target | Pass condition |",
        "|---|---:|---|",
        f"| PSNR | {TARGETS['psnr']:.4f} | DiffW-like >= target |",
        f"| SSIM | {TARGETS['ssim']:.4f} | DiffW-like >= target |",
        f"| LPIPS | {TARGETS['lpips']:.4f} | DiffW-like <= target |",
        f"| Clean BA | {TARGETS['clean_ba']:.2f}% | DiffW-like >= target |",
        f"| Nongeom Avg BA | {TARGETS['nongeom_avg_ba']:.2f}% | DiffW-like >= target |",
        f"| Minimum BA | {TARGETS['minimum_ba']:.2f}% | DiffW-like >= target |",
        "",
        "## Quick Gate Results",
        "",
    ]
    if quick_rows:
        lines.extend(
            [
                "| Steps | Training cost | PSNR | SSIM | LPIPS | Clean BA | Decision |",
                "|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in quick_rows:
            lines.append(
                "| {steps} | {training_time_hhmmss} | {psnr} | {ssim} | {lpips} | {clean_ba}% | {quick_gate_decision} |".format(
                    **row
                )
            )
    else:
        lines.append("No quick gate rows yet.")

    lines.extend(["", "## Full Gate Results", ""])
    if full_rows:
        lines.extend(
            [
                "| Steps | PSNR | SSIM | LPIPS | Clean BA | Nongeom Avg BA | Minimum BA | Target status | Failed targets |",
                "|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in full_rows:
            lines.append(
                "| {steps} | {psnr} | {ssim} | {lpips} | {clean_ba}% | {nongeom_avg_ba}% | {minimum_ba}% | {all_targets_pass} | {failed_targets} |".format(
                    **row
                )
            )
    else:
        lines.append("No full 11-attack gate rows yet.")

    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- This analysis uses the O256-RGB formal result as a strict target threshold.",
            "- It should be described as a supplementary cost-to-target check.",
            "- Do not describe repeated finaltest1000 checks as recipe tuning for the formal main result.",
        ]
    )
    (exp_root / "cost_to_target_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_quick(args: argparse.Namespace) -> None:
    exp_root = Path(args.exp_root)
    steps = args.steps
    run_root = exp_root / "runs" / f"steps_{steps}"
    eval_dir = exp_root / "eval" / f"steps_{steps}" / "finaltest1000" / "CoCo" / "CoCo"

    psnr_text = (exp_root / "logs" / f"steps_{steps}_psnr_ssim.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    clean_text = (exp_root / "logs" / f"steps_{steps}_clean_ba.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    lpips_json = json.loads((exp_root / "lpips" / f"steps_{steps}" / "summary.json").read_text(encoding="utf-8"))
    training_seconds = parse_training_seconds(run_root / "logs" / "training_time.txt")

    psnr = parse_last_float((r"Testing set, PSNR is ([0-9.]+)",), psnr_text, "PSNR")
    ssim = parse_last_float((r"SSIM is ([0-9.]+)",), psnr_text, "SSIM")
    clean_ba = parse_last_float(
        (
            r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9.]+)%",
            r"Clean\s+\|\s+([0-9.]+)",
            r"Average Bit Accuracy \(BA\):\s*([0-9.]+)%",
        ),
        clean_text,
        "Clean BA",
    )
    lpips = float(lpips_json["mean_lpips"])
    decision = decide_quick(psnr, ssim, lpips, clean_ba)

    row = {
        "steps": str(steps),
        "checkpoint_path": str(run_root / "ckpts" / "CoCo_ddpm_final.pth.tar"),
        "training_seconds": "" if training_seconds is None else f"{training_seconds:.2f}",
        "training_time_hhmmss": seconds_to_hhmmss(training_seconds),
        "psnr": format_float(psnr),
        "ssim": format_float(ssim),
        "lpips": format_float(lpips, 6),
        "clean_ba": format_float(clean_ba, 2),
        "psnr_pass": yesno(psnr >= TARGETS["psnr"]),
        "ssim_pass": yesno(ssim >= TARGETS["ssim"]),
        "lpips_pass": yesno(lpips <= TARGETS["lpips"]),
        "clean_ba_pass": yesno(clean_ba >= TARGETS["clean_ba"]),
        "quick_gate_decision": decision,
        "notes": f"eval_dir={eval_dir}",
    }
    fields = [
        "steps",
        "checkpoint_path",
        "training_seconds",
        "training_time_hhmmss",
        "psnr",
        "ssim",
        "lpips",
        "clean_ba",
        "psnr_pass",
        "ssim_pass",
        "lpips_pass",
        "clean_ba_pass",
        "quick_gate_decision",
        "notes",
    ]
    upsert_row(exp_root / "quick_gate_results.csv", row, fields)
    summarize_markdown(exp_root)
    print(json.dumps(row, indent=2, ensure_ascii=False))


def summarize_full(args: argparse.Namespace) -> None:
    exp_root = Path(args.exp_root)
    steps = args.steps
    quick_rows = {int(row["steps"]): row for row in read_csv_rows(exp_root / "quick_gate_results.csv")}
    if steps not in quick_rows:
        raise RuntimeError(f"Missing quick gate row for steps={steps}; run quick first.")

    attack_summary = json.loads(
        (exp_root / "attacked" / f"nongeom_steps_{steps}" / "ba_summary.json").read_text(
            encoding="utf-8"
        )
    )
    quick = quick_rows[steps]
    nongeom_avg = float(attack_summary["average_ba"])
    min_ba = float(attack_summary["minimum_ba"])

    checks = {
        "psnr": float(quick["psnr"]) >= TARGETS["psnr"],
        "ssim": float(quick["ssim"]) >= TARGETS["ssim"],
        "lpips": float(quick["lpips"]) <= TARGETS["lpips"],
        "clean_ba": float(quick["clean_ba"]) >= TARGETS["clean_ba"],
        "nongeom_avg_ba": nongeom_avg >= TARGETS["nongeom_avg_ba"],
        "minimum_ba": min_ba >= TARGETS["minimum_ba"],
    }
    failed = [key for key, ok in checks.items() if not ok]
    row = {
        "steps": str(steps),
        "checkpoint_path": quick["checkpoint_path"],
        "training_seconds": quick["training_seconds"],
        "psnr": quick["psnr"],
        "ssim": quick["ssim"],
        "lpips": quick["lpips"],
        "clean_ba": quick["clean_ba"],
        "nongeom_avg_ba": format_float(nongeom_avg, 2),
        "minimum_ba": format_float(min_ba, 2),
        "minimum_attack": attack_summary["minimum_attack"],
        "all_targets_pass": yesno(not failed),
        "failed_targets": ",".join(failed) if failed else "none",
        "notes": f"attack_summary={exp_root / 'attacked' / f'nongeom_steps_{steps}' / 'ba_summary.json'}",
    }
    fields = [
        "steps",
        "checkpoint_path",
        "training_seconds",
        "psnr",
        "ssim",
        "lpips",
        "clean_ba",
        "nongeom_avg_ba",
        "minimum_ba",
        "minimum_attack",
        "all_targets_pass",
        "failed_targets",
        "notes",
    ]
    upsert_row(exp_root / "full_gate_results.csv", row, fields)
    summarize_markdown(exp_root)
    print(json.dumps(row, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.mode == "quick":
        summarize_quick(args)
    else:
        summarize_full(args)


if __name__ == "__main__":
    main()
