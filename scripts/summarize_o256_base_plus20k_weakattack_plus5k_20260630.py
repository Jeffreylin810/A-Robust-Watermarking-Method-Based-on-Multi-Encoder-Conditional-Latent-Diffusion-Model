#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


BASELINES = [
    {
        "variant": "formal_rgb",
        "psnr": 32.9034,
        "ssim": 0.9589,
        "lpips": 0.0461,
        "clean_ba": 95.81,
        "avg_ba": 87.38,
        "min_ba": 72.93,
        "min_attack": "Paper_Dropout_r03",
        "train_time": "02:33:57.31",
        "apply_time": "N/A",
        "kind": "rgb",
    },
    {
        "variant": "plus20k_rgb_reuse",
        "psnr": 32.2933,
        "ssim": 0.9579,
        "lpips": 0.05290933783538639,
        "clean_ba": 99.55,
        "avg_ba": 96.49181818181819,
        "min_ba": 85.68,
        "min_attack": "Paper_Dropout_r03",
        "train_time": "reuse_base_continuation",
        "apply_time": "65.43s",
        "kind": "rgb",
    },
    {
        "variant": "plus20k_refresh500",
        "psnr": 33.8299,
        "ssim": 0.9738,
        "lpips": 0.03830676269810647,
        "clean_ba": 99.39,
        "avg_ba": 95.91,
        "min_ba": 83.36,
        "min_attack": "Paper_Dropout_r03",
        "train_time": "283.14s",
        "apply_time": "61.13s",
        "kind": "rgb",
    },
]


def parse_last_float(patterns, text, label):
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    raise RuntimeError(f"Could not parse {label}")


def read_attack_rows(path: Path):
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"^\s*(\S.*?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*$", line)
        if match and match.group(1) != "Attack Name":
            rows[match.group(1).strip()] = float(match.group(2))
    if not rows:
        raise RuntimeError(f"No BA rows found in {path}")
    return rows


def read_real_seconds(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"real\s+([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def parse_variant(
    exp_root: Path,
    tag: str,
    attack_run: str,
    lpips_dir: str,
    psnr_log: str,
    clean_log: str,
    train_log: str,
    apply_log: str | None,
    resume_ckpt_kind: str,
):
    logs = exp_root / "logs"
    psnr_text = (logs / psnr_log).read_text(encoding="utf-8", errors="ignore")
    clean_text = (logs / clean_log).read_text(encoding="utf-8", errors="ignore")
    attack_rows = read_attack_rows(exp_root / "attacked_Result" / attack_run / "ba_report.txt")
    avg_ba = sum(attack_rows.values()) / len(attack_rows)
    min_attack = min(attack_rows, key=attack_rows.get)
    min_ba = attack_rows[min_attack]
    lpips_json = exp_root / "lpips" / lpips_dir / "summary.json"
    train_time = read_real_seconds(logs / train_log)
    apply_time = read_real_seconds(logs / apply_log) if apply_log else None
    return {
        "variant": tag,
        "psnr": parse_last_float((r"Testing set, PSNR is ([0-9.]+)",), psnr_text, "PSNR"),
        "ssim": parse_last_float((r"SSIM is ([0-9.]+)",), psnr_text, "SSIM"),
        "lpips": json.loads(lpips_json.read_text(encoding="utf-8"))["mean_lpips"],
        "clean_ba": parse_last_float((r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9.]+)%",), clean_text, "Clean BA"),
        "avg_ba": avg_ba,
        "min_ba": min_ba,
        "min_attack": min_attack,
        "train_time": f"{train_time:.2f}s" if train_time is not None else resume_ckpt_kind,
        "apply_time": f"{apply_time:.2f}s" if apply_time is not None else "N/A",
        "kind": "base" if apply_log is None else "rgb",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    exp_root = Path(args.experiment_root)
    rows = []
    rows.extend(BASELINES)
    rows.append(
        parse_variant(
            exp_root,
            "plus20k_weakattack_plus5k_base",
            "o256_base_plus20k_weakattack_plus5k_nongeom",
            "o256_base_plus20k_weakattack_plus5k",
            "plus20k_weakattack_plus5k_base_psnr_ssim.log",
            "plus20k_weakattack_plus5k_base_clean_ba.log",
            "plus20k_weakattack_plus5k_train.log",
            None,
            "resume_plus20k_base",
        )
    )
    rows.append(
        parse_variant(
            exp_root,
            "plus20k_weakattack_plus5k_rgb_reuse_refresh500",
            "o256_rgb_plus20k_weakattack_plus5k_reuse_refresh500_nongeom",
            "o256_rgb_plus20k_weakattack_plus5k_reuse_refresh500",
            "plus20k_weakattack_plus5k_rgb_reuse_psnr_ssim.log",
            "plus20k_weakattack_plus5k_rgb_reuse_clean_ba.log",
            "plus20k_weakattack_plus5k_train.log",
            "plus20k_weakattack_plus5k_rgb_reuse_apply.log",
            "resume_plus20k_base",
        )
    )

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "kind",
                "psnr",
                "ssim",
                "lpips",
                "clean_ba",
                "avg_ba",
                "min_ba",
                "min_attack",
                "train_time",
                "apply_time",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        "# O256 +20k weak-attack base continuation (+5k) summary",
        "",
        "## Main Table",
        "",
        "| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {variant} | {kind} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {clean_ba:.2f} | {avg_ba:.2f} | {min_ba:.2f} | {min_attack} | {train_time} | {apply_time} |".format(
                **row
            )
        )

    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `plus20k_weakattack_plus5k_base` continues the +20k base by one extra 5k stage using the new `identity_nongeom_targeted_weak` profile.",
            "- `plus20k_weakattack_plus5k_rgb_reuse_refresh500` applies the existing `plus20k_refresh500` refiner to that weak-targeted base to isolate the base-side effect on the final RGB operating point.",
            "- Baseline rows keep the current formal / +20k reference points in view for direct trade-off comparison.",
        ]
    )

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Summary md: {md_path}")
    print(f"Summary csv: {csv_path}")


if __name__ == "__main__":
    main()
