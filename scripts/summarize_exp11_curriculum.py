#!/usr/bin/env python3
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "results/experiment_summaries/exp11_nongeomlight_curriculum_seed61"

RUNS = [
    {
        "name": "Control: identity20k -> light5k",
        "prefix": "exp2b_nongeom_train10000_base20k_then_light5k_seed61",
        "clean": "clean_exp2b_train10000_base20k_then_light5k_seed61",
        "attack": "exp2b_nongeom_train10000_base20k_then_light5k_seed61",
    },
    {
        "name": "Candidate A: light from scratch 20k",
        "prefix": "exp11_nongeomlight_fromscratch20k_seed61",
        "clean": "clean_exp11_nongeomlight_fromscratch20k_seed61",
        "attack": "exp11_nongeomlight_fromscratch20k_seed61_nongeom",
    },
    {
        "name": "Candidate B: light from scratch 25k",
        "prefix": "exp11_nongeomlight_fromscratch25k_seed61",
        "clean": "clean_exp11_nongeomlight_fromscratch25k_seed61",
        "attack": "exp11_nongeomlight_fromscratch25k_seed61_nongeom",
    },
]


def parse_psnr(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"Testing set,\s*PSNR is\s*([0-9.]+)\s*and SSIM is\s*([0-9.]+)", text
    )
    if not matches:
        raise RuntimeError(f"Could not parse PSNR/SSIM from {path}")
    psnr, ssim = matches[-1]
    return float(psnr), float(ssim)


def parse_clean_ba(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^\s*Clean\s*\|\s*([0-9.]+)\s*$", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse Clean BA from {path}")
    return float(match.group(1))


def load_run(spec):
    psnr, ssim = parse_psnr(ROOT / "logs" / f"{spec['prefix']}_psnr_ssim.log")
    clean_ba = parse_clean_ba(
        ROOT / "results/attacked_Result" / spec["clean"] / "ba_report.txt"
    )
    attack = json.loads(
        (
            ROOT
            / "results/attacked_Result"
            / spec["attack"]
            / "ba_summary.json"
        ).read_text(encoding="utf-8")
    )
    return {
        **spec,
        "psnr": psnr,
        "ssim": ssim,
        "clean_ba": clean_ba,
        "average_ba": float(attack["average_ba"]),
        "minimum_ba": float(attack["minimum_ba"]),
        "minimum_attack": attack["minimum_attack"],
    }


def within_tolerance(candidate, control):
    return {
        "psnr": candidate["psnr"] >= control["psnr"] - 0.3,
        "clean_ba": candidate["clean_ba"] >= control["clean_ba"] - 1.0,
        "average_ba": candidate["average_ba"] >= control["average_ba"] - 1.5,
        "minimum_ba": candidate["minimum_ba"] >= control["minimum_ba"] - 2.0,
    }


def main():
    results = [load_run(spec) for spec in RUNS]
    control = results[0]
    candidate_a = results[1]
    candidate_b = results[2]
    checks_a = within_tolerance(candidate_a, control)
    checks_b = within_tolerance(candidate_b, control)

    acceptable = [
        result
        for result, checks in ((candidate_a, checks_a), (candidate_b, checks_b))
        if all(checks.values())
    ]
    if acceptable:
        selected = max(
            acceptable,
            key=lambda row: (
                row["minimum_ba"],
                row["average_ba"],
                row["clean_ba"],
                row["psnr"],
            ),
        )
        decision = (
            f"{selected['name']} passes the control-relative tolerance and is the "
            "preferred from-scratch-light candidate. Exp12 should use this curriculum."
        )
    else:
        selected = control
        decision = (
            "Neither light-from-scratch candidate passes all tolerances. Keep the "
            "identity20k -> light5k curriculum for Exp12."
        )

    lines = [
        "# Experiment 11: Non-geometric Light Curriculum Comparison",
        "",
        "Classification: E (exploratory; dev1000 only)",
        "",
        "| Version | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['name']} | {row['psnr']:.4f} | {row['ssim']:.4f} | "
            f"{row['clean_ba']:.2f}% | {row['average_ba']:.2f}% | "
            f"{row['minimum_ba']:.2f}% | {row['minimum_attack']} |"
        )

    lines += [
        "",
        "## Control-relative tolerance",
        "",
        "| Candidate | PSNR >= control-0.3 | Clean >= control-1.0 | Avg >= control-1.5 | Min >= control-2.0 |",
        "|---|---:|---:|---:|---:|",
        f"| Candidate A | {checks_a['psnr']} | {checks_a['clean_ba']} | "
        f"{checks_a['average_ba']} | {checks_a['minimum_ba']} |",
        f"| Candidate B | {checks_b['psnr']} | {checks_b['clean_ba']} | "
        f"{checks_b['average_ba']} | {checks_b['minimum_ba']} |",
        "",
        "## Decision",
        "",
        decision,
        "",
        f"Selected branch for Exp12: **{selected['name']}**",
        "",
    ]

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (SUMMARY_DIR / "summary.json").write_text(
        json.dumps(
            {
                "results": results,
                "candidate_a_checks": checks_a,
                "candidate_b_checks": checks_b,
                "selected": selected["name"],
                "decision": decision,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(SUMMARY_DIR / "summary.md")


if __name__ == "__main__":
    main()

