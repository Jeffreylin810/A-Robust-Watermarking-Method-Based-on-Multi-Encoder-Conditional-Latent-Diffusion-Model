#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


RUNS = (
    {
        "key": "full",
        "name": "Full reference",
        "steps": "10k / 2k / 500 / 500",
        "total_steps": 13000,
        "prefix": "exp13_full_final_seed61",
        "attack": "exp13_full_final_seed61_nongeom",
    },
    {
        "key": "reduced",
        "name": "Reduced",
        "steps": "5k / 1k / 250 / 250",
        "total_steps": 6500,
        "prefix": "exp13_reduced_final_seed61",
        "attack": "exp13_reduced_final_seed61_nongeom",
    },
    {
        "key": "minimal",
        "name": "Minimal",
        "steps": "3k / 1k / 250 / 250",
        "total_steps": 4500,
        "prefix": "exp13_minimal_final_seed61",
        "attack": "exp13_minimal_final_seed61_nongeom",
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--print_reduced_pass", action="store_true")
    return parser.parse_args()


def parse_last(patterns, text, label):
    if isinstance(patterns, str):
        patterns = (patterns,)
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            return float(matches[-1])
    raise RuntimeError(f"Could not parse {label}")


def read_run(root, spec):
    log_dir = root / "logs"
    psnr_path = log_dir / f"{spec['prefix']}_psnr_ssim.log"
    clean_path = log_dir / f"{spec['prefix']}_clean_ba.log"
    attack_dir = root / "results/attacked_Result" / spec["attack"]
    required = (
        psnr_path,
        clean_path,
        attack_dir / "ba_summary.json",
        attack_dir / "ba_report.txt",
    )
    if not all(path.exists() for path in required):
        return None

    psnr_text = psnr_path.read_text(encoding="utf-8", errors="ignore")
    clean_text = clean_path.read_text(encoding="utf-8", errors="ignore")
    attack = json.loads(
        (attack_dir / "ba_summary.json").read_text(encoding="utf-8")
    )
    return {
        **spec,
        "psnr": parse_last(
            r"Testing set,\s*PSNR is\s*([0-9.]+)", psnr_text, "PSNR"
        ),
        "ssim": parse_last(r"SSIM is\s*([0-9.]+)", psnr_text, "SSIM"),
        "clean_ba": parse_last(
            (
                r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9.]+)%",
                r"^\s*Clean\s*\|\s*([0-9.]+)\s*$",
            ),
            clean_text,
            "Clean BA",
        ),
        "average_ba": float(attack["average_ba"]),
        "minimum_ba": float(attack["minimum_ba"]),
        "minimum_attack": attack["minimum_attack"],
    }


def compare(candidate, full):
    delta = {
        metric: candidate[metric] - full[metric]
        for metric in ("psnr", "ssim", "clean_ba", "average_ba", "minimum_ba")
    }
    checks = {
        "psnr": delta["psnr"] >= -0.5,
        "clean_ba": delta["clean_ba"] >= -1.0,
        "average_ba": delta["average_ba"] >= -1.5,
        "minimum_ba": delta["minimum_ba"] >= -2.0,
    }
    return delta, checks, all(checks.values())


def main():
    args = parse_args()
    root = Path(args.project_root)
    results = {
        spec["key"]: read_run(root, spec)
        for spec in RUNS
    }
    if results["full"] is None or results["reduced"] is None:
        raise RuntimeError("Full and Reduced results are required")

    comparisons = {}
    for key in ("reduced", "minimal"):
        if results[key] is not None:
            delta, checks, passed = compare(results[key], results["full"])
            comparisons[key] = {
                "delta": delta,
                "checks": checks,
                "passed": passed,
            }

    if args.print_reduced_pass:
        print("true" if comparisons["reduced"]["passed"] else "false")
        return

    if results["minimal"] is not None and comparisons["minimal"]["passed"]:
        selected_key = "minimal"
    elif comparisons["reduced"]["passed"]:
        selected_key = "reduced"
    else:
        selected_key = "full"
    selected = results[selected_key]

    rows = []
    for spec in RUNS:
        result = results[spec["key"]]
        if result is None:
            rows.append(
                f"| {spec['name']} | {spec['steps']} | {spec['total_steps']} | "
                "not run | not run | not run | not run | not run |"
            )
            continue
        status = (
            "REFERENCE"
            if spec["key"] == "full"
            else ("PASS" if comparisons[spec["key"]]["passed"] else "FAIL")
        )
        rows.append(
            f"| {result['name']} | {result['steps']} | {result['total_steps']} | "
            f"{result['psnr']:.4f} | {result['ssim']:.4f} | "
            f"{result['clean_ba']:.2f}% | {result['average_ba']:.2f}% | "
            f"{result['minimum_ba']:.2f}% | {status} |"
        )

    delta_rows = []
    for key in ("reduced", "minimal"):
        if key not in comparisons:
            continue
        comparison = comparisons[key]
        delta = comparison["delta"]
        delta_rows.append(
            f"| {results[key]['name']} | {delta['psnr']:+.4f} | "
            f"{delta['clean_ba']:+.2f} | {delta['average_ba']:+.2f} | "
            f"{delta['minimum_ba']:+.2f} | "
            f"{'PASS' if comparison['passed'] else 'FAIL'} |"
        )

    text = f"""# Experiment 13: RGB Refinement Stage-Step Reduction

Classification: E (exploratory; dev1000 only)

All candidates use the same locked 30k O256 Base and the same refine10000 pairs.
Only Stage A/B/B+/C update counts differ.

| Version | A / B / B+ / C | Total refine-steps | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## Delta relative to Full reference

| Candidate | PSNR delta | Clean BA delta | Avg BA delta | Minimum BA delta | Gate |
|---|---:|---:|---:|---:|---|
{chr(10).join(delta_rows)}

Gate limits:

```text
PSNR drop <= 0.5 dB
Clean BA drop <= 1.0 pp
Nongeom Avg BA drop <= 1.5 pp
Minimum BA drop <= 2.0 pp
```

## Decision

Selected lowest-cost passing recipe: **{selected['name']}**

Selected Stage steps: **{selected['steps']}**

Selected total refine-steps: **{selected['total_steps']}**
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "results": results,
                "comparisons": comparisons,
                "selected": selected_key,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(text)


if __name__ == "__main__":
    main()
