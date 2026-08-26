#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


RUNS = (
    {
        "name": "Experiment 4 Stage B",
        "prefix": "exp4_refine10000_nongeomlightbase_A10k_B2k_seed61",
        "attack": "exp4_nongeom_refine10000_nongeomlightbase_A10k_B2k_seed61",
        "message_weight": 0.75,
        "extra_steps": 0,
        "candidate": False,
    },
    {
        "name": "Experiment 5A Stage B+",
        "prefix": "exp5A_refine10000_nongeomlightbase_BplusM0P90_500_seed61",
        "attack": "exp5A_nongeom_refine10000_nongeomlightbase_BplusM0P90_500_seed61",
        "message_weight": 0.90,
        "extra_steps": 500,
        "candidate": True,
    },
    {
        "name": "Experiment 5B Stage B+",
        "prefix": "exp5B_refine10000_nongeomlightbase_BplusM1P00_500_seed61",
        "attack": "exp5B_nongeom_refine10000_nongeomlightbase_BplusM1P00_500_seed61",
        "message_weight": 1.00,
        "extra_steps": 500,
        "candidate": True,
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_last(patterns, text, label):
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    raise RuntimeError(f"Could not parse {label}")


def read_run(root, spec):
    log_dir = root / "logs"
    psnr_text = (log_dir / f"{spec['prefix']}_psnr_ssim.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    clean_text = (log_dir / f"{spec['prefix']}_clean_ba.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    attack_summary = json.loads(
        (
            root
            / "results"
            / "attacked_Result"
            / spec["attack"]
            / "ba_summary.json"
        ).read_text(encoding="utf-8")
    )

    result = dict(spec)
    result["psnr"] = parse_last(
        (r"Testing set, PSNR is ([0-9.]+)",), psnr_text, "PSNR"
    )
    result["ssim"] = parse_last((r"SSIM is ([0-9.]+)",), psnr_text, "SSIM")
    result["clean_ba"] = parse_last(
        (
            r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9.]+)%",
            r"Clean\s+\|\s+([0-9.]+)",
        ),
        clean_text,
        "Clean BA",
    )
    result["average_ba"] = float(attack_summary["average_ba"])
    result["minimum_ba"] = float(attack_summary["minimum_ba"])
    result["minimum_attack"] = attack_summary["minimum_attack"]
    result["core_pass"] = (
        result["psnr"] >= 30.0
        and result["clean_ba"] >= 95.0
        and result["average_ba"] >= 70.0
    )
    result["strict_pass"] = result["core_pass"] and result["minimum_ba"] >= 70.0
    return result


def choose_candidate(results):
    candidates = [result for result in results if result["candidate"]]
    strict = [result for result in candidates if result["strict_pass"]]
    if strict:
        return max(
            strict,
            key=lambda item: (item["minimum_ba"], item["average_ba"], item["psnr"]),
        ), "strict"

    core = [result for result in candidates if result["core_pass"]]
    if core:
        return max(
            core,
            key=lambda item: (item["minimum_ba"], item["average_ba"], item["psnr"]),
        ), "core"

    viable = [
        result
        for result in candidates
        if result["psnr"] >= 30.0 and result["average_ba"] >= 70.0
    ]
    if viable:
        return max(
            viable,
            key=lambda item: (item["clean_ba"], item["minimum_ba"], item["psnr"]),
        ), "diagnostic"

    return max(
        candidates,
        key=lambda item: (
            item["core_pass"],
            item["clean_ba"],
            item["average_ba"],
            item["psnr"],
        ),
    ), "failed"


def main():
    args = parse_args()
    root = Path(args.project_root)
    results = [read_run(root, spec) for spec in RUNS]
    baseline = results[0]
    selected, selection_level = choose_candidate(results)

    rows = []
    for result in results:
        rows.append(
            "| {name} | {weight:.2f} | {steps} | {psnr:.4f} | {ssim:.4f} | "
            "{clean:.2f}% | {average:.2f}% | {minimum:.2f}% | {attack} | "
            "{core} | {strict} |".format(
                name=result["name"],
                weight=result["message_weight"],
                steps=result["extra_steps"],
                psnr=result["psnr"],
                ssim=result["ssim"],
                clean=result["clean_ba"],
                average=result["average_ba"],
                minimum=result["minimum_ba"],
                attack=result["minimum_attack"],
                core="PASS" if result["core_pass"] else "FAIL",
                strict="PASS" if result["strict_pass"] else "FAIL",
            )
        )

    delta_rows = []
    for result in results[1:]:
        delta_rows.append(
            "| {name} | {psnr:+.4f} | {clean:+.2f}% | {average:+.2f}% | "
            "{minimum:+.2f}% |".format(
                name=result["name"],
                psnr=result["psnr"] - baseline["psnr"],
                clean=result["clean_ba"] - baseline["clean_ba"],
                average=result["average_ba"] - baseline["average_ba"],
                minimum=result["minimum_ba"] - baseline["minimum_ba"],
            )
        )

    if selection_level == "strict":
        decision = (
            f"- `{selected['name']}` passes all four gates and becomes the next "
            "formal candidate. No targeted Stage C is required for this round."
        )
    elif selection_level == "core":
        decision = (
            f"- `{selected['name']}` passes PSNR, Clean BA, and average BA, but "
            "the weakest attack is still below 70%. Use this checkpoint as the "
            "starting point for one short targeted Stage C."
        )
    elif selection_level == "diagnostic":
        decision = (
            f"- Neither Stage B+ candidate reaches Clean BA 95%. "
            f"`{selected['name']}` is the closest diagnostic candidate, but it "
            "must not replace the formal anchor. Stop the Stage B scalar search "
            "and inspect the message-preserving objective before Stage C."
        )
    else:
        decision = (
            "- Both Stage B+ candidates fail the quality/robustness gates. Keep "
            "Experiment 4 as the reference and do not continue this branch."
        )

    text = f"""# Experiment 5: bounded Stage B+ calibration

Base: `train10000-base20k-then-nongeomlight5k`

Starting refiner: `refine10000-nongeomlightbase-A10k-B2k`

Controlled variable: `message_weight` only. Both candidates start from the same
Experiment 4 Stage B checkpoint and use 500 additional steps at `lr=2e-5`.

| Version | Message weight | Extra steps | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack | Core gate | Strict gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
{chr(10).join(rows)}

Core gate: PSNR >= 30 dB, Clean BA >= 95%, non-geometric average BA >= 70%.

Strict gate: core gate plus minimum BA >= 70%.

## Change from Experiment 4 Stage B

| Candidate | Delta PSNR | Delta Clean BA | Delta Avg BA | Delta Minimum BA |
|---|---:|---:|---:|---:|
{chr(10).join(delta_rows)}

## Decision

{decision}

Selected candidate for the decision above: `{selected['name']}`.
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
