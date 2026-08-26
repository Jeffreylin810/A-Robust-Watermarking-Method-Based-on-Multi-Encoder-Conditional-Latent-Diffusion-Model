#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


RUNS = (
    {
        "name": "Experiment 5B Stage B+",
        "prefix": "exp5B_refine10000_nongeomlightbase_BplusM1P00_500_seed61",
        "attack": "exp5B_nongeom_refine10000_nongeomlightbase_BplusM1P00_500_seed61",
        "stage_c_steps": 0,
        "candidate": False,
    },
    {
        "name": "Experiment 6 Stage C-250",
        "prefix": "exp6_refine10000_targetedformal_C250_seed61",
        "attack": "exp6_nongeom_refine10000_targetedformal_C250_seed61",
        "stage_c_steps": 250,
        "candidate": True,
    },
    {
        "name": "Experiment 6 Stage C-500",
        "prefix": "exp6_refine10000_targetedformal_C500_seed61",
        "attack": "exp6_nongeom_refine10000_targetedformal_C500_seed61",
        "stage_c_steps": 500,
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


def read_attack_rows(path):
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"^\s*(\S.*?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*$", line)
        if match and match.group(1) != "Attack Name":
            rows[match.group(1).strip()] = float(match.group(2))
    if not rows:
        raise RuntimeError(f"No BA rows found in {path}")
    return rows


def read_run(root, spec):
    log_dir = root / "logs"
    psnr_text = (log_dir / f"{spec['prefix']}_psnr_ssim.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    clean_text = (log_dir / f"{spec['prefix']}_clean_ba.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    attack_dir = root / "results" / "attacked_Result" / spec["attack"]
    attack_summary = json.loads(
        (attack_dir / "ba_summary.json").read_text(encoding="utf-8")
    )
    attacks = read_attack_rows(attack_dir / "ba_report.txt")

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
    result["attacks"] = attacks
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

    quality = [
        result
        for result in candidates
        if result["psnr"] >= 30.0 and result["average_ba"] >= 70.0
    ]
    if quality:
        return max(
            quality,
            key=lambda item: (item["minimum_ba"], item["clean_ba"], item["psnr"]),
        ), "diagnostic"

    return max(
        candidates,
        key=lambda item: (item["clean_ba"], item["minimum_ba"], item["psnr"]),
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
            "| {name} | {steps} | {psnr:.4f} | {ssim:.4f} | {clean:.2f}% | "
            "{average:.2f}% | {minimum:.2f}% | {attack} | {core} | {strict} |".format(
                name=result["name"],
                steps=result["stage_c_steps"],
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

    target_names = (
        "Paper_Dropout_r03",
        "Paper_GaussianNoise_arg03",
        "Paper_GaussianNoise_sigma03_var009",
        "Paper_SaltPepper_r02",
    )
    target_rows = []
    for attack_name in target_names:
        values = [result["attacks"][attack_name] for result in results]
        target_rows.append(
            f"| {attack_name} | {values[0]:.2f}% | {values[1]:.2f}% | "
            f"{values[2]:.2f}% | {values[2] - values[0]:+.2f}% |"
        )

    if selection_level == "strict":
        decision = (
            f"- `{selected['name']}` passes all gates and becomes the next formal "
            "candidate. Stop the targeted Stage C search."
        )
    elif selection_level == "core":
        decision = (
            f"- `{selected['name']}` passes the core gate but its weakest attack "
            "remains below 70%. Keep it as the best completed candidate; do not "
            "extend this scalar Stage C without a new objective."
        )
    elif selection_level == "diagnostic":
        decision = (
            f"- `{selected['name']}` is the best diagnostic candidate, but the "
            "core gate is still not met. Stop this targeted scalar search and "
            "reconsider the message-preserving objective."
        )
    else:
        decision = (
            "- Targeted Stage C does not preserve the required quality. Keep "
            "Experiment 5B as the reference and stop this branch."
        )

    text = f"""# Experiment 6: formal-scale targeted Stage C

Starting point: `Experiment 5B Stage B+`

Training profile:

```text
500 total steps, lr=1e-5
clean message weight=1.0
targeted attack message weight=0.15
formal-scale targets: GN arg03, GN sigma03, Dropout r03, SaltPepper r02
```

The 250-step and 500-step checkpoints come from the same training run.

| Version | Stage C steps | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack | Core gate | Strict gate |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
{chr(10).join(rows)}

## Targeted attacks

| Attack | 5B baseline | C-250 | C-500 | C-500 delta |
|---|---:|---:|---:|---:|
{chr(10).join(target_rows)}

Core gate: PSNR >= 30 dB, Clean BA >= 95%, non-geometric average BA >= 70%.

Strict gate: core gate plus minimum BA >= 70%.

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
