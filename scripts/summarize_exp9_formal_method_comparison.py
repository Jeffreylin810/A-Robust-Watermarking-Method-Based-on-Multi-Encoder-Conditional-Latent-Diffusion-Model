#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


PSNR_PATTERN = re.compile(
    r"Testing set, PSNR is\s+([0-9.]+)\s+and SSIM is\s+([0-9.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffw_psnr_log", required=True)
    parser.add_argument("--diffw_clean_ba", required=True)
    parser.add_argument("--diffw_attack_ba", required=True)
    parser.add_argument("--ours_base_psnr_log", required=True)
    parser.add_argument("--ours_base_clean_ba", required=True)
    parser.add_argument("--ours_base_attack_ba", required=True)
    parser.add_argument("--ours_report_json", required=True)
    parser.add_argument("--ours_clean_ba", required=True)
    parser.add_argument("--ours_attack_ba", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_psnr_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    matches = PSNR_PATTERN.findall(text)
    if not matches:
        raise RuntimeError(f"Could not parse PSNR/SSIM from {path}")
    psnr, ssim = matches[-1]
    return float(psnr), float(ssim)


def parse_ba(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "|" not in line or "Attack Name" in line:
            continue
        left, right = line.split("|", 1)
        name = left.strip()
        try:
            value = float(right.strip())
        except ValueError:
            continue
        if name:
            rows[name] = value
    if not rows:
        raise RuntimeError(f"Could not parse BA rows from {path}")
    return rows


def single_ba(path):
    rows = parse_ba(path)
    if "Clean" in rows:
        return rows["Clean"]
    if len(rows) == 1:
        return next(iter(rows.values()))
    raise RuntimeError(f"Could not identify Clean BA in {path}")


def stats(rows):
    values = list(rows.values())
    minimum_name = min(rows, key=rows.get)
    return {
        "average": sum(values) / len(values),
        "minimum": rows[minimum_name],
        "minimum_name": minimum_name,
    }


def main():
    args = parse_args()
    diffw_psnr, diffw_ssim = parse_psnr_log(args.diffw_psnr_log)
    diffw_clean = single_ba(args.diffw_clean_ba)
    diffw_attacks = parse_ba(args.diffw_attack_ba)
    diffw_stats = stats(diffw_attacks)

    base_psnr, base_ssim = parse_psnr_log(args.ours_base_psnr_log)
    base_clean = single_ba(args.ours_base_clean_ba)
    base_attacks = parse_ba(args.ours_base_attack_ba)
    base_stats = stats(base_attacks)

    ours_report = json.loads(Path(args.ours_report_json).read_text(encoding="utf-8"))
    ours_psnr = float(ours_report["metrics"]["psnr"])
    ours_ssim = float(ours_report["metrics"]["ssim"])
    ours_clean = single_ba(args.ours_clean_ba)
    ours_attacks = parse_ba(args.ours_attack_ba)
    ours_stats = stats(ours_attacks)

    attack_names = sorted(set(diffw_attacks) | set(base_attacks) | set(ours_attacks))
    attack_rows = []
    for name in attack_names:
        diffw_value = diffw_attacks.get(name)
        base_value = base_attacks.get(name)
        ours_value = ours_attacks.get(name)
        delta = None
        if diffw_value is not None and ours_value is not None:
            delta = ours_value - diffw_value
        attack_rows.append(
                "| {} | {} | {} | {} | {} |".format(
                    name,
                    f"{diffw_value:.2f}" if diffw_value is not None else "N/A",
                    f"{base_value:.2f}" if base_value is not None else "N/A",
                    f"{ours_value:.2f}" if ours_value is not None else "N/A",
                f"{delta:+.2f}" if delta is not None else "N/A",
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    md = f"""# M01 / Experiment 9: Formal Method Comparison

## Claim Boundary

The `DiffW-like` row is our locally trained 256x256 / 32-bit controlled
baseline. It is not the untouched 128x128 DiffW paper result.

The DiffW paper reports that encoded-image PSNR was deliberately controlled
to `33.5 +/- 0.5 dB` for its comparison tables. The paper does not fully
disclose the strength-control procedure required to reproduce that operating
point. Therefore:

- We may claim that O256-RGBREFINE improves fidelity over our controlled
  DiffW-like baseline and our O256 base under the protocol below.
- We may not claim that O256-RGBREFINE has higher fidelity than the official
  DiffW paper result.
- The official paper number must be shown as a reported reference, not mixed
  into the same directly comparable row.

## Locked Protocol

- COCO train10000: byte-identical manifest
- COCO unseen eval1000
- Image size: 256x256
- Message length: 32 bits
- Main seed: 61
- Evaluation sampling steps: 3
- Robustness: 11 non-geometric attacks
- Geometry attacks: excluded

## Same and Different Settings

| Setting | DiffW-like | O256 base / O256-RGBREFINE | Alignment |
|---|---|---|---|
| Train images | same COCO train10000 manifest | same COCO train10000 manifest | same |
| Evaluation images | same unseen COCO eval1000 | same unseen COCO eval1000 | same |
| Image size | 256x256 | 256x256 | same |
| Message length | 32 bits | 32 bits | same |
| Main seed | 61 | 61 | same |
| Training batch size | 2 | 2 | same |
| Optimizer | Adam | Adam | same |
| Initial learning rate | 2e-5 | 2e-5 | same |
| Diffusion schedule | linear, 1000 steps | linear, 1000 steps | same |
| Evaluation sampling steps | 3 | 3 | same |
| Base training steps | 20k | 20k + 5k continuation | different |
| Continuation learning rate | none | 1e-5 | different |
| Message representation | linear | bitplane multi-scale | method difference |
| Training attack schedule | combined attack | identity -> nongeom-light | method difference |
| Latent VAE | no | frozen SD2.1 VAE | method difference |
| RGB refinement | no | refine10000 Stage A/B/B+/targeted C | method difference |

This is a final-method comparison. It is not an identical training-budget claim.

## Official Paper Reference

| Source | Image size | Payload | PSNR condition | Directly comparable here? |
|---|---:|---:|---:|---|
| DiffW paper reported setting | 128x128 | 30/64/etc. | 33.5 +/- 0.5 dB, explicitly strength-controlled | No |
| Our DiffW-like controlled baseline | 256x256 | 32 | naturally obtained 26.6109 dB | Yes |

## Main Results

| Method | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack |
|---|---:|---:|---:|---:|---:|---|
| DiffW-like | {diffw_psnr:.4f} | {diffw_ssim:.4f} | {diffw_clean:.2f}% | {diffw_stats['average']:.2f}% | {diffw_stats['minimum']:.2f}% | {diffw_stats['minimum_name']} |
| O256 base, no plugin | {base_psnr:.4f} | {base_ssim:.4f} | {base_clean:.2f}% | {base_stats['average']:.2f}% | {base_stats['minimum']:.2f}% | {base_stats['minimum_name']} |
| O256-RGBREFINE | {ours_psnr:.4f} | {ours_ssim:.4f} | {ours_clean:.2f}% | {ours_stats['average']:.2f}% | {ours_stats['minimum']:.2f}% | {ours_stats['minimum_name']} |

## Per-Attack BA

| Attack | DiffW-like | O256 base | O256-RGBREFINE | Final - DiffW-like |
|---|---:|---:|---:|---:|
{chr(10).join(attack_rows)}

## Source Paths

- DiffW PSNR log: `{args.diffw_psnr_log}`
- DiffW clean BA: `{args.diffw_clean_ba}`
- DiffW attack BA: `{args.diffw_attack_ba}`
- O256 base PSNR log: `{args.ours_base_psnr_log}`
- O256 base clean BA: `{args.ours_base_clean_ba}`
- O256 base attack BA: `{args.ours_base_attack_ba}`
- O256 report: `{args.ours_report_json}`
- O256 clean BA: `{args.ours_clean_ba}`
- O256 attack BA: `{args.ours_attack_ba}`
"""
    output.write_text(md, encoding="utf-8")
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
