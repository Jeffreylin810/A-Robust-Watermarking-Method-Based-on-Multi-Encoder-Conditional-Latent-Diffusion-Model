#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


PATTERN = re.compile(r"bitwise_avg_err:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--max_mean", type=float, default=0.10)
    parser.add_argument("--max_last", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()
    text = Path(args.log).read_text(encoding="utf-8", errors="ignore")
    values = [float(match.group(1)) for match in PATTERN.finditer(text)]
    if len(values) < args.tail:
        raise SystemExit(
            f"Training convergence gate FAILED: only {len(values)} logged values "
            f"were found, expected at least {args.tail}."
        )

    recent = values[-args.tail :]
    mean_value = sum(recent) / len(recent)
    last_value = recent[-1]
    print(
        f"Message convergence gate: tail={args.tail}, "
        f"mean_bit_error={mean_value:.6f}, last_bit_error={last_value:.6f}"
    )
    if mean_value > args.max_mean or last_value > args.max_last:
        raise SystemExit(
            "Training convergence gate FAILED. Stop before continuation because "
            "the base model has not learned a reliable message signal."
        )
    print("Training convergence gate passed.")


if __name__ == "__main__":
    main()
