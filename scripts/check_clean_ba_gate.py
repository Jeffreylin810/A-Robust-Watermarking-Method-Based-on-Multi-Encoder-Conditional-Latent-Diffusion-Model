#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--minimum", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    text = Path(args.report).read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^\s*Clean\s*\|\s*([0-9]+(?:\.[0-9]+)?)", text, re.MULTILINE)
    if not match:
        raise SystemExit(f"Could not parse Clean BA from {args.report}")

    clean_ba = float(match.group(1))
    print(f"Clean BA gate: measured={clean_ba:.2f}%, minimum={args.minimum:.2f}%")
    if clean_ba < args.minimum:
        raise SystemExit(
            "Clean BA gate FAILED. Stop before plugin transfer because the base "
            "checkpoint is not a valid message-bearing model."
        )
    print("Clean BA gate passed.")


if __name__ == "__main__":
    main()
