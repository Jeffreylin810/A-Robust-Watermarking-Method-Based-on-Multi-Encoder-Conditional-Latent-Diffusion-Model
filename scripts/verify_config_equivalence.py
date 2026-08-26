#!/usr/bin/env python3
import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--ignore", action="append", default=[])
    return parser.parse_args()


def remove_key(payload, dotted_key):
    parts = dotted_key.split(".")
    current = payload
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    args = parse_args()
    reference = deepcopy(load_config(args.reference))
    candidate = deepcopy(load_config(args.candidate))
    for dotted_key in args.ignore:
        remove_key(reference, dotted_key)
        remove_key(candidate, dotted_key)

    if reference != candidate:
        raise SystemExit(
            "Config equivalence check FAILED:\n"
            f"reference: {args.reference}\n"
            f"candidate: {args.candidate}\n"
            f"ignored keys: {args.ignore}"
        )

    print("Config equivalence check passed.")
    print(f"reference: {args.reference}")
    print(f"candidate: {args.candidate}")
    print(f"ignored keys: {args.ignore}")


if __name__ == "__main__":
    main()
