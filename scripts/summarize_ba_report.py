#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize a DiffW ba_report.txt file.")
    parser.add_argument("report")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    report_path = Path(args.report)
    rows = []
    for line in report_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"^\s*(\S.*?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*$", line)
        if match and match.group(1) != "Attack Name":
            rows.append((match.group(1).strip(), float(match.group(2))))
    if not rows:
        raise RuntimeError(f"No BA rows found in {report_path}")

    values = [value for _, value in rows]
    minimum_name, minimum_value = min(rows, key=lambda item: item[1])
    summary = {
        "report": str(report_path),
        "count": len(rows),
        "average_ba": sum(values) / len(values),
        "minimum_ba": minimum_value,
        "minimum_attack": minimum_name,
        "maximum_ba": max(values),
    }
    text = (
        f"count: {summary['count']}\n"
        f"average_ba: {summary['average_ba']:.2f}\n"
        f"minimum_ba: {summary['minimum_ba']:.2f}\n"
        f"minimum_attack: {summary['minimum_attack']}\n"
        f"maximum_ba: {summary['maximum_ba']:.2f}\n"
    )
    print(text, end="")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        output_path.with_suffix(".json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
