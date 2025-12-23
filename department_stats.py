#!/usr/bin/env python3
"""
Count department occurrences in a JSONL file and print a summary table.

Assumes each line is a JSON object containing a "department" field.

Usage:
  python department_stats.py \
    --input /Users/yufei/Downloads/SDBench-main/shzyk/DiagnosisArena/data/test-00000-of-00001.with_dept.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Count department frequencies in a JSONL file.")
    parser.add_argument("--input", required=True, help="Path to labeled JSONL (with 'department').")
    args = parser.parse_args()

    input_path = Path(args.input)
    counter = Counter()
    total = 0

    with input_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            dept = obj.get("department", "Unknown")
            counter[dept] += 1
            total += 1

    print(f"Total records: {total}")
    print("Department counts:")
    for dept, count in counter.most_common():
        pct = (count / total * 100) if total else 0
        print(f"- {dept}: {count} ({pct:.2f}%)")


if __name__ == "__main__":
    main()

