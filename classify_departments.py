#!/usr/bin/env python3
"""
Classify each case in a JSONL file into a medical department using OpenAI.

Outputs a new JSONL where each original record gains a "department" field
(and optional "confidence" and "reason" metadata from the model response).

Usage:
  python classify_departments.py \
    --input /Users/yufei/Downloads/SDBench-main/shzyk/DiagnosisArena/data/test-00000-of-00001.jsonl \
    --output /Users/yufei/Downloads/SDBench-main/shzyk/DiagnosisArena/data/test-00000-of-00001.with_dept.jsonl \
    --model gpt-4o-mini
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI, OpenAIError

# Candidate departments to classify into.
DEPARTMENTS = [
    "Dermatology",
    "Endocrinology",
    "Gastroenterology/Hepatology",
    "Pulmonology",
    "Cardiology",
    "Neurology",
    "Otolaryngology (ENT)",
    "Ophthalmology",
    "Dentistry/Oral Surgery",
    "Urology/Andrology",
    "Obstetrics & Gynecology",
    "Pediatrics",
    "General Surgery/Breast & Thyroid",
    "Orthopedics",
    "Oncology",
    "Rheumatology",
    "Hematology",
    "Infectious Disease",
    "Nephrology",
    "Other",
]

# System prompt to steer the model.
SYSTEM_PROMPT = f"""You are a triage assistant in a tertiary hospital. Read the case
data and choose exactly one department from this list:
{", ".join(DEPARTMENTS)}.
If unsure, choose "Other".
Respond with JSON: {{"department": "...", "confidence": 0-1, "reason": "brief rationale"}}."""


def classify_case(
    client: OpenAI, case: Dict[str, Any], model: str, max_retries: int = 3
) -> Dict[str, Any]:
    """Call OpenAI to classify a single case."""
    user_content = (
        "Case details:\n"
        f"- History: {case.get('case_information', '')}\n"
        f"- Physical exam: {case.get('physical_examination', '')}\n"
        f"- Tests: {case.get('diagnostic_tests', '')}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            payload = json.loads(resp.choices[0].message.content)
            return {
                "department": payload.get("department", "Other"),
                "confidence": payload.get("confidence", ""),
                "reason": payload.get("reason", ""),
            }
        except OpenAIError:
            if attempt == max_retries:
                raise
            time.sleep(1.5 * attempt)


def main():
    parser = argparse.ArgumentParser(description="Classify cases by department.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model id")
    args = parser.parse_args()

    client = OpenAI()  # requires OPENAI_API_KEY in environment

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r") as infile, output_path.open("w") as outfile:
        for line in infile:
            if not line.strip():
                continue
            case = json.loads(line)
            result = classify_case(client, case, args.model)
            # Merge new fields into the record.
            case["department"] = result["department"]
            case["confidence"] = result["confidence"]
            case["reason"] = result["reason"]
            outfile.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Saved labeled JSONL to {output_path}")


if __name__ == "__main__":
    main()

