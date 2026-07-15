"""
Extract unique speakerposition values from Speeches_final.csv.

Place and run this script inside:
    work_1953_1993/Gemini/position_classification

Default input:
    ../../output/Speeches_final.csv

Default output:
    speakerposition_unique_values.csv

Usage:
    python 01_extract_unique_positions.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = ROOT.parent.parent / "output" / "Speeches_final.csv"
DEFAULT_OUTPUT_PATH = ROOT / "speakerposition_unique_values.csv"

POSITION_COLUMN = "speakerposition"


def clean_position(value: Any) -> str:
    """Normalize only obvious whitespace, without changing the substantive title."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return " ".join(text.split())


def extract_unique_positions_streaming(
    input_path: Path,
    position_column: str,
    encoding: str = "utf-8-sig",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Extract unique non-empty speakerposition values from a large CSV file.

    The function reads line by line and only keeps speakerposition, avoiding
    large text columns such as speechtext.
    """
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    counts: dict[str, int] = {}
    total_rows = 0
    missing_or_empty_rows = 0
    malformed_rows = 0

    with open(input_path, mode="r", encoding=encoding, newline="") as file:
        reader = csv.reader(file)

        try:
            header = next(reader)
        except StopIteration:
            sys.exit(f"Input file is empty: {input_path}")

        header = [column.strip("\ufeff") for column in header]

        if position_column not in header:
            sys.exit(
                f"Could not find position column: {position_column}. "
                f"Available columns: {header}"
            )

        position_index = header.index(position_column)

        for row_number, row in enumerate(reader, start=2):
            total_rows += 1

            if len(row) <= position_index:
                malformed_rows += 1
                continue

            position = clean_position(row[position_index])

            if not position:
                missing_or_empty_rows += 1
                continue

            counts[position] = counts.get(position, 0) + 1

    positions = sorted(counts.keys(), key=lambda x: x.lower())

    output = pd.DataFrame(
        {
            "position_id": list(range(len(positions))),
            "speakerposition": positions,
            "n_speeches_with_position": [counts[position] for position in positions],
        }
    )

    metadata = {
        "total_rows_scanned": total_rows,
        "unique_nonempty_positions": len(output),
        "missing_or_empty_rows": missing_or_empty_rows,
        "malformed_rows": malformed_rows,
        "position_column": position_column,
    }

    return output, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract unique speakerposition values.")

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--position-column", default=POSITION_COLUMN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)

    args = parser.parse_args()

    print(f"Input: {args.input}")
    print(f"Position column: {args.position_column}")

    unique_positions, metadata = extract_unique_positions_streaming(
        input_path=args.input,
        position_column=args.position_column,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    unique_positions.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\nExtraction summary:")
    for key, value in metadata.items():
        print(f"- {key}: {value}")

    print(f"\nSaved unique positions to: {args.output}")
    print("\nFirst 20 unique positions:")
    print(unique_positions.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
