"""Clean Speech_merged.csv using original speechtext length."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "output" / "Speech_merged.csv"
OUTPUT_FILE = PROJECT_DIR / "output" / "Speech_cleaned.csv"
TEMP_OUTPUT_FILE = PROJECT_DIR / "output" / "Speech_cleaned.tmp.csv"

ORIGINAL_TEXT_COLUMN = "speechtext"
ORIGINAL_WORD_COUNT_COLUMN = "speechtext_word_count"

MIN_WORD_COUNT = 50
PERCENTILES = [1, 25, 50, 75, 99]
CHUNK_SIZE = 10_000

REQUIRED_NON_MISSING_COLUMNS = [
    "speakername",
    ORIGINAL_TEXT_COLUMN,
    "speechdate",
    "speakerparty",
]


def validate_input():
    """Require Speech_merged.csv and the columns used for cleaning."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    columns = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()
    required_columns = [
        "basepk",
        "year",
        *REQUIRED_NON_MISSING_COLUMNS,
    ]
    missing_columns = [
        column for column in required_columns if column not in columns
    ]
    if missing_columns:
        raise ValueError(
            "Speech_merged.csv is missing columns required for cleaning: "
            f"{missing_columns}."
        )


def add_original_word_count(data):
    """Count whitespace-separated words in the original speechtext."""
    counted = data.copy()
    counted[ORIGINAL_WORD_COUNT_COLUMN] = (
        counted[ORIGINAL_TEXT_COLUMN].fillna("").astype(str).str.split().str.len()
    )
    return counted


def complete_value_mask(data, column):
    """Treat NaN, empty strings, and whitespace-only strings as missing."""
    return data[column].fillna("").astype(str).str.strip().ne("")


def complete_row_mask(data):
    """Keep rows with all required values; speakerposition may be missing."""
    keep_rows = pd.Series(True, index=data.index)
    for column in REQUIRED_NON_MISSING_COLUMNS:
        keep_rows &= complete_value_mask(data, column)
    return keep_rows


def calculate_percentiles_after_missing_filter():
    """Remove missing rows conceptually and calculate raw-text percentiles."""
    word_count_parts = []
    total_rows = 0
    complete_rows = 0

    for chunk in pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        dtype={"basepk": "string"},
        low_memory=False,
    ):
        chunk = add_original_word_count(chunk)
        keep_rows = complete_row_mask(chunk)
        word_count_parts.append(
            chunk.loc[keep_rows, ORIGINAL_WORD_COUNT_COLUMN].to_numpy(dtype="int32")
        )
        total_rows += len(chunk)
        complete_rows += int(keep_rows.sum())

    non_empty_parts = [part for part in word_count_parts if len(part) > 0]
    if not non_empty_parts:
        raise ValueError("No rows remain after removing incomplete records")

    all_word_counts = np.concatenate(non_empty_parts)
    percentile_values = np.percentile(all_word_counts, PERCENTILES)
    return total_rows, complete_rows, percentile_values


def write_cleaned_output(max_word_count):
    """Remove incomplete, short, and above-p99 rows and write the output."""
    if TEMP_OUTPUT_FILE.exists():
        TEMP_OUTPUT_FILE.unlink()

    kept_rows = 0
    removed_short_rows = 0
    removed_above_percentile = 0
    yearly_counts = pd.Series(dtype="int64")
    missing_year_rows = 0
    first_write = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            dtype={"basepk": "string"},
            low_memory=False,
        ),
        start=1,
    ):
        chunk = add_original_word_count(chunk)
        complete_rows = complete_row_mask(chunk)
        short_rows = complete_rows & chunk[ORIGINAL_WORD_COUNT_COLUMN].lt(
            MIN_WORD_COUNT
        )
        long_rows = (
            complete_rows
            & ~short_rows
            & chunk[ORIGINAL_WORD_COUNT_COLUMN].gt(max_word_count)
        )
        final_rows = complete_rows & ~short_rows & ~long_rows

        removed_short_rows += int(short_rows.sum())
        removed_above_percentile += int(long_rows.sum())
        cleaned = chunk.loc[final_rows].copy()
        kept_rows += len(cleaned)

        numeric_year = pd.to_numeric(cleaned["year"], errors="coerce")
        missing_year_rows += int(numeric_year.isna().sum())
        chunk_yearly_counts = numeric_year.dropna().astype(int).value_counts()
        yearly_counts = yearly_counts.add(chunk_yearly_counts, fill_value=0)

        if not cleaned.empty:
            cleaned.to_csv(
                TEMP_OUTPUT_FILE,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
                encoding="utf-8-sig" if first_write else "utf-8",
            )
            first_write = False

        print(f"Processed chunk {chunk_number}; retained: {kept_rows:,}")

    if first_write:
        raise ValueError("No rows remain after applying the length filters")

    TEMP_OUTPUT_FILE.replace(OUTPUT_FILE)
    yearly_counts = yearly_counts.fillna(0).astype(int).sort_index()
    return (
        kept_rows,
        removed_short_rows,
        removed_above_percentile,
        yearly_counts,
        missing_year_rows,
    )


def main():
    validate_input()
    total_rows, complete_rows, percentile_values = (
        calculate_percentiles_after_missing_filter()
    )

    print("\nAfter removing missing/blank required values")
    print(f"Input rows: {total_rows:,}")
    print(f"Rows remaining: {complete_rows:,}")
    print(f"Rows removed: {total_rows - complete_rows:,}")
    print("\nOriginal speechtext word-count percentiles")
    for percentile, value in zip(PERCENTILES, percentile_values):
        print(f"{percentile}%: {value:.2f}")

    max_word_count = float(percentile_values[-1])
    (
        kept_rows,
        removed_short_rows,
        removed_above_percentile,
        yearly_counts,
        missing_year_rows,
    ) = write_cleaned_output(max_word_count)

    print("\nLength filtering completed")
    print(f"Removed below {MIN_WORD_COUNT} words: {removed_short_rows:,}")
    print(
        f"Removed above the 99th percentile ({max_word_count:.2f} words): "
        f"{removed_above_percentile:,}"
    )

    print("\nRows by year")
    for year, count in yearly_counts.items():
        print(f"{int(year)}: {int(count):,}")
    if missing_year_rows > 0:
        print(f"Missing year: {missing_year_rows:,}")

    print(f"\nTotal cleaned rows: {kept_rows:,}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
