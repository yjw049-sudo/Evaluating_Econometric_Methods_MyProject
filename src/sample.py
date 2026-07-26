"""Create a reproducible 20% sample stratified by yearly word-count percentiles."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "output" / "Speech_cleaned.csv"
OUTPUT_FILE = PROJECT_DIR / "output" / "Speech_sample.csv"

YEAR_COLUMN = "year"
WORD_COUNT_COLUMN = "speechtext_word_count"
PERCENTILES = [1, 25, 50, 75, 99]
PERCENTILE_GROUPS = ["0-1", "1-25", "25-50", "50-75", "75-99", "99-100"]
SAMPLE_FRACTION = 0.10
RANDOM_SEED = 42
CHUNK_SIZE = 50_000


def validate_input():
    """Check that the input file and required columns are available."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    columns = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()
    required_columns = [YEAR_COLUMN, WORD_COUNT_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in columns
    ]
    if missing_columns:
        raise ValueError(
            f"Input file is missing required columns: {missing_columns}"
        )


def read_sampling_columns():
    """Read only the columns needed to define the sampling strata."""
    parts = []
    next_row_id = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=[YEAR_COLUMN, WORD_COUNT_COLUMN],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):
        prepared = chunk.copy()
        prepared[YEAR_COLUMN] = pd.to_numeric(
            prepared[YEAR_COLUMN], errors="coerce"
        )
        prepared[WORD_COUNT_COLUMN] = pd.to_numeric(
            prepared[WORD_COUNT_COLUMN], errors="coerce"
        )

        if prepared[[YEAR_COLUMN, WORD_COUNT_COLUMN]].isna().any().any():
            raise ValueError(
                "year and speechtext_word_count must not contain missing "
                "or non-numeric values"
            )

        prepared = prepared.astype(
            {YEAR_COLUMN: "int32", WORD_COUNT_COLUMN: "int32"}
        )
        prepared["_row_id"] = np.arange(
            next_row_id,
            next_row_id + len(prepared),
            dtype="int64",
        )
        next_row_id += len(prepared)
        parts.append(prepared)
        print(f"Sampling-data pass: processed chunk {chunk_number}")

    if not parts:
        raise ValueError("The input file contains no rows")

    return pd.concat(parts, ignore_index=True)


def add_percentile_groups(sampling_data):
    """Assign every row to a yearly word-count percentile interval."""
    grouped_parts = []

    for year, year_data in sampling_data.groupby(YEAR_COLUMN, sort=True):
        grouped = year_data.copy()
        thresholds = grouped[WORD_COUNT_COLUMN].quantile(
            [percentile / 100 for percentile in PERCENTILES]
        ).to_numpy()
        word_counts = grouped[WORD_COUNT_COLUMN].to_numpy()

        grouped["percentile_group"] = np.select(
            [word_counts <= threshold for threshold in thresholds],
            PERCENTILE_GROUPS[:-1],
            default=PERCENTILE_GROUPS[-1],
        )
        grouped_parts.append(grouped)

        threshold_text = ", ".join(
            f"p{percentile}={threshold:.2f}"
            for percentile, threshold in zip(PERCENTILES, thresholds)
        )
        print(f"Year {int(year)}: {threshold_text}")

    return pd.concat(grouped_parts, ignore_index=True)


def calculate_group_sample_sizes(sampling_data):
    """Allocate an exact overall 20% sample proportionally across strata."""
    group_columns = [YEAR_COLUMN, "percentile_group"]
    group_sizes = sampling_data.groupby(group_columns).size().rename("rows")

    allocation = group_sizes.to_frame()
    allocation["exact_sample_size"] = allocation["rows"] * SAMPLE_FRACTION
    allocation["sample_size"] = np.floor(
        allocation["exact_sample_size"]
    ).astype(int)
    allocation["remainder"] = (
        allocation["exact_sample_size"] - allocation["sample_size"]
    )

    target_total = int(len(sampling_data) * SAMPLE_FRACTION)
    rows_to_allocate = target_total - int(allocation["sample_size"].sum())

    if rows_to_allocate > 0:
        largest_remainders = allocation.sort_values(
            ["remainder", "rows"], ascending=[False, False]
        ).head(rows_to_allocate)
        allocation.loc[largest_remainders.index, "sample_size"] += 1

    allocated_total = int(allocation["sample_size"].sum())
    if allocated_total != target_total:
        raise RuntimeError(
            f"Expected an allocation of {target_total:,}, got {allocated_total:,}"
        )

    return allocation, target_total


def select_sample_rows(sampling_data, allocation):
    """Select row IDs independently within every sampling stratum."""
    selected_parts = []
    group_columns = [YEAR_COLUMN, "percentile_group"]

    for (year, percentile_group), group in sampling_data.groupby(
        group_columns, sort=True
    ):
        sample_size = int(
            allocation.loc[(year, percentile_group), "sample_size"]
        )
        if sample_size == 0:
            continue

        group_number = PERCENTILE_GROUPS.index(percentile_group)
        group_seed = RANDOM_SEED + int(year) * 10 + group_number
        selected_parts.append(
            group.sample(n=sample_size, random_state=group_seed)[
                ["_row_id", "percentile_group"]
            ]
        )

    selected_rows = pd.concat(selected_parts, ignore_index=True)
    return selected_rows.sort_values("_row_id").reset_index(drop=True)


def write_sample(selected_rows, total_input_rows):
    """Read the full CSV in chunks and write only the selected records."""
    group_by_row_id = selected_rows.set_index("_row_id")[
        "percentile_group"
    ]
    selected_mask = np.zeros(total_input_rows, dtype=bool)
    selected_mask[group_by_row_id.index.to_numpy(dtype="int64")] = True

    next_row_id = 0
    first_write = True
    written_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            dtype={"basepk": "string"},
            low_memory=False,
        ),
        start=1,
    ):
        end_row_id = next_row_id + len(chunk)
        keep_rows = selected_mask[next_row_id:end_row_id]
        sampled_chunk = chunk.loc[keep_rows].copy()
        sampled_row_ids = np.arange(
            next_row_id, end_row_id, dtype="int64"
        )[keep_rows]
        next_row_id = end_row_id

        sampled_chunk["percentile_group"] = group_by_row_id.loc[
            sampled_row_ids
        ].to_numpy()

        if not sampled_chunk.empty:
            sampled_chunk.to_csv(
                OUTPUT_FILE,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
                encoding="utf-8-sig" if first_write else "utf-8",
            )
            first_write = False
            written_rows += len(sampled_chunk)

        print(
            f"Output pass: processed chunk {chunk_number}; "
            f"written rows: {written_rows:,}"
        )

    return written_rows


def main():
    validate_input()
    sampling_data = read_sampling_columns()
    sampling_data = add_percentile_groups(sampling_data)
    allocation, target_total = calculate_group_sample_sizes(sampling_data)
    selected_rows = select_sample_rows(sampling_data, allocation)
    written_rows = write_sample(selected_rows, len(sampling_data))

    if written_rows != target_total:
        raise RuntimeError(
            f"Expected {target_total:,} output rows, wrote {written_rows:,}"
        )

    print("\nSampling completed")
    print(f"Input rows: {len(sampling_data):,}")
    print(f"Sampling fraction: {SAMPLE_FRACTION:.0%}")
    print(f"Total sampled rows: {written_rows:,}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()