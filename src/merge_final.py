from __future__ import annotations

"""
Merge chunk-level average scores into the original speech-level dataset.

Place this file at:
    src/merge_final.py

Inputs:
    output/Speech_chunk.csv
    output/Speech_sample.csv

Output:
    output/Speech_final.csv

Run from the project root:
    python src/merge_final.py
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# Paths
# =============================================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

CHUNK_FILE = PROJECT_ROOT / "output" / "Speech_chunk.csv"
SAMPLE_FILE = PROJECT_ROOT / "output" / "Speech_sample.csv"
OUTPUT_FILE = PROJECT_ROOT / "output" / "Speech_final.csv"


# =============================================================================
# Column settings
# =============================================================================

ID_COLUMN = "basepk"

# Score columns in Speech_chunk.csv
CHUNK_LLM_SCORE_COLUMN = "procedural_score"
CHUNK_KMEANS_SCORE_COLUMN = "cluster_score"

# New speech-level columns added to Speech_final.csv
OUTPUT_CHUNK_LLM_SCORE_COLUMN = "chunk_procedural_score"
OUTPUT_CHUNK_KMEANS_SCORE_COLUMN = "chunk_cluster_score"


# =============================================================================
# Helpers
# =============================================================================

def validate_file_exists(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file was not found: {file_path}")


def validate_columns(
    data: pd.DataFrame,
    required_columns: list[str],
    file_path: Path,
) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing required columns: "
            f"{missing_columns}\n"
            f"Available columns are: {list(data.columns)}"
        )


# =============================================================================
# Main processing
# =============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_file_exists(CHUNK_FILE)
    validate_file_exists(SAMPLE_FILE)

    chunk_data = pd.read_csv(
        CHUNK_FILE,
        low_memory=False,
    )

    sample_data = pd.read_csv(
        SAMPLE_FILE,
        low_memory=False,
    )

    validate_columns(
        chunk_data,
        [
            ID_COLUMN,
            CHUNK_LLM_SCORE_COLUMN,
            CHUNK_KMEANS_SCORE_COLUMN,
        ],
        CHUNK_FILE,
    )

    validate_columns(
        sample_data,
        [ID_COLUMN],
        SAMPLE_FILE,
    )

    return chunk_data, sample_data


def aggregate_chunk_scores(
    chunk_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate an equal-weight arithmetic mean across all chunks belonging
    to the same basepk.

    Example:
        basepk = 1001
        procedural_score = [0.75, 0.25, 0.00]

        chunk_procedural_score = (0.75 + 0.25 + 0.00) / 3
                               = 0.3333

    Missing score values are excluded from the corresponding mean by pandas.
    If all chunk scores for one speech are missing, the aggregated value is NaN.
    """
    data = chunk_data.copy()

    data[CHUNK_LLM_SCORE_COLUMN] = pd.to_numeric(
        data[CHUNK_LLM_SCORE_COLUMN],
        errors="coerce",
    )

    data[CHUNK_KMEANS_SCORE_COLUMN] = pd.to_numeric(
        data[CHUNK_KMEANS_SCORE_COLUMN],
        errors="coerce",
    )

    aggregated = (
        data.groupby(
            ID_COLUMN,
            as_index=False,
            dropna=False,
        )
        .agg(
            **{
                OUTPUT_CHUNK_LLM_SCORE_COLUMN: (
                    CHUNK_LLM_SCORE_COLUMN,
                    "mean",
                ),
                OUTPUT_CHUNK_KMEANS_SCORE_COLUMN: (
                    CHUNK_KMEANS_SCORE_COLUMN,
                    "mean",
                ),
                "chunk_row_count": (
                    ID_COLUMN,
                    "size",
                ),
                "valid_chunk_procedural_score_count": (
                    CHUNK_LLM_SCORE_COLUMN,
                    "count",
                ),
                "valid_chunk_cluster_score_count": (
                    CHUNK_KMEANS_SCORE_COLUMN,
                    "count",
                ),
            }
        )
    )

    return aggregated


def merge_scores(
    sample_data: pd.DataFrame,
    aggregated_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge aggregated chunk scores into Speech_sample.csv by basepk.

    A left merge is used so that every row in Speech_sample.csv is retained.
    """
    if sample_data[ID_COLUMN].duplicated().any():
        duplicate_count = int(
            sample_data[ID_COLUMN].duplicated(keep=False).sum()
        )
        print(
            "Warning: Speech_sample.csv contains duplicate basepk rows. "
            f"Number of rows involved in duplicates: {duplicate_count:,}"
        )

    merged = sample_data.merge(
        aggregated_scores,
        on=ID_COLUMN,
        how="left",
        validate="many_to_one",
    )

    return merged


def print_summary(
    chunk_data: pd.DataFrame,
    sample_data: pd.DataFrame,
    aggregated_scores: pd.DataFrame,
    merged_data: pd.DataFrame,
) -> None:
    matched_count = int(
        merged_data[OUTPUT_CHUNK_LLM_SCORE_COLUMN]
        .notna()
        .sum()
    )

    unmatched_count = int(
        merged_data[OUTPUT_CHUNK_LLM_SCORE_COLUMN]
        .isna()
        .sum()
    )

    print("\nMerge summary")
    print("-" * 60)
    print(f"Chunk rows: {len(chunk_data):,}")
    print(
        f"Unique basepk values in chunk data: "
        f"{chunk_data[ID_COLUMN].nunique(dropna=False):,}"
    )
    print(f"Rows in Speech_sample.csv: {len(sample_data):,}")
    print(
        f"Aggregated speech-level rows: "
        f"{len(aggregated_scores):,}"
    )
    print(f"Rows in Speech_final.csv: {len(merged_data):,}")
    print(
        f"Rows with a merged chunk procedural score: "
        f"{matched_count:,}"
    )
    print(
        f"Rows without a merged chunk procedural score: "
        f"{unmatched_count:,}"
    )
    print(f"Output file: {OUTPUT_FILE}")


def main() -> None:
    chunk_data, sample_data = load_data()

    aggregated_scores = aggregate_chunk_scores(
        chunk_data
    )

    merged_data = merge_scores(
        sample_data,
        aggregated_scores,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged_data.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(
        chunk_data=chunk_data,
        sample_data=sample_data,
        aggregated_scores=aggregated_scores,
        merged_data=merged_data,
    )


if __name__ == "__main__":
    main()
