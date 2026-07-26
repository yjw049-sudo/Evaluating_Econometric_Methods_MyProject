"""Merge final Gemini labels and K-means cluster scores into output/Speech_chunk.csv.

The script uses ``basepk`` and ``chunk_index`` as the composite merge key,
preserves every row and column in
Speech_chunk.csv, adds the final topic/category fields and ``cluster_score``, creates a
one-time backup, and atomically replaces the input CSV.

Run from Gemini_chunk:
    python 06_merge_gemini_results_into_speech_chunk.py
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT.parent

DEFAULT_INPUT_PATH = PROJECT_DIR / "output" / "Speech_chunk.csv"
DEFAULT_RESULTS_PATH = ROOT / "speeches_topics_procedural_categories.csv"
DEFAULT_CLUSTER_SCORES_PATH = ROOT.parent / "Gemini_merged" /"cluster_procedural_scores_gemini.csv"
DEFAULT_DEPENDENCY_PATHS = [
    ROOT / "speeches_topics_procedural.csv",
    ROOT / "topic_to_category.csv",
]
DEFAULT_BACKUP_PATH = PROJECT_DIR / "output" / "Speech_chunk_before_gemini_merge.csv"

KEY_COLUMNS = ["basepk", "chunk_index"]
RESULT_COLUMN_MAP = {
    "topic": "topic",
    "category": "category",
    "procedural_score": "procedural_score",
    "error_x": "topic_error",
    "error_y": "category_error",
    "error": "error",
}


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def prepare_results(results: pd.DataFrame) -> pd.DataFrame:
    missing_keys = [column for column in KEY_COLUMNS if column not in results.columns]
    if missing_keys:
        raise ValueError(f"Results file is missing required key columns: {missing_keys}")

    duplicate_count = int(results.duplicated(subset=KEY_COLUMNS).sum())
    if duplicate_count:
        raise ValueError(
            f"Results contain {duplicate_count:,} duplicate (basepk, chunk_index) keys"
        )

    available = [column for column in RESULT_COLUMN_MAP if column in results.columns]
    if not available:
        raise ValueError(
            "Results contain none of the expected Gemini columns: "
            f"{list(RESULT_COLUMN_MAP)}"
        )

    selected = results[[*KEY_COLUMNS, *available]].copy()
    return selected.rename(columns=RESULT_COLUMN_MAP)


def prepare_cluster_scores(cluster_scores: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["cluster", "cluster_score"]
    missing_columns = [
        column for column in required_columns if column not in cluster_scores.columns
    ]
    if missing_columns:
        raise ValueError(
            "Cluster scores file is missing required columns: "
            f"{missing_columns}"
        )

    duplicate_count = int(cluster_scores["cluster"].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"Cluster scores contain {duplicate_count:,} duplicate cluster values"
        )

    return cluster_scores[["cluster", "cluster_score"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge final Gemini results into output/Speech_chunk.csv"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument(
        "--cluster-scores", type=Path, default=DEFAULT_CLUSTER_SCORES_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP_PATH)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="do not create a backup when output overwrites input",
    )
    args = parser.parse_args()

    if args.results.exists() and args.results.resolve() == DEFAULT_RESULTS_PATH.resolve():
        newer_dependencies = [
            path
            for path in DEFAULT_DEPENDENCY_PATHS
            if path.exists() and path.stat().st_mtime > args.results.stat().st_mtime
        ]
        if newer_dependencies:
            names = ", ".join(path.name for path in newer_dependencies)
            raise RuntimeError(
                "Final Gemini results are older than their inputs "
                f"({names}). Run 04_merge_gemini_outputs.py first."
            )

    speeches = read_csv(args.input, "Speech chunk")
    results = prepare_results(read_csv(args.results, "Gemini results"))
    cluster_scores = prepare_cluster_scores(
        read_csv(args.cluster_scores, "Cluster scores")
    )

    missing_keys = [column for column in KEY_COLUMNS if column not in speeches.columns]
    if missing_keys:
        raise ValueError(f"Speech chunk file is missing required key columns: {missing_keys}")
    if "cluster" not in speeches.columns:
        raise ValueError("Speech chunk file is missing required column: cluster")

    duplicate_count = int(speeches.duplicated(subset=KEY_COLUMNS).sum())
    if duplicate_count:
        raise ValueError(
            f"Speech chunk file contains {duplicate_count:,} duplicate (basepk, chunk_index) keys"
        )

    result_columns = [column for column in results.columns if column not in KEY_COLUMNS]
    existing_result_columns = [
        column for column in result_columns if column in speeches.columns
    ]
    if existing_result_columns:
        speeches = speeches.drop(columns=existing_result_columns)
        print(f"Replacing existing result columns: {existing_result_columns}")
    if "cluster_score" in speeches.columns:
        speeches = speeches.drop(columns="cluster_score")
        print("Replacing existing result column: cluster_score")

    input_rows = len(speeches)
    merged = speeches.merge(results, on=KEY_COLUMNS, how="left", validate="one_to_one")
    merged = merged.merge(
        cluster_scores, on="cluster", how="left", validate="many_to_one"
    )
    if len(merged) != input_rows:
        raise RuntimeError(
            f"Row count changed during merge: {input_rows:,} -> {len(merged):,}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overwrites_input = args.output.resolve() == args.input.resolve()
    if overwrites_input and not args.no_backup:
        if args.backup.exists():
            print(f"Backup already exists; keeping it unchanged: {args.backup}")
        else:
            args.backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.input, args.backup)
            print(f"Created backup: {args.backup}")

    temporary_output = args.output.with_name(f"{args.output.name}.tmp")
    merged.to_csv(temporary_output, index=False, encoding="utf-8-sig")
    os.replace(temporary_output, args.output)

    matched = int(merged["topic"].notna().sum()) if "topic" in merged else 0
    missing_categories = (
        int(merged["category"].isna().sum())
        if "category" in merged
        else 0
    )
    print(f"Input rows: {input_rows:,}")
    print(f"Rows with topic: {matched:,}")
    print(f"Rows with missing category: {missing_categories:,}")
    matched_cluster_scores = int(merged["cluster_score"].notna().sum())
    print(f"Rows with cluster score: {matched_cluster_scores:,}")
    print(f"Wrote merged file: {args.output}")


if __name__ == "__main__":
    main()






















