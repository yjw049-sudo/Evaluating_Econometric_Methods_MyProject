"""
Step 4: Merge speech-level topic/procedural output with topic-level categories.

Place and run this script inside:
    Gemini_merged

Inputs generated inside Gemini_merged:
    speeches_topics_procedural.csv
    topic_to_category.csv

Output:
    speeches_topics_procedural_categories.csv

Usage:
    python 04_merge_gemini_outputs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SPEECH_TOPIC_PATH = ROOT / "speeches_topics_procedural.csv"
TOPIC_CATEGORY_PATH = ROOT / "topic_to_category.csv"
OUTPUT_PATH = ROOT / "speeches_topics_procedural_categories.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speeches", type=Path, default=SPEECH_TOPIC_PATH)
    parser.add_argument("--topic-categories", type=Path, default=TOPIC_CATEGORY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    speeches = pd.read_csv(args.speeches, encoding="utf-8-sig")
    topic_categories = pd.read_csv(args.topic_categories, encoding="utf-8-sig")

    topic_categories = topic_categories.drop_duplicates(subset=["topic"], keep="first")
    merged = speeches.merge(topic_categories, on="topic", how="left")

    ordered_cols = [
        "basepk",
        "topic",
        "category",
        "is_high_procedural",
        "error",
    ]
    ordered_cols = [col for col in ordered_cols if col in merged.columns]
    remaining_cols = [col for col in merged.columns if col not in ordered_cols]
    merged = merged[ordered_cols + remaining_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(merged):,} rows to {args.output}")
    if "category" in merged.columns:
        print("\nCategory distribution:")
        print(merged["category"].value_counts(dropna=False).to_string())
    if "is_high_procedural" in merged.columns:
        print("\nHigh-procedural distribution:")
        print(merged["is_high_procedural"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()


