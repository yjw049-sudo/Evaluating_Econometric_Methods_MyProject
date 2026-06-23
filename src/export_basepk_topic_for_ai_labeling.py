import pandas as pd

from project_config import (
    BASEPK_COLUMN,
    CSV_SEPARATOR,
    FULL_CORPUS_OUTPUT_DIR,
)

INPUT_FILE = (
    FULL_CORPUS_OUTPUT_DIR / "speeches_1920_1949_kmeans_ai_label_structure.csv"
)

OUTPUT_DIR = FULL_CORPUS_OUTPUT_DIR / "ai_topic_labeling"
OUTPUT_FILE = OUTPUT_DIR / "basepk_topic_for_ai_labeling.csv"
UNIQUE_TOPIC_FILE = OUTPUT_DIR / "unique_topics_for_ai_labeling.csv"


# =============================================================================
# Main workflow
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        usecols=[BASEPK_COLUMN, "topic"],
    )

    data = data.dropna(subset=[BASEPK_COLUMN, "topic"]).copy()
    data[BASEPK_COLUMN] = pd.to_numeric(
        data[BASEPK_COLUMN],
        errors="coerce",
    ).astype("Int64")
    data["topic"] = data["topic"].astype(str).str.strip()

    data = data.dropna(subset=[BASEPK_COLUMN])
    data = data[data["topic"] != ""].copy()
    data = data.sort_values(["topic", BASEPK_COLUMN]).reset_index(drop=True)

    unique_topics = (
        data[["topic"]]
        .drop_duplicates()
        .sort_values("topic")
        .reset_index(drop=True)
    )

    data.to_csv(OUTPUT_FILE, sep=CSV_SEPARATOR, index=False)
    unique_topics.to_csv(UNIQUE_TOPIC_FILE, sep=CSV_SEPARATOR, index=False)

    print(f"Saved basepk-topic file to: {OUTPUT_FILE}")
    print(f"Saved unique-topic file to: {UNIQUE_TOPIC_FILE}")
    print(f"Rows in basepk-topic file: {len(data)}")
    print(f"Unique topics: {len(unique_topics)}")


if __name__ == "__main__":
    main()
