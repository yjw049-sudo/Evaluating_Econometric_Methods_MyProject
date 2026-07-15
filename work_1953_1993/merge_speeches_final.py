from pathlib import Path

import pandas as pd


WORK_DIR = Path(__file__).resolve().parent

PARLINFO_FILE = WORK_DIR / "output" / "speeches_with_parlinfo.csv"
GEMINI_FILE = WORK_DIR / "Gemini" / "speeches_topics_procedural_categories.csv"
KMEANS_FILE = WORK_DIR / "output" / "Kmeans" / "speeches_with_parlinfo_kmeans.csv"
CLUSTER_SCORE_FILE = WORK_DIR / "Gemini" / "cluster_procedural_scores_gemini.csv"
OUTPUT_FILE = WORK_DIR / "output" / "Speeches_final.csv"

KMEANS_COLUMNS = [
    "basepk",
    "cluster",
    "cluster_distance",
    "second_cluster",
    "second_cluster_distance",
]

CLUSTER_SCORE_COLUMNS = [
    "cluster",
    "procedural_score",
]


def main() -> None:
    print("Reading speeches_with_parlinfo.csv...")
    speeches = pd.read_csv(PARLINFO_FILE)

    print("Reading speeches_topics_procedural_categories.csv...")
    gemini = pd.read_csv(GEMINI_FILE)

    print("Reading speeches_with_parlinfo_kmeans.csv...")
    kmeans = pd.read_csv(KMEANS_FILE, usecols=KMEANS_COLUMNS)

    print("Reading cluster_procedural_scores_gemini.csv...")
    cluster_scores = pd.read_csv(CLUSTER_SCORE_FILE, usecols=CLUSTER_SCORE_COLUMNS)
    cluster_scores = cluster_scores.rename(columns={"procedural_score": "cluster_score"})

    gemini_total_rows = len(gemini)
    kmeans_total_rows = len(kmeans)
    cluster_score_total_rows = len(cluster_scores)

    error_x_has_value = gemini["error_x"].notna() & (
        gemini["error_x"].astype(str).str.strip() != ""
    )
    error_y_has_value = gemini["error_y"].notna() & (
        gemini["error_y"].astype(str).str.strip() != ""
    )

    clean_gemini = gemini.loc[~(error_x_has_value | error_y_has_value)].copy()
    removed_error_rows = gemini_total_rows - len(clean_gemini)

    merged = speeches.merge(
        clean_gemini,
        on="basepk",
        how="inner",
    )

    rows_after_gemini_merge = len(merged)

    merged = merged.merge(
        kmeans,
        on="basepk",
        how="inner",
    )

    rows_after_kmeans_merge = len(merged)

    merged = merged.merge(
        cluster_scores,
        on="cluster",
        how="inner",
    )

    merged = merged.drop(columns=["error_x", "error_y"])
    merged["disagreement"] = (
        merged["procedural_score"].astype(float) - merged["cluster_score"].astype(float)
    )

    unmatched_gemini_rows = len(speeches) - rows_after_gemini_merge
    unmatched_kmeans_rows = rows_after_gemini_merge - rows_after_kmeans_merge
    unmatched_cluster_score_rows = rows_after_kmeans_merge - len(merged)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Done.")
    print(f"Gemini rows: {gemini_total_rows}")
    print(f"Gemini rows removed because error_x/error_y has a value: {removed_error_rows}")
    print(f"Kmeans rows: {kmeans_total_rows}")
    print(f"Cluster score rows: {cluster_score_total_rows}")
    print(f"Parlinfo rows: {len(speeches)}")
    print(f"Rows removed because no clean Gemini match exists: {unmatched_gemini_rows}")
    print(f"Rows removed because no Kmeans match exists: {unmatched_kmeans_rows}")
    print(f"Rows removed because no cluster score match exists: {unmatched_cluster_score_rows}")
    print(f"Final rows: {len(merged)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
