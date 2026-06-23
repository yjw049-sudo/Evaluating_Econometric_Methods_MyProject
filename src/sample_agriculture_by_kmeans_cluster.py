import pandas as pd

from project_config import (
    AI_LABEL_COLUMN,
    BASEPK_COLUMN,
    CLUSTER_COLUMN,
    CSV_SEPARATOR,
    FULL_CORPUS_OUTPUT_DIR,
)

INPUT_FILE = (
    FULL_CORPUS_OUTPUT_DIR / "speeches_1920_1949_kmeans_ai_label_structure.csv"
)

OUTPUT_DIR = FULL_CORPUS_OUTPUT_DIR / "samples"
OUTPUT_FILE = OUTPUT_DIR / "agriculture_kmeans_clusters_sample.csv"

TARGET_CATEGORY = "Agriculture and Food Policy"
TARGET_CLUSTERS = [11, 15, 17, 5, 18]
SAMPLE_SIZE_PER_CLUSTER = 10
RANDOM_STATE = 42


# =============================================================================
# Main workflow
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_FILE, sep=CSV_SEPARATOR)

    filtered_data = data[
        (data[AI_LABEL_COLUMN] == TARGET_CATEGORY)
        & (data[CLUSTER_COLUMN].isin(TARGET_CLUSTERS))
    ].copy()

    sample_frames = []

    for cluster_id in TARGET_CLUSTERS:
        cluster_data = filtered_data[filtered_data[CLUSTER_COLUMN] == cluster_id]

        if len(cluster_data) < SAMPLE_SIZE_PER_CLUSTER:
            print(
                f"Warning: cluster {cluster_id} has only {len(cluster_data)} rows. "
                f"Sampling all available rows."
            )
            sampled_cluster_data = cluster_data.copy()
        else:
            sampled_cluster_data = cluster_data.sample(
                n=SAMPLE_SIZE_PER_CLUSTER,
                random_state=RANDOM_STATE,
            )

        sample_frames.append(sampled_cluster_data)

    sampled_data = pd.concat(sample_frames, ignore_index=True)
    sampled_data = sampled_data.sort_values(
        [CLUSTER_COLUMN, BASEPK_COLUMN]
    ).reset_index(drop=True)

    sampled_data.to_csv(OUTPUT_FILE, sep=CSV_SEPARATOR, index=False)

    print(f"Saved sampled data to: {OUTPUT_FILE}")
    print(f"Rows in sampled data: {len(sampled_data)}")
    print(sampled_data[CLUSTER_COLUMN].value_counts().sort_index())


if __name__ == "__main__":
    main()
