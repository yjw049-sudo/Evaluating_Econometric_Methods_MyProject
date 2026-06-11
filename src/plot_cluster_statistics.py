from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse


# =============================================================================
# Parameters
# =============================================================================

PROJECT_DIR = Path(
    r"E:\study\S2\Evaluating-Econometric-Methods_data\Evaluating_Econometric_Methods_MyProject"
)

OUTPUT_DIR = PROJECT_DIR / "output" / "1920-1949-v1"
MODEL_DIR = OUTPUT_DIR / "models"
STATISTICS_DIR = OUTPUT_DIR / "cluster_statistics"
FIGURE_DIR = OUTPUT_DIR / "figures"

DATA_FILE = OUTPUT_DIR / "speeches_1920_1949_second_stage.csv"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer_1930_1945.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_all_1920_1949.npz"

CLUSTER_SUMMARY_FILE = STATISTICS_DIR / "cluster_share_summary.csv"
CLUSTER_TOP_WORDS_FILE = STATISTICS_DIR / "cluster_top_10_words_by_tfidf_share.csv"
CLUSTER_SHARE_PLOT_FILE = FIGURE_DIR / "cluster_share_bar.png"

TOP_WORDS = 10
CSV_SEPARATOR = ";"


# =============================================================================
# Data loading
# =============================================================================

def load_inputs():
    data = pd.read_csv(DATA_FILE, sep=CSV_SEPARATOR)
    vectorizer = joblib.load(VECTORIZER_FILE)
    tfidf_matrix = sparse.load_npz(TFIDF_FILE)

    if tfidf_matrix.shape[0] != len(data):
        raise ValueError(
            "TF-IDF matrix row count does not match the second-stage data row count."
        )

    return data, vectorizer, tfidf_matrix


# =============================================================================
# Statistics
# =============================================================================

def calculate_cluster_summary(data):
    total_count = len(data)

    cluster_summary = (
        data.groupby("kmeans_cluster")
        .size()
        .reset_index(name="speech_count")
        .sort_values("kmeans_cluster")
    )

    cluster_summary["speech_share"] = cluster_summary["speech_count"] / total_count
    cluster_summary["speech_share_percent"] = cluster_summary["speech_share"] * 100

    return cluster_summary


def calculate_top_words_by_cluster(data, vectorizer, tfidf_matrix):
    feature_names = vectorizer.get_feature_names_out()
    top_word_rows = []

    for cluster_id in sorted(data["kmeans_cluster"].unique()):
        cluster_mask = data["kmeans_cluster"].to_numpy() == cluster_id
        cluster_tfidf = tfidf_matrix[cluster_mask]

        word_weights = cluster_tfidf.sum(axis=0).A1
        total_weight = word_weights.sum()

        if total_weight == 0:
            continue

        top_indices = word_weights.argsort()[::-1][:TOP_WORDS]

        for rank, word_index in enumerate(top_indices, start=1):
            word_weight = word_weights[word_index]

            top_word_rows.append(
                {
                    "kmeans_cluster": cluster_id,
                    "rank": rank,
                    "word": feature_names[word_index],
                    "tfidf_weight_sum": word_weight,
                    "word_share_in_cluster": word_weight / total_weight,
                    "word_share_percent_in_cluster": word_weight / total_weight * 100,
                }
            )

    top_words = pd.DataFrame(top_word_rows)
    return top_words


# =============================================================================
# Plots
# =============================================================================

def plot_cluster_share(cluster_summary):
    figure, axis = plt.subplots(figsize=(12, 6))

    axis.bar(
        cluster_summary["kmeans_cluster"].astype(str),
        cluster_summary["speech_share_percent"],
    )

    axis.set_xlabel("KMeans cluster")
    axis.set_ylabel("Share of speeches (%)")
    axis.set_title("Share of Speeches by KMeans Cluster")

    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)

    figure.tight_layout()
    figure.savefig(CLUSTER_SHARE_PLOT_FILE, dpi=300)
    plt.close(figure)


# =============================================================================
# Main workflow
# =============================================================================

def main():
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data, vectorizer, tfidf_matrix = load_inputs()

    cluster_summary = calculate_cluster_summary(data)
    cluster_top_words = calculate_top_words_by_cluster(data, vectorizer, tfidf_matrix)

    cluster_summary.to_csv(CLUSTER_SUMMARY_FILE, sep=CSV_SEPARATOR, index=False)
    cluster_top_words.to_csv(CLUSTER_TOP_WORDS_FILE, sep=CSV_SEPARATOR, index=False)

    plot_cluster_share(cluster_summary)

    print(f"Saved cluster summary to: {CLUSTER_SUMMARY_FILE}")
    print(f"Saved cluster top words to: {CLUSTER_TOP_WORDS_FILE}")
    print(f"Saved cluster share plot to: {CLUSTER_SHARE_PLOT_FILE}")


if __name__ == "__main__":
    main()
