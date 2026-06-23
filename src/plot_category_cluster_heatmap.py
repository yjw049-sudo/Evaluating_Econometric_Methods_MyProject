import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from project_config import (
    AI_LABEL_COLUMN,
    CLUSTER_COLUMN,
    CSV_SEPARATOR,
    FULL_CORPUS_OUTPUT_DIR,
)

INPUT_FILE = (
    FULL_CORPUS_OUTPUT_DIR / "speeches_1920_1949_kmeans_ai_label_structure.csv"
)

FIGURE_DIR = FULL_CORPUS_OUTPUT_DIR / "figures"
TABLE_DIR = FULL_CORPUS_OUTPUT_DIR / "tables"

COUNTS_FILE = TABLE_DIR / "category_by_kmeans_cluster_counts.csv"
CATEGORY_TO_CLUSTER_SHARES_FILE = TABLE_DIR / "category_to_kmeans_cluster_shares.csv"
CLUSTER_TO_CATEGORY_SHARES_FILE = TABLE_DIR / "kmeans_cluster_to_category_shares.csv"
CATEGORY_TO_CLUSTER_HEATMAP_FILE = (
    FIGURE_DIR / "category_to_kmeans_cluster_share_heatmap.png"
)
CLUSTER_TO_CATEGORY_HEATMAP_FILE = (
    FIGURE_DIR / "kmeans_cluster_to_category_share_heatmap.png"
)

CATEGORY_COLUMN = AI_LABEL_COLUMN
Y_LABEL_WRAP_WIDTH = 28


# =============================================================================
# Data preparation
# =============================================================================

def load_data():
    data = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        usecols=[CATEGORY_COLUMN, CLUSTER_COLUMN],
    )

    data = data.dropna(subset=[CATEGORY_COLUMN, CLUSTER_COLUMN]).copy()
    data[CLUSTER_COLUMN] = data[CLUSTER_COLUMN].astype(int)

    return data


def create_category_cluster_tables(data):
    counts = pd.crosstab(
        index=data[CATEGORY_COLUMN],
        columns=data[CLUSTER_COLUMN],
    )

    counts = counts.reindex(sorted(counts.columns), axis=1)
    category_to_cluster_shares = counts.div(counts.sum(axis=1), axis=0)
    cluster_to_category_shares = counts.div(counts.sum(axis=0), axis=1).T

    return counts, category_to_cluster_shares, cluster_to_category_shares


# =============================================================================
# Plotting
# =============================================================================

def plot_category_to_cluster_heatmap(category_to_cluster_shares):
    wrapped_row_shares = category_to_cluster_shares.copy()
    wrapped_row_shares.index = [
        textwrap.fill(label, width=Y_LABEL_WRAP_WIDTH)
        for label in wrapped_row_shares.index
    ]

    figure_height = max(8, 0.75 * len(wrapped_row_shares))
    plt.figure(figsize=(16, figure_height))

    axis = sns.heatmap(
        wrapped_row_shares,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Share within category"},
    )

    axis.set_xlabel("KMeans cluster")
    axis.set_ylabel("AI category")
    axis.set_title("Share of Each AI Category Falling into Each KMeans Cluster")

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.subplots_adjust(left=0.22)
    plt.savefig(CATEGORY_TO_CLUSTER_HEATMAP_FILE, dpi=300)
    plt.close()


def plot_cluster_to_category_heatmap(cluster_to_category_shares):
    category_by_cluster_shares = cluster_to_category_shares.T

    wrapped_category_by_cluster_shares = category_by_cluster_shares.copy()
    wrapped_category_by_cluster_shares.index = [
        textwrap.fill(label, width=Y_LABEL_WRAP_WIDTH)
        for label in wrapped_category_by_cluster_shares.index
    ]

    figure_height = max(8, 0.75 * len(wrapped_category_by_cluster_shares))
    plt.figure(figsize=(16, figure_height))

    axis = sns.heatmap(
        wrapped_category_by_cluster_shares,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Share within KMeans cluster"},
    )

    axis.set_xlabel("KMeans cluster")
    axis.set_ylabel("AI category")
    axis.set_title("Share of Each KMeans Cluster Falling into Each AI Category")

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.subplots_adjust(left=0.22)
    plt.savefig(CLUSTER_TO_CATEGORY_HEATMAP_FILE, dpi=300)
    plt.close()


# =============================================================================
# Main workflow
# =============================================================================

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    counts, category_to_cluster_shares, cluster_to_category_shares = (
        create_category_cluster_tables(data)
    )

    counts.to_csv(COUNTS_FILE, sep=CSV_SEPARATOR)
    category_to_cluster_shares.to_csv(CATEGORY_TO_CLUSTER_SHARES_FILE, sep=CSV_SEPARATOR)
    cluster_to_category_shares.to_csv(CLUSTER_TO_CATEGORY_SHARES_FILE, sep=CSV_SEPARATOR)

    plot_category_to_cluster_heatmap(category_to_cluster_shares)
    plot_cluster_to_category_heatmap(cluster_to_category_shares)

    print(f"Saved counts table to: {COUNTS_FILE}")
    print(f"Saved category-to-cluster share table to: {CATEGORY_TO_CLUSTER_SHARES_FILE}")
    print(f"Saved cluster-to-category share table to: {CLUSTER_TO_CATEGORY_SHARES_FILE}")
    print(f"Saved category-to-cluster heatmap to: {CATEGORY_TO_CLUSTER_HEATMAP_FILE}")
    print(f"Saved cluster-to-category heatmap to: {CLUSTER_TO_CATEGORY_HEATMAP_FILE}")


if __name__ == "__main__":
    main()
