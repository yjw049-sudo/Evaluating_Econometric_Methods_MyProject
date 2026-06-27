from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


# =============================================================================
# User settings
# =============================================================================

CLUSTERS_TO_PLOT = [18, 12, 9]
REFERENCE_YEARS = [1965, 1971, 1975, 1977, 1982]


# =============================================================================
# Paths and column settings
# =============================================================================

WORK_DIR = Path(__file__).resolve().parent
KMEANS_DIR = WORK_DIR / "output" / "Kmeans"

INPUT_FILE = KMEANS_DIR / "speeches_with_parlinfo_kmeans.csv"
OUTPUT_TABLE_FILE = KMEANS_DIR / "cluster_1_9_12_share_by_year.csv"

YEAR_COLUMN = "year"
CLUSTER_COLUMN = "cluster"
CSV_SEPARATOR = ","


# =============================================================================
# Data preparation
# =============================================================================

def load_year_and_cluster():
    """Load only the columns needed for the time-series figures."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file was not found: {INPUT_FILE}")

    data = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        usecols=[YEAR_COLUMN, CLUSTER_COLUMN],
        low_memory=True,
    )

    data = data.dropna(subset=[YEAR_COLUMN, CLUSTER_COLUMN]).copy()
    data[YEAR_COLUMN] = pd.to_numeric(
        data[YEAR_COLUMN],
        errors="coerce",
    )
    data[CLUSTER_COLUMN] = pd.to_numeric(
        data[CLUSTER_COLUMN],
        errors="coerce",
    )
    data = data.dropna(subset=[YEAR_COLUMN, CLUSTER_COLUMN]).copy()
    data[YEAR_COLUMN] = data[YEAR_COLUMN].astype(int)
    data[CLUSTER_COLUMN] = data[CLUSTER_COLUMN].astype(int)

    return data


def calculate_cluster_share_by_year(data):
    """
    Calculate each selected cluster's share of all speeches in each year.

    share = speeches in the cluster that year / all speeches that year
    """
    yearly_totals = (
        data.groupby(YEAR_COLUMN)
        .size()
        .rename("total_speeches")
        .reset_index()
    )

    selected_counts = (
        data[data[CLUSTER_COLUMN].isin(CLUSTERS_TO_PLOT)]
        .groupby([YEAR_COLUMN, CLUSTER_COLUMN])
        .size()
        .rename("cluster_speeches")
        .reset_index()
    )

    all_years = sorted(data[YEAR_COLUMN].unique())
    complete_index = pd.MultiIndex.from_product(
        [all_years, CLUSTERS_TO_PLOT],
        names=[YEAR_COLUMN, CLUSTER_COLUMN],
    )

    selected_counts = (
        selected_counts
        .set_index([YEAR_COLUMN, CLUSTER_COLUMN])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    summary = selected_counts.merge(
        yearly_totals,
        on=YEAR_COLUMN,
        how="left",
        validate="many_to_one",
    )
    summary["share"] = (
        summary["cluster_speeches"]
        / summary["total_speeches"]
    )

    return summary


# =============================================================================
# Plotting
# =============================================================================

def plot_one_cluster(summary, cluster_id):
    """Save one yearly share figure for a selected cluster."""
    cluster_data = summary[
        summary[CLUSTER_COLUMN].eq(cluster_id)
    ].copy()

    figure, axis = plt.subplots(figsize=(12, 6))

    axis.plot(
        cluster_data[YEAR_COLUMN],
        cluster_data["share"],
        color="#1f77b4",
        linewidth=2,
        marker="o",
        markersize=3.5,
    )

    axis.set_title(
        f"Cluster {cluster_id} Share of Parliamentary Speeches over Time"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Share of speeches")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axis.grid(True, alpha=0.3)
    axis.set_xlim(
        cluster_data[YEAR_COLUMN].min(),
        cluster_data[YEAR_COLUMN].max(),
    )

    for reference_year in REFERENCE_YEARS:
        axis.axvline(
            x=reference_year,
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
        )
        axis.text(
            reference_year,
            1.01,
            str(reference_year),
            transform=axis.get_xaxis_transform(),
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#a51f1f",
        )

    figure.tight_layout()

    output_file = KMEANS_DIR / f"cluster_{cluster_id}_share_over_time.png"
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return output_file


# =============================================================================
# Main workflow
# =============================================================================

def main():
    KMEANS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_year_and_cluster()

    available_clusters = set(data[CLUSTER_COLUMN].unique())
    missing_clusters = [
        cluster_id
        for cluster_id in CLUSTERS_TO_PLOT
        if cluster_id not in available_clusters
    ]

    if missing_clusters:
        raise ValueError(
            f"These clusters are not present in the input data: "
            f"{missing_clusters}"
        )

    summary = calculate_cluster_share_by_year(data)
    summary.to_csv(
        OUTPUT_TABLE_FILE,
        sep=CSV_SEPARATOR,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved yearly share data to: {OUTPUT_TABLE_FILE}")

    for cluster_id in CLUSTERS_TO_PLOT:
        output_file = plot_one_cluster(summary, cluster_id)
        print(f"Saved Cluster {cluster_id} figure to: {output_file}")


if __name__ == "__main__":
    main()
