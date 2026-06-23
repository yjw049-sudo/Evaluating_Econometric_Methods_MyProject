import pandas as pd

from cluster_type_analysis import (
    add_cluster_type,
    calculate_cluster_type_share_by_year,
    plot_cluster_type_trends,
)
from project_config import (
    CLUSTER_COLUMN,
    CSV_SEPARATOR,
    FULL_CORPUS_OUTPUT_DIR,
    YEAR_COLUMN,
)


INPUT_FILE = FULL_CORPUS_OUTPUT_DIR / "speeches_1920_1949_kmeans_ai_label_structure.csv"
FIGURE_DIR = FULL_CORPUS_OUTPUT_DIR / "figures"
TABLE_DIR = FULL_CORPUS_OUTPUT_DIR / "tables"

OUTPUT_TABLE_FILE = TABLE_DIR / "overall_cluster_type_share_by_year.csv"
OUTPUT_FIGURE_FILE = FIGURE_DIR / "overall_cluster_type_share_by_year.png"


def load_data():
    data = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        usecols=[YEAR_COLUMN, CLUSTER_COLUMN],
    )

    data = data.dropna(subset=[YEAR_COLUMN, CLUSTER_COLUMN]).copy()
    data[YEAR_COLUMN] = data[YEAR_COLUMN].astype(int)
    data[CLUSTER_COLUMN] = data[CLUSTER_COLUMN].astype(int)
    data = add_cluster_type(data, cluster_column=CLUSTER_COLUMN)

    return data


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    summary = calculate_cluster_type_share_by_year(data)

    summary.to_csv(OUTPUT_TABLE_FILE, sep=CSV_SEPARATOR, index=False)
    plot_cluster_type_trends(
        summary=summary,
        output_file=OUTPUT_FIGURE_FILE,
        title="Cluster Type Shares across All Speeches, 1920-1949",
        y_label="Share of all speeches (%)",
        count_label="Number of speeches",
    )

    print(f"Saved overall yearly cluster-type table to: {OUTPUT_TABLE_FILE}")
    print(f"Saved overall yearly cluster-type figure to: {OUTPUT_FIGURE_FILE}")
    print(f"Rows in all speeches: {len(data)}")


if __name__ == "__main__":
    main()
