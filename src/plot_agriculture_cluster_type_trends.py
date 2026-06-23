import re

import pandas as pd

from cluster_type_analysis import (
    add_cluster_type,
    calculate_cluster_type_share_by_period,
    calculate_cluster_type_share_by_year,
    plot_cluster_type_trends,
    sample_speeches_by_period,
)
from project_config import (
    AI_LABEL_COLUMN,
    BASEPK_COLUMN,
    CLUSTER_COLUMN,
    CSV_SEPARATOR,
    FULL_CORPUS_OUTPUT_DIR,
    ORIGINAL_TEXT_COLUMN,
    TEXT_COLUMN,
    YEAR_COLUMN,
)


INPUT_FILE = FULL_CORPUS_OUTPUT_DIR / "speeches_1920_1949_kmeans_ai_label_structure.csv"
FIGURE_DIR = FULL_CORPUS_OUTPUT_DIR / "figures"
TABLE_DIR = FULL_CORPUS_OUTPUT_DIR / "tables"

PERIOD_SAMPLE_SIZE = 5
RANDOM_STATE = 42

FILE_PREFIX_OVERRIDES = {
    "Agriculture and Food Policy": "agriculture",
    "Transportation and Communications": "transportation_communications",
    "Public Finance and Taxation": "public_finance_taxation",
}


def make_file_prefix(category_name):
    if category_name in FILE_PREFIX_OVERRIDES:
        return FILE_PREFIX_OVERRIDES[category_name]

    file_prefix = category_name.lower()
    file_prefix = re.sub(r"[^a-z0-9]+", "_", file_prefix)
    file_prefix = file_prefix.strip("_")

    return file_prefix


def make_output_paths(file_prefix):
    return {
        "year_table": TABLE_DIR / f"{file_prefix}_cluster_type_share_by_year.csv",
        "period_table": TABLE_DIR / f"{file_prefix}_cluster_type_share_by_period.csv",
        "period_sample": TABLE_DIR / f"{file_prefix}_period_sample_5_each.csv",
        "figure": FIGURE_DIR / f"{file_prefix}_cluster_type_share_by_year.png",
    }


def load_all_data():
    data = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        usecols=[
            BASEPK_COLUMN,
            YEAR_COLUMN,
            "topic",
            AI_LABEL_COLUMN,
            TEXT_COLUMN,
            ORIGINAL_TEXT_COLUMN,
            CLUSTER_COLUMN,
        ],
    )

    data[YEAR_COLUMN] = data[YEAR_COLUMN].astype(int)
    data[CLUSTER_COLUMN] = data[CLUSTER_COLUMN].astype(int)
    data = add_cluster_type(data, cluster_column=CLUSTER_COLUMN)

    return data


def save_category_outputs(all_data, target_category):
    file_prefix = make_file_prefix(target_category)
    plot_title = target_category
    output_paths = make_output_paths(file_prefix)

    data = all_data[all_data[AI_LABEL_COLUMN] == target_category].copy()

    if data.empty:
        print(f"Warning: no rows found for category: {target_category}")
        return

    yearly_summary = calculate_cluster_type_share_by_year(data)
    period_summary = calculate_cluster_type_share_by_period(data)
    period_sample = sample_speeches_by_period(
        data,
        sample_size=PERIOD_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    yearly_summary.to_csv(output_paths["year_table"], sep=CSV_SEPARATOR, index=False)
    period_summary.to_csv(output_paths["period_table"], sep=CSV_SEPARATOR, index=False)
    period_sample.to_csv(output_paths["period_sample"], sep=CSV_SEPARATOR, index=False)

    plot_cluster_type_trends(
        summary=yearly_summary,
        output_file=output_paths["figure"],
        title=f"Cluster Type Shares within {plot_title}, 1920-1949",
        y_label=f"Share within {plot_title} (%)",
        count_label=f"Number of {plot_title} speeches",
    )

    print(f"Category: {target_category}")
    print(f"Saved yearly cluster-type table to: {output_paths['year_table']}")
    print(f"Saved period cluster-type table to: {output_paths['period_table']}")
    print(f"Saved period sample to: {output_paths['period_sample']}")
    print(f"Saved yearly cluster-type figure to: {output_paths['figure']}")
    print(f"Rows in category: {len(data)}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    all_data = load_all_data()
    target_categories = sorted(all_data[AI_LABEL_COLUMN].dropna().unique())

    for target_category in target_categories:
        save_category_outputs(all_data, target_category)


if __name__ == "__main__":
    main()
