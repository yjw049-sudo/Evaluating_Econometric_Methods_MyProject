from pathlib import Path
import math

import pandas as pd


# =============================================================================
# Parameters
# =============================================================================

PROJECT_DIR = Path(
    r"E:\study\S2\Evaluating-Econometric-Methods_data\Evaluating_Econometric_Methods_MyProject"
)

OUTPUT_DIR = PROJECT_DIR / "output" / "1920-1949-v1"
ANALYSIS_DIR = OUTPUT_DIR / "cluster_label_consistency"

INPUT_FILE = OUTPUT_DIR / "speeches_1920_1949_second_stage.csv"

CLUSTER_COLUMN = "kmeans_cluster"
AI_LABEL_COLUMN = "category"
YEAR_COLUMN = "year"

PRE_TEST_START_YEAR = 1920
PRE_TEST_END_YEAR = 1929
TRAIN_START_YEAR = 1930
TRAIN_END_YEAR = 1945
POST_TEST_START_YEAR = 1946
POST_TEST_END_YEAR = 1949

CSV_SEPARATOR = ";"

CROSSTAB_COUNTS_FILE = ANALYSIS_DIR / "cluster_ai_label_crosstab_counts.csv"
CROSSTAB_ROW_SHARES_FILE = ANALYSIS_DIR / "cluster_ai_label_crosstab_row_shares.csv"
CROSSTAB_COUNTS_TRAIN_FILE = ANALYSIS_DIR / "crosstab_counts_train.csv"
CROSSTAB_COUNTS_TEST_PRE_FILE = ANALYSIS_DIR / "crosstab_counts_test_pre.csv"
CROSSTAB_COUNTS_TEST_POST_FILE = ANALYSIS_DIR / "crosstab_counts_test_post.csv"
CROSSTAB_ROW_SHARES_TRAIN_FILE = ANALYSIS_DIR / "crosstab_row_shares_train.csv"
CROSSTAB_ROW_SHARES_TEST_PRE_FILE = ANALYSIS_DIR / "crosstab_row_shares_test_pre.csv"
CROSSTAB_ROW_SHARES_TEST_POST_FILE = ANALYSIS_DIR / "crosstab_row_shares_test_post.csv"
TRAIN_MAPPING_FILE = ANALYSIS_DIR / "train_cluster_dominant_label_mapping.csv"
CONSISTENCY_DATA_FILE = ANALYSIS_DIR / "texts_with_train_mapping_consistency.csv"
PERIOD_SUMMARY_FILE = ANALYSIS_DIR / "consistency_summary_by_period.csv"
CLUSTER_PERIOD_SUMMARY_FILE = ANALYSIS_DIR / "consistency_summary_by_period_and_cluster.csv"
LABEL_PERIOD_SUMMARY_FILE = ANALYSIS_DIR / "consistency_summary_by_period_and_ai_label.csv"

PERIOD_CROSSTAB_FILES = {
    "train": (CROSSTAB_COUNTS_TRAIN_FILE, CROSSTAB_ROW_SHARES_TRAIN_FILE),
    "test_pre": (CROSSTAB_COUNTS_TEST_PRE_FILE, CROSSTAB_ROW_SHARES_TEST_PRE_FILE),
    "test_post": (CROSSTAB_COUNTS_TEST_POST_FILE, CROSSTAB_ROW_SHARES_TEST_POST_FILE),
}


# =============================================================================
# Data preparation
# =============================================================================

def load_data():
    data = pd.read_csv(INPUT_FILE, sep=CSV_SEPARATOR)

    required_columns = [YEAR_COLUMN, CLUSTER_COLUMN, AI_LABEL_COLUMN]
    missing_columns = []

    for column in required_columns:
        if column not in data.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data.dropna(subset=required_columns).copy()
    data[YEAR_COLUMN] = data[YEAR_COLUMN].astype(int)
    data[CLUSTER_COLUMN] = data[CLUSTER_COLUMN].astype(int)

    return data


def assign_analysis_period(data):
    data = data.copy()
    data["analysis_period"] = "outside_analysis_period"

    pre_test_mask = data[YEAR_COLUMN].between(PRE_TEST_START_YEAR, PRE_TEST_END_YEAR)
    train_mask = data[YEAR_COLUMN].between(TRAIN_START_YEAR, TRAIN_END_YEAR)
    post_test_mask = data[YEAR_COLUMN].between(POST_TEST_START_YEAR, POST_TEST_END_YEAR)

    data.loc[pre_test_mask, "analysis_period"] = "test_pre"
    data.loc[train_mask, "analysis_period"] = "train"
    data.loc[post_test_mask, "analysis_period"] = "test_post"

    selected_data = data[data["analysis_period"] != "outside_analysis_period"].copy()

    return selected_data


# =============================================================================
# Cross-tabulation
# =============================================================================

def create_crosstab_counts(data):
    crosstab = pd.crosstab(
        index=[data["analysis_period"], data[CLUSTER_COLUMN]],
        columns=data[AI_LABEL_COLUMN],
    )

    crosstab = crosstab.reset_index()
    return crosstab


def create_crosstab_row_shares(crosstab_counts):
    label_columns = [
        column
        for column in crosstab_counts.columns
        if column not in ["analysis_period", CLUSTER_COLUMN]
    ]

    crosstab_shares = crosstab_counts.copy()
    row_totals = crosstab_shares[label_columns].sum(axis=1)

    for column in label_columns:
        crosstab_shares[column] = crosstab_shares[column] / row_totals

    return crosstab_shares


def create_single_period_crosstab(data, period):
    period_data = data[data["analysis_period"] == period].copy()

    crosstab = pd.crosstab(
        index=period_data[CLUSTER_COLUMN],
        columns=period_data[AI_LABEL_COLUMN],
    )

    crosstab = crosstab.reset_index()
    return crosstab


def calculate_entropy_from_counts(counts):
    total_count = counts.sum()

    if total_count == 0:
        return 0.0

    entropy = 0.0

    for count in counts:
        if count == 0:
            continue

        probability = count / total_count
        entropy -= probability * math.log(probability)

    return entropy


def calculate_period_cluster_entropy(data):
    entropy_rows = []

    grouped_data = data.groupby(["analysis_period", CLUSTER_COLUMN])

    for (period, cluster_id), group in grouped_data:
        label_counts = group[AI_LABEL_COLUMN].value_counts()
        entropy = calculate_entropy_from_counts(label_counts)

        entropy_rows.append(
            {
                "analysis_period": period,
                CLUSTER_COLUMN: cluster_id,
                "cluster_entropy": entropy,
            }
        )

    entropy_data = pd.DataFrame(entropy_rows)
    return entropy_data


# =============================================================================
# Train-period mapping
# =============================================================================

def create_train_mapping(data):
    train_data = data[data["analysis_period"] == "train"].copy()

    if train_data.empty:
        raise ValueError("The training period has no observations.")

    train_counts = (
        train_data.groupby([CLUSTER_COLUMN, AI_LABEL_COLUMN])
        .size()
        .reset_index(name="count")
    )

    cluster_totals = (
        train_counts.groupby(CLUSTER_COLUMN)["count"]
        .sum()
        .reset_index(name="cluster_total")
    )

    train_counts = train_counts.merge(cluster_totals, on=CLUSTER_COLUMN, how="left")
    train_counts["label_share_in_train_cluster"] = (
        train_counts["count"] / train_counts["cluster_total"]
    )

    train_counts = train_counts.sort_values(
        [CLUSTER_COLUMN, "count", AI_LABEL_COLUMN],
        ascending=[True, False, True],
    )

    mapping = train_counts.drop_duplicates(subset=[CLUSTER_COLUMN], keep="first").copy()
    mapping = mapping.rename(
        columns={
            AI_LABEL_COLUMN: "train_dominant_ai_label",
            "count": "train_dominant_label_count",
        }
    )

    mapping = mapping[
        [
            CLUSTER_COLUMN,
            "train_dominant_ai_label",
            "train_dominant_label_count",
            "cluster_total",
            "label_share_in_train_cluster",
        ]
    ]

    train_entropy = calculate_period_cluster_entropy(train_data)
    train_entropy = train_entropy.rename(columns={"cluster_entropy": "train_entropy"})
    train_entropy = train_entropy[[CLUSTER_COLUMN, "train_entropy"]]
    mapping = mapping.merge(train_entropy, on=CLUSTER_COLUMN, how="left")

    return mapping


# =============================================================================
# Consistency analysis
# =============================================================================

def apply_train_mapping(data, mapping):
    mapped_data = data.merge(mapping, on=CLUSTER_COLUMN, how="left", validate="many_to_one")

    mapped_data["has_train_mapping"] = mapped_data["train_dominant_ai_label"].notna()
    mapped_data["is_consistent_with_train_mapping"] = (
        mapped_data[AI_LABEL_COLUMN] == mapped_data["train_dominant_ai_label"]
    )

    mapped_data.loc[
        ~mapped_data["has_train_mapping"],
        "is_consistent_with_train_mapping",
    ] = False

    mapped_data["consistency_status"] = "inconsistent"
    mapped_data.loc[
        mapped_data["is_consistent_with_train_mapping"],
        "consistency_status",
    ] = "consistent"
    mapped_data.loc[
        ~mapped_data["has_train_mapping"],
        "consistency_status",
    ] = "no_train_mapping"

    return mapped_data


def summarize_consistency(data, group_columns):
    summary = (
        data.groupby(group_columns)
        .agg(
            speech_count=("basepk", "count"),
            consistent_count=("is_consistent_with_train_mapping", "sum"),
            no_train_mapping_count=("has_train_mapping", lambda values: (~values).sum()),
        )
        .reset_index()
    )

    summary["inconsistent_count"] = (
        summary["speech_count"]
        - summary["consistent_count"]
        - summary["no_train_mapping_count"]
    )
    summary["consistency_rate"] = summary["consistent_count"] / summary["speech_count"]
    summary["inconsistency_rate"] = summary["inconsistent_count"] / summary["speech_count"]

    return summary


# =============================================================================
# Main workflow
# =============================================================================

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    data = assign_analysis_period(data)

    crosstab_counts = create_crosstab_counts(data)
    crosstab_row_shares = create_crosstab_row_shares(crosstab_counts)
    period_cluster_entropy = calculate_period_cluster_entropy(data)
    train_mapping = create_train_mapping(data)
    mapped_data = apply_train_mapping(data, train_mapping)

    period_summary = summarize_consistency(mapped_data, ["analysis_period"])
    cluster_period_summary = summarize_consistency(
        mapped_data,
        ["analysis_period", CLUSTER_COLUMN],
    )
    cluster_period_summary = cluster_period_summary.merge(
        period_cluster_entropy,
        on=["analysis_period", CLUSTER_COLUMN],
        how="left",
        validate="one_to_one",
    )
    label_period_summary = summarize_consistency(
        mapped_data,
        ["analysis_period", AI_LABEL_COLUMN],
    )

    crosstab_counts.to_csv(CROSSTAB_COUNTS_FILE, sep=CSV_SEPARATOR, index=False)
    crosstab_row_shares.to_csv(CROSSTAB_ROW_SHARES_FILE, sep=CSV_SEPARATOR, index=False)

    for period, output_files in PERIOD_CROSSTAB_FILES.items():
        counts_file, row_shares_file = output_files
        period_crosstab_counts = create_single_period_crosstab(data, period)
        period_crosstab_row_shares = create_crosstab_row_shares(period_crosstab_counts)

        period_crosstab_counts.to_csv(counts_file, sep=CSV_SEPARATOR, index=False)
        period_crosstab_row_shares.to_csv(row_shares_file, sep=CSV_SEPARATOR, index=False)

    train_mapping.to_csv(TRAIN_MAPPING_FILE, sep=CSV_SEPARATOR, index=False)
    mapped_data.to_csv(CONSISTENCY_DATA_FILE, sep=CSV_SEPARATOR, index=False)
    period_summary.to_csv(PERIOD_SUMMARY_FILE, sep=CSV_SEPARATOR, index=False)
    cluster_period_summary.to_csv(
        CLUSTER_PERIOD_SUMMARY_FILE,
        sep=CSV_SEPARATOR,
        index=False,
    )
    label_period_summary.to_csv(
        LABEL_PERIOD_SUMMARY_FILE,
        sep=CSV_SEPARATOR,
        index=False,
    )

    print(f"Saved crosstab counts to: {CROSSTAB_COUNTS_FILE}")
    print(f"Saved crosstab row shares to: {CROSSTAB_ROW_SHARES_FILE}")
    print("Saved period-specific crosstab counts and row shares.")
    print(f"Saved train mapping to: {TRAIN_MAPPING_FILE}")
    print(f"Saved consistency data to: {CONSISTENCY_DATA_FILE}")
    print(f"Saved period summary to: {PERIOD_SUMMARY_FILE}")
    print(f"Saved cluster-period summary to: {CLUSTER_PERIOD_SUMMARY_FILE}")
    print(f"Saved label-period summary to: {LABEL_PERIOD_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
