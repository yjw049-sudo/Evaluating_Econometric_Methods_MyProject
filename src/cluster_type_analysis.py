import matplotlib.pyplot as plt
import pandas as pd

from project_config import BASEPK_COLUMN, CLUSTER_COLUMN, YEAR_COLUMN


CLUSTER_TYPE_MAPPING = {
    0: "substantive_policy_discourse",
    1: "substantive_policy_discourse",
    2: "substantive_policy_discourse",
    3: "general_parliamentary_discourse",
    4: "substantive_policy_discourse",
    5: "institutional_procedural_discourse",
    6: "substantive_policy_discourse",
    7: "institutional_procedural_discourse",
    8: "institutional_procedural_discourse",
    9: "substantive_policy_discourse",
    10: "substantive_policy_discourse",
    11: "substantive_policy_discourse",
    12: "general_parliamentary_discourse",
    13: "substantive_policy_discourse",
    14: "substantive_policy_discourse",
    15: "general_parliamentary_discourse",
    16: "substantive_policy_discourse",
    17: "substantive_policy_discourse",
    18: "institutional_procedural_discourse",
    19: "substantive_policy_discourse",
}

CLUSTER_TYPE_ORDER = [
    "substantive_policy_discourse",
    "general_parliamentary_discourse",
    "institutional_procedural_discourse",
]

PERIOD_DEFINITIONS = [
    ("postwar_1920s", 1920, 1929),
    ("depression_1930s", 1930, 1938),
    ("wartime", 1939, 1945),
    ("postwar_reconstruction", 1946, 1949),
]

PERIOD_BACKGROUNDS = [
    (1920, 1929, "#d9ead3"),
    (1930, 1938, "#fff2cc"),
    (1939, 1945, "#f4cccc"),
    (1946, 1949, "#cfe2f3"),
]


def add_cluster_type(data, cluster_column=CLUSTER_COLUMN):
    data = data.copy()
    data["cluster_type"] = data[cluster_column].map(CLUSTER_TYPE_MAPPING)

    missing_count = data["cluster_type"].isna().sum()
    if missing_count > 0:
        raise ValueError(f"{missing_count} rows have unmapped kmeans_cluster values.")

    return data


def calculate_cluster_type_share_by_year(data):
    counts = (
        data.groupby([YEAR_COLUMN, "cluster_type"])
        .size()
        .reset_index(name="count")
    )
    year_totals = data.groupby(YEAR_COLUMN).size().reset_index(name="year_total")

    summary = counts.merge(year_totals, on=YEAR_COLUMN, how="left")
    summary["share"] = summary["count"] / summary["year_total"]
    summary["share_percent"] = summary["share"] * 100

    all_years = sorted(data[YEAR_COLUMN].unique())
    full_index = pd.MultiIndex.from_product(
        [all_years, CLUSTER_TYPE_ORDER],
        names=[YEAR_COLUMN, "cluster_type"],
    )

    summary = (
        summary.set_index([YEAR_COLUMN, "cluster_type"])
        .reindex(full_index)
        .reset_index()
    )
    summary["count"] = summary["count"].fillna(0).astype(int)
    summary["year_total"] = summary[YEAR_COLUMN].map(
        year_totals.set_index(YEAR_COLUMN)["year_total"]
    )
    summary["share"] = summary["share"].fillna(0)
    summary["share_percent"] = summary["share_percent"].fillna(0)

    return summary


def add_historical_period(data):
    data = data.copy()
    data["historical_period"] = pd.NA

    for period_name, start_year, end_year in PERIOD_DEFINITIONS:
        period_mask = data[YEAR_COLUMN].between(start_year, end_year)
        data.loc[period_mask, "historical_period"] = period_name

    return data.dropna(subset=["historical_period"]).copy()


def calculate_cluster_type_share_by_period(data):
    data = add_historical_period(data)

    counts = (
        data.groupby(["historical_period", "cluster_type"])
        .size()
        .reset_index(name="count")
    )
    period_totals = (
        data.groupby("historical_period")
        .size()
        .reset_index(name="speech_count")
    )

    summary = counts.merge(period_totals, on="historical_period", how="left")
    summary["share_percent"] = summary["count"] / summary["speech_count"] * 100

    wide_summary = summary.pivot(
        index="historical_period",
        columns="cluster_type",
        values="share_percent",
    )
    wide_summary = wide_summary.reindex([period[0] for period in PERIOD_DEFINITIONS])
    wide_summary = wide_summary.reindex(columns=CLUSTER_TYPE_ORDER).fillna(0)

    period_counts = period_totals.set_index("historical_period")["speech_count"]
    wide_summary.insert(0, "speech_count", period_counts)

    wide_summary = wide_summary.rename(
        columns={
            "substantive_policy_discourse": "substantive %",
            "general_parliamentary_discourse": "general %",
            "institutional_procedural_discourse": "institutional %",
        }
    )
    wide_summary = wide_summary.reset_index()
    wide_summary = wide_summary.rename(columns={"historical_period": "period"})

    return wide_summary


def sample_speeches_by_period(data, sample_size, random_state):
    data = add_historical_period(data)
    sample_frames = []

    for period_name, _, _ in PERIOD_DEFINITIONS:
        period_data = data[data["historical_period"] == period_name].copy()

        if len(period_data) < sample_size:
            print(f"Warning: {period_name} has only {len(period_data)} rows.")
            sampled_period_data = period_data.copy()
        else:
            sampled_period_data = period_data.sample(
                n=sample_size,
                random_state=random_state,
            )

        sample_frames.append(sampled_period_data)

    sampled_data = pd.concat(sample_frames, ignore_index=True)
    sampled_data = sampled_data.sort_values(["historical_period", BASEPK_COLUMN])
    sampled_data = sampled_data.reset_index(drop=True)

    return sampled_data


def plot_cluster_type_trends(summary, output_file, title, y_label, count_label):
    plot_data = summary.pivot(
        index=YEAR_COLUMN,
        columns="cluster_type",
        values="share_percent",
    )
    plot_data = plot_data[CLUSTER_TYPE_ORDER]

    yearly_speech_counts = (
        summary[[YEAR_COLUMN, "year_total"]]
        .drop_duplicates()
        .set_index(YEAR_COLUMN)
        .sort_index()["year_total"]
    )

    figure, axis = plt.subplots(figsize=(14, 8))

    for start_year, end_year, color in PERIOD_BACKGROUNDS:
        axis.axvspan(start_year - 0.5, end_year + 0.5, color=color, alpha=0.35)

    for cluster_type in CLUSTER_TYPE_ORDER:
        axis.plot(
            plot_data.index,
            plot_data[cluster_type],
            marker="o",
            linewidth=2,
            markersize=4,
            label=cluster_type,
        )

    axis.set_xlabel("Year")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.set_xlim(plot_data.index.min(), plot_data.index.max())
    axis.grid(axis="y", alpha=0.3)

    count_axis = axis.twinx()
    count_axis.plot(
        yearly_speech_counts.index,
        yearly_speech_counts.values,
        color="#777777",
        linestyle="--",
        marker="s",
        linewidth=1.8,
        markersize=4,
        label="speech_count",
    )
    count_axis.set_ylabel(count_label)

    count_min = yearly_speech_counts.min()
    count_max = yearly_speech_counts.max()
    count_range = count_max - count_min
    count_axis.set_ylim(
        max(0, count_min - count_range * 1.2),
        count_max + count_range * 1.2,
    )

    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)
    count_axis.spines["top"].set_visible(False)

    lines, labels = axis.get_legend_handles_labels()
    count_lines, count_labels = count_axis.get_legend_handles_labels()
    axis.legend(
        lines + count_lines,
        labels + count_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
    )

    figure.tight_layout()
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)
