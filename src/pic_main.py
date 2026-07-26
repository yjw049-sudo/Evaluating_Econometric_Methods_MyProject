from __future__ import annotations

"""
Main speech-length figures for the pre-chunking data.

Place this file at:
    src/pic_main.py

Expected input:
    output/Speech_sample.csv

Outputs:
    output/pic/main/

Run from the project root:
    python src/pic_main.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter


# =============================================================================
# User settings
# =============================================================================

HIGH_SCORE_THRESHOLD = 0.75
N_LENGTH_BINS = 10
TOP_N_CATEGORIES = 10
MIN_CATEGORY_BIN_N = 30
FIGURE_DPI = 300
FONT_SIZE = 16
MIN_FONT_SIZE = 12

# Set to False for paper-ready figures without embedded titles.
SHOW_TITLES = False

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 1,
})

# Public-facing terminology used in figures and exported tables.
LLM_SCORE_LABEL = "LLM score"
KMEANS_SCORE_LABEL = "K-means score"

# Difference is defined as:
#     procedural_score - cluster_score
# Positive values mean that the LLM gives the higher procedural score.
# Negative values mean that K-means gives the higher procedural score.
DIFFERENCE_COLUMN = "llm_minus_kmeans"


# =============================================================================
# Paths
# =============================================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

INPUT_FILE = PROJECT_ROOT / "output" / "Speech_sample.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pic" / "main"



OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Column names
# =============================================================================

WORD_COUNT_COLUMN = "speechtext_word_count"
LLM_SCORE_COLUMN = "procedural_score"
KMEANS_SCORE_COLUMN = "cluster_score"
CATEGORY_COLUMN = "category"
CLUSTER_COLUMN = "cluster"


# =============================================================================
# General helpers
# =============================================================================

def validate_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {missing}\n"
            f"Available columns are: {list(data.columns)}"
        )


def safe_filename(text: str) -> str:
    invalid_characters = '<>:"/\\|?*'
    result = str(text).strip()

    for character in invalid_characters:
        result = result.replace(character, "_")

    return result.replace(" ", "_")


def save_figure(figure: plt.Figure, filename: str) -> None:
    output_file = OUTPUT_DIR / filename
    enlarge_figure_fonts(figure)
    figure.savefig(
        output_file,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)
    print(f"Saved figure: {output_file}")


def enlarge_figure_fonts(figure: plt.Figure) -> None:
    """Ensure every text element is large enough for paper figures."""
    for text in figure.findobj(match=Text):
        text.set_fontsize(max(text.get_fontsize(), MIN_FONT_SIZE))


def save_table(data: pd.DataFrame, filename: str) -> None:
    output_file = OUTPUT_DIR / filename
    data.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved table: {output_file}")


def set_axis_title(
    axis: plt.Axes,
    title: str,
    **kwargs,
) -> None:
    """Add an axis title only when SHOW_TITLES is enabled."""
    if SHOW_TITLES:
        axis.set_title(title, **kwargs)


def set_figure_suptitle(
    figure: plt.Figure,
    title: str,
    **kwargs,
) -> None:
    """Add a figure-level title only when SHOW_TITLES is enabled."""
    if SHOW_TITLES:
        figure.suptitle(title, **kwargs)


def mean_ci_95(values: pd.Series) -> tuple[float, float, float, int]:
    """
    Return mean, lower 95% CI, upper 95% CI, and n.

    This uses the usual normal approximation:
        mean ± 1.96 * standard error
    """
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    n = len(clean_values)

    if n == 0:
        return np.nan, np.nan, np.nan, 0

    mean_value = clean_values.mean()

    if n == 1:
        return mean_value, np.nan, np.nan, 1

    standard_error = clean_values.std(ddof=1) / np.sqrt(n)
    margin = 1.96 * standard_error

    return (
        mean_value,
        mean_value - margin,
        mean_value + margin,
        n,
    )


def make_length_bins(data: pd.DataFrame) -> pd.DataFrame:
    """
    Divide speeches into approximately equal-sized word-count groups.

    Quantile bins are used because the speech-length distribution is strongly
    right-skewed. Each plotted point therefore has a broadly comparable sample
    size.

    The labels show the observed minimum and maximum word count in each group.
    """
    data = data.copy()

    # rank(method="first") avoids qcut failures when many speeches share
    # exactly the same word count.
    rank_values = data[WORD_COUNT_COLUMN].rank(method="first")

    data["length_bin_id"] = pd.qcut(
        rank_values,
        q=N_LENGTH_BINS,
        labels=False,
        duplicates="drop",
    )

    bin_ranges = (
        data.groupby("length_bin_id", observed=True)[WORD_COUNT_COLUMN]
        .agg(
            bin_min_words="min",
            bin_max_words="max",
            bin_mean_words="mean",
            bin_median_words="median",
            bin_n="size",
        )
        .reset_index()
        .sort_values("length_bin_id")
    )

    bin_ranges["length_bin_label"] = (
        bin_ranges["bin_min_words"].round().astype(int).astype(str)
        + "–"
        + bin_ranges["bin_max_words"].round().astype(int).astype(str)
    )

    data = data.merge(
        bin_ranges,
        on="length_bin_id",
        how="left",
        validate="many_to_one",
    )

    data["length_bin_label"] = pd.Categorical(
        data["length_bin_label"],
        categories=bin_ranges["length_bin_label"].tolist(),
        ordered=True,
    )

    save_table(bin_ranges, "table_00_length_bin_definitions.csv")
    return data


# =============================================================================
# Data loading and cleaning
# =============================================================================

def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file was not found: {INPUT_FILE}\n"
            "Place Speech_sample.csv in the project's output folder."
        )

    data = pd.read_csv(INPUT_FILE, low_memory=False)

    required_columns = [
        WORD_COUNT_COLUMN,
        LLM_SCORE_COLUMN,
        KMEANS_SCORE_COLUMN,
        CATEGORY_COLUMN,
        CLUSTER_COLUMN,
    ]
    validate_columns(data, required_columns)

    for column in [
        WORD_COUNT_COLUMN,
        LLM_SCORE_COLUMN,
        KMEANS_SCORE_COLUMN,
        CLUSTER_COLUMN,
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[CATEGORY_COLUMN] = (
        data[CATEGORY_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    original_n = len(data)

    data = data.dropna(
        subset=[
            WORD_COUNT_COLUMN,
            LLM_SCORE_COLUMN,
            KMEANS_SCORE_COLUMN,
            CLUSTER_COLUMN,
        ]
    ).copy()

    data = data[
        data[WORD_COUNT_COLUMN].gt(0)
        & data[LLM_SCORE_COLUMN].between(0, 1)
        & data[KMEANS_SCORE_COLUMN].between(0, 1)
        & data[CATEGORY_COLUMN].ne("")
    ].copy()

    data["llm_high"] = data[LLM_SCORE_COLUMN] >= HIGH_SCORE_THRESHOLD
    data["kmeans_high"] = data[KMEANS_SCORE_COLUMN] >= HIGH_SCORE_THRESHOLD

    data[DIFFERENCE_COLUMN] = (
        data[LLM_SCORE_COLUMN] - data[KMEANS_SCORE_COLUMN]
    )

    conditions = [
        data["llm_high"] & data["kmeans_high"],
        data["llm_high"] & ~data["kmeans_high"],
        ~data["llm_high"] & data["kmeans_high"],
    ]

    choices = [
        "Both high",
        "LLM only",
        "K-means only",
    ]

    data["high_score_group"] = np.select(
        conditions,
        choices,
        default="Neither high",
    )

    data = make_length_bins(data)

    print(f"Rows loaded: {original_n:,}")
    print(f"Rows retained after cleaning: {len(data):,}")
    print(
        f"Word-count range: "
        f"{data[WORD_COUNT_COLUMN].min():.0f}–"
        f"{data[WORD_COUNT_COLUMN].max():.0f}"
    )

    return data


# =============================================================================
# Figure 1: high-score probability by speech length
# =============================================================================

def make_high_probability_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for bin_label, group in data.groupby("length_bin_label", observed=True):
        llm_mean, llm_low, llm_high, llm_n = mean_ci_95(
            group["llm_high"].astype(float)
        )
        km_mean, km_low, km_high, km_n = mean_ci_95(
            group["kmeans_high"].astype(float)
        )

        rows.append(
            {
                "length_bin_label": str(bin_label),
                "mean_words": group[WORD_COUNT_COLUMN].mean(),
                "median_words": group[WORD_COUNT_COLUMN].median(),
                "n": len(group),
                "llm_score_high_probability": llm_mean,
                "llm_ci_low": max(0, llm_low),
                "llm_ci_high": min(1, llm_high),
                "kmeans_score_high_probability": km_mean,
                "kmeans_ci_low": max(0, km_low),
                "kmeans_ci_high": min(1, km_high),
            }
        )

    return pd.DataFrame(rows)


def plot_high_score_probability(data: pd.DataFrame) -> None:
    summary = make_high_probability_summary(data)
    save_table(summary, "table_01_high_score_probability_by_length.csv")

    x = np.arange(len(summary))

    figure, axis = plt.subplots(figsize=(13, 7))

    axis.plot(
        x,
        summary["llm_score_high_probability"],
        marker="o",
        linewidth=2,
        label=f"LLM score ≥ {HIGH_SCORE_THRESHOLD}",
    )

    axis.fill_between(
        x,
        summary["llm_ci_low"],
        summary["llm_ci_high"],
        alpha=0.15,
    )

    axis.plot(
        x,
        summary["kmeans_score_high_probability"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label=f"K-means score ≥ {HIGH_SCORE_THRESHOLD}",
    )

    axis.fill_between(
        x,
        summary["kmeans_ci_low"],
        summary["kmeans_ci_high"],
        alpha=0.15,
    )

    axis.set_xticks(x)
    axis.set_xticklabels(
        summary["length_bin_label"],
        rotation=35,
        ha="right",
    )

    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Original speech length group (words)")
    axis.set_ylabel("Probability of a high procedural score")
    set_axis_title(
        axis,
        "High-Score Probability by Original Speech Length\n"
        "Shaded bands are 95% confidence intervals",
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)

    figure.tight_layout()
    save_figure(figure, "fig_01_high_score_probability_by_length.png")


# =============================================================================
# Figure 2: mean scores by speech length
# =============================================================================

def make_mean_score_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for bin_label, group in data.groupby("length_bin_label", observed=True):
        llm_mean, llm_low, llm_high, _ = mean_ci_95(
            group[LLM_SCORE_COLUMN]
        )
        km_mean, km_low, km_high, _ = mean_ci_95(
            group[KMEANS_SCORE_COLUMN]
        )

        rows.append(
            {
                "length_bin_label": str(bin_label),
                "mean_words": group[WORD_COUNT_COLUMN].mean(),
                "n": len(group),
                "mean_llm_score": llm_mean,
                "llm_ci_low": llm_low,
                "llm_ci_high": llm_high,
                "mean_kmeans_score": km_mean,
                "kmeans_ci_low": km_low,
                "kmeans_ci_high": km_high,
            }
        )

    return pd.DataFrame(rows)


def plot_mean_scores(data: pd.DataFrame) -> None:
    summary = make_mean_score_summary(data)
    save_table(summary, "table_02_mean_scores_by_length.csv")

    x = np.arange(len(summary))

    figure, axis = plt.subplots(figsize=(13, 7))

    axis.plot(
        x,
        summary["mean_llm_score"],
        marker="o",
        linewidth=2,
        label="Mean LLM score",
    )
    axis.fill_between(
        x,
        summary["llm_ci_low"],
        summary["llm_ci_high"],
        alpha=0.15,
    )

    axis.plot(
        x,
        summary["mean_kmeans_score"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Mean K-means score",
    )
    axis.fill_between(
        x,
        summary["kmeans_ci_low"],
        summary["kmeans_ci_high"],
        alpha=0.15,
    )

    axis.set_xticks(x)
    axis.set_xticklabels(
        summary["length_bin_label"],
        rotation=35,
        ha="right",
    )

    axis.set_xlabel("Original speech length group (words)")
    axis.set_ylabel("Mean procedural score")
    set_axis_title(
        axis,
        "Mean LLM and K-means Scores by Original Speech Length",
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)

    figure.tight_layout()
    save_figure(figure, "fig_02_mean_scores_by_length.png")


# =============================================================================
# Figure 3: mean LLM-minus-K-means score difference
# =============================================================================

def make_difference_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for bin_label, group in data.groupby("length_bin_label", observed=True):
        mean_value, ci_low, ci_high, n = mean_ci_95(
            group[DIFFERENCE_COLUMN]
        )

        rows.append(
            {
                "length_bin_label": str(bin_label),
                "mean_words": group[WORD_COUNT_COLUMN].mean(),
                "median_words": group[WORD_COUNT_COLUMN].median(),
                "n": n,
                "mean_llm_minus_kmeans": mean_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return pd.DataFrame(rows)


def plot_score_difference(data: pd.DataFrame) -> None:
    summary = make_difference_summary(data)
    save_table(summary, "table_03_llm_minus_kmeans_by_length.csv")

    x = np.arange(len(summary))

    figure, axis = plt.subplots(figsize=(13, 7))

    axis.axhline(
        0,
        linewidth=1.3,
        linestyle="--",
        label="LLM score = K-means score",
    )

    axis.plot(
        x,
        summary["mean_llm_minus_kmeans"],
        marker="o",
        linewidth=2.2,
        label="Mean LLM score − K-means score",
    )

    axis.fill_between(
        x,
        summary["ci_low"],
        summary["ci_high"],
        alpha=0.18,
        label="95% confidence interval",
    )

    axis.set_xticks(x)
    axis.set_xticklabels(
        summary["length_bin_label"],
        rotation=35,
        ha="right",
    )

    axis.set_xlabel("Original speech length group (words)")
    axis.set_ylabel("Mean score difference: LLM score − K-means score")
    set_axis_title(
        axis,
        "Relative Scoring of LLM and K-means by Speech Length\n"
        "Above zero: LLM score is higher; below zero: K-means score is higher",
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)

    figure.tight_layout()
    save_figure(figure, "fig_03_llm_minus_kmeans_by_length.png")


# =============================================================================
# Figure 4: four high-score groups by speech length
# =============================================================================

def make_quadrant_summary(data: pd.DataFrame) -> pd.DataFrame:
    group_order = [
        "Both high",
        "LLM only",
        "K-means only",
        "Neither high",
    ]

    counts = (
        data.groupby(
            ["length_bin_label", "high_score_group"],
            observed=True,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    totals = (
        data.groupby("length_bin_label", observed=True)
        .size()
        .rename("bin_total")
        .reset_index()
    )

    summary = counts.merge(
        totals,
        on="length_bin_label",
        how="left",
        validate="many_to_one",
    )

    summary["share"] = summary["n"] / summary["bin_total"]

    complete_index = pd.MultiIndex.from_product(
        [
            data["length_bin_label"].cat.categories,
            group_order,
        ],
        names=["length_bin_label", "high_score_group"],
    )

    summary = (
        summary.set_index(["length_bin_label", "high_score_group"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    return summary


def plot_quadrant_shares(data: pd.DataFrame) -> None:
    group_order = [
        "Both high",
        "LLM only",
        "K-means only",
        "Neither high",
    ]

    summary = make_quadrant_summary(data)
    save_table(summary, "table_04_high_score_groups_by_length.csv")

    wide = summary.pivot(
        index="length_bin_label",
        columns="high_score_group",
        values="share",
    ).reindex(
        index=data["length_bin_label"].cat.categories,
        columns=group_order,
        fill_value=0,
    )

    # Keep the x-axis in the original numeric speech-length order.
    # Without this explicit reindexing, pivot() may sort labels
    # alphabetically, which places groups such as 114–139 before 50–59.
    wide.index = pd.CategoricalIndex(
        wide.index,
        categories=data["length_bin_label"].cat.categories,
        ordered=True,
        name="length_bin_label",
    )

    x = np.arange(len(wide))
    bottom = np.zeros(len(wide))

    figure, axis = plt.subplots(figsize=(14, 7))

    for group_name in group_order:
        values = wide[group_name].to_numpy()

        axis.bar(
            x,
            values,
            bottom=bottom,
            label=group_name,
        )

        bottom = bottom + values

    axis.set_xticks(x)
    axis.set_xticklabels(
        [str(value) for value in wide.index],
        rotation=35,
        ha="right",
    )

    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Original speech length group (words)")
    axis.set_ylabel("Share within speech-length group")
    set_axis_title(
        axis,
        "Composition of LLM–K-means High-Score Outcomes by Speech Length\n"
        f"High score is defined as score ≥ {HIGH_SCORE_THRESHOLD}",
    )
    axis.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    save_figure(figure, "fig_04_high_score_group_composition_by_length.png")


# =============================================================================
# Figure 5: score-difference heatmap
# =============================================================================

def plot_difference_heatmap(data: pd.DataFrame) -> None:
    # Preserve all score differences actually present in the data.
    difference_levels = sorted(
        data[DIFFERENCE_COLUMN].dropna().unique().tolist()
    )

    heatmap = pd.crosstab(
        data[DIFFERENCE_COLUMN],
        data["length_bin_label"],
        normalize="columns",
    )

    heatmap = heatmap.reindex(
        index=difference_levels,
        columns=data["length_bin_label"].cat.categories,
        fill_value=0,
    )

    heatmap_out = heatmap.reset_index()
    save_table(
        heatmap_out,
        "table_05_score_difference_distribution_by_length.csv",
    )

    figure_width = max(12, len(heatmap.columns) * 1.1)
    figure_height = max(6, len(heatmap.index) * 0.55)

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height)
    )

    image = axis.imshow(
        heatmap.to_numpy(),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    axis.set_xticks(np.arange(len(heatmap.columns)))
    axis.set_xticklabels(
        [str(value) for value in heatmap.columns],
        rotation=35,
        ha="right",
    )

    axis.set_yticks(np.arange(len(heatmap.index)))
    axis.set_yticklabels(
        [f"{value:+.2f}" for value in heatmap.index]
    )

    axis.set_xlabel("Original speech length group (words)")
    axis.set_ylabel("Score difference: LLM score − K-means score")
    set_axis_title(
        axis,
        "Distribution of LLM–K-means Score Differences by Speech Length\n"
        "Each column sums to 100%",
    )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Share within speech-length group")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # Add percentages when the heatmap is not too dense.
    if heatmap.shape[0] <= 20:
        maximum = heatmap.to_numpy().max()

        for row_index in range(heatmap.shape[0]):
            for column_index in range(heatmap.shape[1]):
                value = heatmap.iloc[row_index, column_index]

                if value <= 0:
                    continue

                text_color = "white" if value > maximum * 0.5 else "black"

                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

    figure.tight_layout()
    save_figure(figure, "fig_05_score_difference_heatmap_by_length.png")


# =============================================================================
# Figure 6: category composition across speech-length groups
# Diagnostic figure for possible category confounding
# =============================================================================

def plot_category_composition(data: pd.DataFrame) -> None:
    top_categories = (
        data[CATEGORY_COLUMN]
        .value_counts()
        .head(TOP_N_CATEGORIES)
        .index
        .tolist()
    )

    temp = data.copy()
    temp["category_plot"] = np.where(
        temp[CATEGORY_COLUMN].isin(top_categories),
        temp[CATEGORY_COLUMN],
        "Other categories",
    )

    category_order = top_categories + ["Other categories"]

    summary = (
        temp.groupby(
            ["length_bin_label", "category_plot"],
            observed=True,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    totals = (
        temp.groupby("length_bin_label", observed=True)
        .size()
        .rename("bin_total")
        .reset_index()
    )

    summary = summary.merge(
        totals,
        on="length_bin_label",
        how="left",
        validate="many_to_one",
    )
    summary["share"] = summary["n"] / summary["bin_total"]

    save_table(summary, "table_06_category_composition_by_length.csv")

    wide = summary.pivot(
        index="length_bin_label",
        columns="category_plot",
        values="share",
    ).reindex(columns=category_order, fill_value=0)

    x = np.arange(len(wide))
    bottom = np.zeros(len(wide))

    figure, axis = plt.subplots(figsize=(15, 8))

    for category in category_order:
        values = wide[category].fillna(0).to_numpy()

        axis.bar(
            x,
            values,
            bottom=bottom,
            label=category,
        )

        bottom += values

    axis.set_xticks(x)
    axis.set_xticklabels(
        [str(value) for value in wide.index],
        rotation=35,
        ha="right",
    )

    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Original speech length group (words)")
    axis.set_ylabel("Category share within length group")
    set_axis_title(
        axis,
        "Category Composition across Speech-Length Groups\n"
        "Descriptive diagnostic; not a decomposition of the length effect",
    )
    axis.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    figure.tight_layout()
    save_figure(figure, "fig_06_category_composition_by_length.png")


# =============================================================================
# Figure 7: within-category score-difference patterns
# =============================================================================

def plot_difference_within_categories(data: pd.DataFrame) -> None:
    top_categories = (
        data[CATEGORY_COLUMN]
        .value_counts()
        .head(TOP_N_CATEGORIES)
        .index
        .tolist()
    )

    rows = []

    for category in top_categories:
        category_data = data[data[CATEGORY_COLUMN].eq(category)]

        for bin_label, group in category_data.groupby(
            "length_bin_label",
            observed=True,
        ):
            mean_value, ci_low, ci_high, n = mean_ci_95(
                group[DIFFERENCE_COLUMN]
            )

            rows.append(
                {
                    CATEGORY_COLUMN: category,
                    "length_bin_label": str(bin_label),
                    "n": n,
                    "mean_llm_minus_kmeans": mean_value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    summary = pd.DataFrame(rows)

    # Small category-bin cells are unstable and are therefore not plotted.
    summary["keep_for_plot"] = summary["n"] >= MIN_CATEGORY_BIN_N

    save_table(
        summary,
        "table_07_llm_minus_kmeans_by_length_and_category.csv",
    )

    figure, axis = plt.subplots(figsize=(15, 8))

    bin_order = [
        str(value)
        for value in data["length_bin_label"].cat.categories
    ]
    x = np.arange(len(bin_order))

    axis.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    for category in top_categories:
        category_summary = summary[
            summary[CATEGORY_COLUMN].eq(category)
        ].copy()

        category_summary = (
            category_summary.set_index("length_bin_label")
            .reindex(bin_order)
            .reset_index()
        )

        values = category_summary["mean_llm_minus_kmeans"].where(
            category_summary["keep_for_plot"]
        )

        axis.plot(
            x,
            values,
            marker="o",
            linewidth=1.8,
            label=str(category),
        )

    axis.set_xticks(x)
    axis.set_xticklabels(
        bin_order,
        rotation=45,
        ha="right",
    )
    axis.set_xlabel("Original speech length group (words)")
    axis.set_ylabel(
        "Mean score difference\n"
        "positive = LLM higher; negative = K-means higher"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        title="Category",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    set_axis_title(
        axis,
        "LLM − K-means Score Difference by Speech Length within Major Categories\n"
        f"Category–length cells with n < {MIN_CATEGORY_BIN_N} are omitted",
        fontsize=15,
    )

    figure.tight_layout()
    save_figure(
        figure,
        "fig_07_llm_minus_kmeans_by_length_within_categories.png",
    )


# =============================================================================
# Optional cluster diagnostic table
# =============================================================================

def save_cluster_diagnostic(data: pd.DataFrame) -> None:
    """
    Export cluster composition by length.

    This is a diagnostic table rather than a main causal adjustment because
    cluster_score is assigned at the cluster level. Conditioning on cluster
    would mechanically condition on the source of the K-means score.
    """
    cluster_table = pd.crosstab(
        data["length_bin_label"],
        data[CLUSTER_COLUMN],
        normalize="index",
    )

    cluster_table = cluster_table.reset_index()
    save_table(
        cluster_table,
        "table_08_cluster_composition_by_length.csv",
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    data = load_data()

    plot_mean_scores(data)
    plot_score_difference(data)

    # Category is included as a diagnostic and stratification variable.
    plot_category_composition(data)
    plot_difference_within_categories(data)

    # Cluster is exported as a diagnostic table, not used as a control.
    save_cluster_diagnostic(data)

    print("\nAll outputs completed.")
    print(f"Input: {INPUT_FILE}")
    print(f"Descriptive output directory: {OUTPUT_DIR}")
    print(f"Figure titles enabled: {SHOW_TITLES}")


if __name__ == "__main__":
    main()
