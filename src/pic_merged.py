"""Generate merged diagnostic figures for the speech sample.

Input:
    output/Speech_sample.csv

Output:
    output/pic/merged/

Run:
    python pic_merged.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter


# =============================================================================
# User settings
# =============================================================================

HIGH_SCORE_THRESHOLD = 0.75
SCORE_LEVELS = [0, 0.25, 0.5, 0.75, 1.0]

# Sample files can be small. Increase this for the full data set if needed.
MIN_CATEGORY_YEAR_N = 5

TOP_N_CATEGORIES = 10
TOP_N_CLUSTERS = 20

REFERENCE_YEARS = [1968, 1982, 1986]

FIGURE_DPI = 300
FONT_SIZE = 16
MIN_FONT_SIZE = 12
SHOW_TITLES = False
SHOW_FIGURES = False

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 1,
})


# =============================================================================
# Paths and columns
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "output" / "Speech_sample.csv"
OUTPUT_DIR = PROJECT_DIR / "output" / "pic" / "merged"
TABLE_DIR = OUTPUT_DIR / "tables"

CSV_SEPARATOR = ","

BASEPK_COLUMN = "basepk"
YEAR_COLUMN = "year"
TOPIC_COLUMN = "topic"
CATEGORY_COLUMN = "category"
PROCEDURAL_SCORE_COLUMN = "procedural_score"
CLUSTER_SCORE_COLUMN = "cluster_score"
CLUSTER_COLUMN = "cluster"
WORD_COUNT_COLUMN = "speechtext_word_count"
TOPIC_ERROR_COLUMN = "topic_error"
CATEGORY_ERROR_COLUMN = "category_error"

REQUIRED_COLUMNS = [
    BASEPK_COLUMN,
    YEAR_COLUMN,
    TOPIC_COLUMN,
    CATEGORY_COLUMN,
    PROCEDURAL_SCORE_COLUMN,
    CLUSTER_SCORE_COLUMN,
    CLUSTER_COLUMN,
    WORD_COUNT_COLUMN,
    TOPIC_ERROR_COLUMN,
    CATEGORY_ERROR_COLUMN,
]


# =============================================================================
# Utility functions
# =============================================================================

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


def validate_columns(data: pd.DataFrame) -> None:
    """Check that all columns needed by the figures exist."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The input file is missing required columns: "
            f"{missing_columns}\n"
            f"Available columns: {list(data.columns)}"
        )


def safe_filename(text: str) -> str:
    """Convert arbitrary text into a compact Windows-safe filename."""
    safe_text = re.sub(r'[<>:"/\\|?*]+', "_", str(text).strip())
    safe_text = re.sub(r"\s+", "_", safe_text)
    safe_text = re.sub(r"_+", "_", safe_text).strip("_")
    return safe_text[:100] or "unknown"


def save_figure(figure: plt.Figure, filename: str) -> None:
    """Save and close one figure."""
    output_file = OUTPUT_DIR / filename
    enlarge_figure_fonts(figure)

    figure.savefig(
        output_file,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close(figure)
    print(f"Saved figure: {output_file}")


def enlarge_figure_fonts(figure: plt.Figure) -> None:
    """Ensure every text element is large enough for paper figures."""
    for text in figure.findobj(match=Text):
        text.set_fontsize(max(text.get_fontsize(), MIN_FONT_SIZE))


def save_table(data: pd.DataFrame, filename: str) -> None:
    """Save a CSV table with Excel-friendly encoding."""
    output_file = TABLE_DIR / filename
    data.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved table: {output_file}")


def add_reference_years(axis: plt.Axes) -> None:
    """Add vertical reference lines when the year is inside the plot range."""
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    for year in REFERENCE_YEARS:
        if xmin <= year <= xmax:
            axis.axvline(
                year,
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
            )
            axis.text(
                year + 0.08,
                ymax - (ymax - ymin) * 0.02,
                str(year),
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
            )


def round_to_nearest_quarter(values: pd.Series) -> pd.Series:
    """Round scores to 0, 0.25, 0.50, 0.75, or 1.00."""
    return (values * 4).round().div(4).clip(0, 1)


# =============================================================================
# Data loading
# =============================================================================

def load_data() -> pd.DataFrame:
    """Read and clean the columns used by the figure pipeline."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file was not found: {INPUT_FILE}")

    header = pd.read_csv(INPUT_FILE, nrows=0, sep=CSV_SEPARATOR)
    validate_columns(header)

    data = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        usecols=REQUIRED_COLUMNS,
        low_memory=False,
    )

    numeric_columns = [
        YEAR_COLUMN,
        PROCEDURAL_SCORE_COLUMN,
        CLUSTER_SCORE_COLUMN,
        CLUSTER_COLUMN,
        WORD_COUNT_COLUMN,
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in [TOPIC_COLUMN, CATEGORY_COLUMN]:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    data = data.dropna(
        subset=[
            YEAR_COLUMN,
            PROCEDURAL_SCORE_COLUMN,
            CLUSTER_SCORE_COLUMN,
            CLUSTER_COLUMN,
            WORD_COUNT_COLUMN,
        ]
    ).copy()


    # Remove rows whose topic_error or category_error is not empty.
    # Empty strings and whitespace-only values are treated as empty.
    rows_before_error_filter = len(data)

    topic_error_mask = (
        data[TOPIC_ERROR_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    category_error_mask = (
        data[CATEGORY_ERROR_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    removed_topic_error_rows = int(topic_error_mask.sum())
    removed_category_error_rows = int(category_error_mask.sum())
    combined_error_mask = topic_error_mask | category_error_mask
    removed_unique_error_rows = int(combined_error_mask.sum())

    data = data.loc[~combined_error_mask].copy()

    print("-" * 80)
    print("Error-row filtering")
    print(
        "Rows removed because topic_error is not empty: "
        f"{removed_topic_error_rows:,}"
    )
    print(
        "Rows removed because category_error is not empty: "
        f"{removed_category_error_rows:,}"
    )
    print(
        "Total unique rows removed: "
        f"{removed_unique_error_rows:,}"
    )
    print(
        "Rows remaining after error filtering: "
        f"{len(data):,} "
        f"(from {rows_before_error_filter:,})"
    )

    data = data[
        data[PROCEDURAL_SCORE_COLUMN].between(0, 1)
        & data[CLUSTER_SCORE_COLUMN].between(0, 1)
        & data[WORD_COUNT_COLUMN].ge(0)
        & data[CATEGORY_COLUMN].ne("")
    ].copy()

    data[YEAR_COLUMN] = data[YEAR_COLUMN].astype(int)
    data[CLUSTER_COLUMN] = data[CLUSTER_COLUMN].astype(int)

    if data.empty:
        raise ValueError("No valid rows remain after cleaning.")

    data["disagreement"] = (
        data[PROCEDURAL_SCORE_COLUMN]
        - data[CLUSTER_SCORE_COLUMN]
    )
    data["absolute_disagreement"] = data["disagreement"].abs()

    data["procedural_high"] = (
        data[PROCEDURAL_SCORE_COLUMN] >= HIGH_SCORE_THRESHOLD
    )
    data["cluster_high"] = (
        data[CLUSTER_SCORE_COLUMN] >= HIGH_SCORE_THRESHOLD
    )

    conditions = [
        data["procedural_high"] & data["cluster_high"],
        ~data["procedural_high"] & data["cluster_high"],
        data["procedural_high"] & ~data["cluster_high"],
    ]
    choices = [
        "Both high",
        "K-means only",
        "LLM only",
    ]

    data["quadrant"] = np.select(
        conditions,
        choices,
        default="Neither high",
    )

    year_mean_word_count = (
        data.groupby(YEAR_COLUMN)[WORD_COUNT_COLUMN]
        .transform("mean")
    )
    data["relative_word_count"] = (
        data[WORD_COUNT_COLUMN] / year_mean_word_count
    )

    return data


# =============================================================================
# 1. LLM category composition within each cluster
# =============================================================================

def plot_cluster_category_heatmap(data: pd.DataFrame) -> None:
    """Plot category composition within clusters ordered by cluster score."""
    cluster_scores = (
        data.groupby(CLUSTER_COLUMN)[CLUSTER_SCORE_COLUMN]
        .mean()
        .sort_values()
    )

    selected_clusters = (
        data[CLUSTER_COLUMN]
        .value_counts()
        .head(TOP_N_CLUSTERS)
        .index
    )

    cluster_scores = cluster_scores[
        cluster_scores.index.isin(selected_clusters)
    ].sort_values()

    selected_categories = (
        data[CATEGORY_COLUMN]
        .value_counts()
        .head(TOP_N_CATEGORIES)
        .index
        .tolist()
    )

    plot_data = data[
        data[CLUSTER_COLUMN].isin(cluster_scores.index)
        & data[CATEGORY_COLUMN].isin(selected_categories)
    ].copy()

    counts = pd.crosstab(
        plot_data[CATEGORY_COLUMN],
        plot_data[CLUSTER_COLUMN],
    )

    cluster_totals = (
        data[data[CLUSTER_COLUMN].isin(cluster_scores.index)]
        .groupby(CLUSTER_COLUMN)
        .size()
    )

    shares = counts.div(cluster_totals, axis=1).fillna(0) * 100
    shares = shares.reindex(
        index=selected_categories,
        columns=cluster_scores.index,
        fill_value=0,
    )

    figure_width = max(12, 0.75 * len(shares.columns))
    figure_height = max(6, 0.65 * len(shares.index))

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height)
    )

    values = shares.to_numpy()
    norm = PowerNorm(
        gamma=0.65,
        vmin=0,
        vmax=max(values.max(), 1),
    )

    image = axis.imshow(
        values,
        cmap="YlGnBu",
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )

    set_axis_title(axis, "LLM Label Composition within Each Cluster")
    axis.set_xlabel(
        "Cluster ordered by cluster_score\n"
        "(cluster number shown above cluster_score)"
    )
    axis.set_ylabel("LLM category")

    axis.set_xticks(range(len(shares.columns)))
    axis.set_xticklabels(
        [
            f"{cluster}\n{cluster_scores.loc[cluster]:.2f}"
            for cluster in shares.columns
        ],
        fontsize=8,
    )

    axis.set_yticks(range(len(shares.index)))
    axis.set_yticklabels(shares.index)

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]

            if value >= 10:
                text_color = (
                    "white"
                    if value > values.max() * 0.5
                    else "black"
                )
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Percentage")

    figure.tight_layout()

    table = (
        shares
        .reset_index()
        .rename(columns={CATEGORY_COLUMN: "category"})
    )
    save_table(table, "cluster_category_composition.csv")
    save_figure(figure, "cluster_category_heatmap.png")


# =============================================================================
# 2. Mean absolute disagreement over time
# =============================================================================

def plot_disagreement_over_time(data: pd.DataFrame) -> None:
    yearly = (
        data.groupby(YEAR_COLUMN, as_index=False)
        .agg(
            mean_abs_disagreement=("absolute_disagreement", "mean"),
            n=(BASEPK_COLUMN, "size"),
        )
        .sort_values(YEAR_COLUMN)
    )

    figure, axis = plt.subplots(figsize=(12, 7))

    axis.plot(
        yearly[YEAR_COLUMN],
        yearly["mean_abs_disagreement"],
        marker="o",
        linewidth=2,
        label="mean_abs_disagreement",
    )

    set_axis_title(axis, "Mean Absolute Disagreement over Time")
    axis.set_xlabel("Year")
    axis.set_ylabel("Mean absolute disagreement")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="lower center", frameon=False)

    add_reference_years(axis)
    figure.tight_layout()

    save_table(yearly, "disagreement_by_year.csv")
    save_figure(figure, "disagreement_by_year.png")


# =============================================================================
# 3. High cluster-score share by category
# =============================================================================

def plot_cluster_high_share_by_category(data: pd.DataFrame) -> None:
    summary = (
        data.groupby(CATEGORY_COLUMN, as_index=False)
        .agg(
            n=(BASEPK_COLUMN, "size"),
            high_share=("cluster_high", "mean"),
        )
        .sort_values("high_share", ascending=True)
    )

    figure, axis = plt.subplots(
        figsize=(12, max(6, 0.55 * len(summary)))
    )

    axis.barh(
        summary[CATEGORY_COLUMN],
        summary["high_share"],
    )

    for row_position, row in summary.reset_index(drop=True).iterrows():
        axis.text(
            row["high_share"] + 0.008,
            row_position,
            f"{row['high_share']:.1%}",
            va="center",
            fontsize=9,
        )

    set_axis_title(
        axis,
        f"Share of Speeches with cluster_score >= "
        f"{HIGH_SCORE_THRESHOLD} by Category"
    )
    axis.set_xlabel("Share")
    axis.set_ylabel("")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="x", alpha=0.3)

    max_share = summary["high_share"].max()
    axis.set_xlim(0, min(1.0, max_share + 0.12))

    figure.tight_layout()

    save_table(summary, "cluster_high_share_by_category.csv")
    save_figure(figure, "cluster_high_share_by_category.png")


# =============================================================================
# 4. Mean absolute disagreement by category
# =============================================================================

def plot_disagreement_by_category(data: pd.DataFrame) -> None:
    summary = (
        data.groupby(CATEGORY_COLUMN, as_index=False)
        .agg(
            n=(BASEPK_COLUMN, "size"),
            mean_abs_disagreement=("absolute_disagreement", "mean"),
        )
        .sort_values("mean_abs_disagreement", ascending=True)
    )

    figure, axis = plt.subplots(
        figsize=(12, max(6, 0.55 * len(summary)))
    )

    axis.barh(
        summary[CATEGORY_COLUMN],
        summary["mean_abs_disagreement"],
    )

    for row_position, row in summary.reset_index(drop=True).iterrows():
        axis.text(
            row["mean_abs_disagreement"] + 0.004,
            row_position,
            f"{row['mean_abs_disagreement']:.3f}",
            va="center",
            fontsize=9,
        )

    set_axis_title(axis, "Mean Absolute Disagreement by Category")
    axis.set_xlabel("Mean absolute disagreement")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.3)

    max_value = summary["mean_abs_disagreement"].max()
    axis.set_xlim(0, max_value * 1.15 if max_value > 0 else 1)

    figure.tight_layout()

    save_table(summary, "disagreement_by_category.csv")
    save_figure(figure, "disagreement_by_category.png")


# =============================================================================
# 5. Category-year high-score small multiples
# =============================================================================

def plot_category_high_score_small_multiples(data: pd.DataFrame) -> None:
    category_order = (
        data[CATEGORY_COLUMN]
        .value_counts()
        .head(TOP_N_CATEGORIES)
        .index
        .tolist()
    )

    plot_data = (
        data[data[CATEGORY_COLUMN].isin(category_order)]
        .groupby([CATEGORY_COLUMN, YEAR_COLUMN])
        .agg(
            cluster_high_share=("cluster_high", "mean"),
            procedural_high_share=("procedural_high", "mean"),
            n=(BASEPK_COLUMN, "size"),
        )
        .reset_index()
    )

    save_table(plot_data, "category_high_share_by_year.csv")

    number_of_columns = 2
    number_of_rows = math.ceil(len(category_order) / number_of_columns)

    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(14, 3.4 * number_of_rows),
        sharex=True,
    )

    axes = np.array(axes).reshape(-1)

    for axis_index, category in enumerate(category_order):
        axis = axes[axis_index]
        subset = plot_data[
            plot_data[CATEGORY_COLUMN].eq(category)
        ].sort_values(YEAR_COLUMN)

        high_n = subset["n"] >= MIN_CATEGORY_YEAR_N
        low_n = subset["n"] < MIN_CATEGORY_YEAR_N

        cluster_line, = axis.plot(
            subset[YEAR_COLUMN],
            subset["cluster_high_share"],
            linewidth=1.8,
            label=f"cluster_score >= {HIGH_SCORE_THRESHOLD}",
        )

        procedural_line, = axis.plot(
            subset[YEAR_COLUMN],
            subset["procedural_high_share"],
            linewidth=1.8,
            label=f"procedural_score >= {HIGH_SCORE_THRESHOLD}",
        )

        axis.scatter(
            subset.loc[high_n, YEAR_COLUMN],
            subset.loc[high_n, "cluster_high_share"],
            marker="o",
            s=18,
            color=cluster_line.get_color(),
            zorder=3,
        )
        axis.scatter(
            subset.loc[high_n, YEAR_COLUMN],
            subset.loc[high_n, "procedural_high_share"],
            marker="s",
            s=18,
            color=procedural_line.get_color(),
            zorder=3,
        )
        axis.scatter(
            subset.loc[low_n, YEAR_COLUMN],
            subset.loc[low_n, "cluster_high_share"],
            marker="o",
            s=32,
            facecolors="none",
            edgecolors=cluster_line.get_color(),
            linewidths=1.2,
            zorder=4,
        )
        axis.scatter(
            subset.loc[low_n, YEAR_COLUMN],
            subset.loc[low_n, "procedural_high_share"],
            marker="s",
            s=32,
            facecolors="none",
            edgecolors=procedural_line.get_color(),
            linewidths=1.2,
            zorder=4,
        )

        set_axis_title(axis, category, fontsize=12)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", alpha=0.3)

        add_reference_years(axis)

        axis.text(
            0.99,
            0.02,
            f"hollow: n < {MIN_CATEGORY_YEAR_N}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            alpha=0.65,
        )

    for axis_index in range(len(category_order), len(axes)):
        figure.delaxes(axes[axis_index])

    hollow_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=6,
        label=f"n < {MIN_CATEGORY_YEAR_N}: hollow marker",
    )

    handles, labels = axes[0].get_legend_handles_labels()

    set_figure_suptitle(
        figure,
        f"Top {len(category_order)} Categories: Yearly Share of "
        "High-Score Speeches\n"
        f"(cluster_score >= {HIGH_SCORE_THRESHOLD} vs "
        f"procedural_score >= {HIGH_SCORE_THRESHOLD})",
        fontsize=15,
        y=0.995,
    )

    figure.supxlabel("Year")
    figure.supylabel("Share")

    figure.legend(
        handles + [hollow_handle],
        labels + [hollow_handle.get_label()],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )

    figure.tight_layout(rect=[0.03, 0.03, 1, 0.92])
    save_figure(figure, "category_high_share_small_multiples.png")


# =============================================================================
# 6. Mean speech length by category
# =============================================================================

def plot_mean_word_count_by_category(data: pd.DataFrame) -> None:
    summary = (
        data.groupby(CATEGORY_COLUMN, as_index=False)
        .agg(
            n=(BASEPK_COLUMN, "size"),
            mean_word_count=(WORD_COUNT_COLUMN, "mean"),
        )
        .sort_values("mean_word_count", ascending=True)
    )

    figure, axis = plt.subplots(
        figsize=(12, max(6, 0.55 * len(summary)))
    )

    axis.barh(
        summary[CATEGORY_COLUMN],
        summary["mean_word_count"],
    )

    for row_position, row in summary.reset_index(drop=True).iterrows():
        axis.text(
            row["mean_word_count"] + 1,
            row_position,
            f"{row['mean_word_count']:.1f} (n={int(row['n']):,})",
            va="center",
            fontsize=9,
        )

    set_axis_title(axis, "Mean Speech Length by Category")
    axis.set_xlabel(f"Mean {WORD_COUNT_COLUMN}")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.3)

    max_value = summary["mean_word_count"].max()
    axis.set_xlim(0, max_value * 1.22 if max_value > 0 else 1)

    figure.tight_layout()

    save_table(summary, "mean_word_count_by_category.csv")
    save_figure(figure, "mean_word_count_by_category.png")


# =============================================================================
# 7. Quadrant composition within each category
# =============================================================================

def plot_quadrant_composition_by_category(data: pd.DataFrame) -> None:
    quadrant_order = [
        "Both high",
        "K-means only",
        "LLM only",
        "Neither high",
    ]

    counts = pd.crosstab(
        data[CATEGORY_COLUMN],
        data["quadrant"],
    )

    for quadrant in quadrant_order:
        if quadrant not in counts.columns:
            counts[quadrant] = 0

    shares = (
        counts[quadrant_order]
        .div(counts[quadrant_order].sum(axis=1), axis=0)
    )

    shares = shares.sort_values(
        "Both high",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(12, max(6, 0.6 * len(shares)))
    )

    left = pd.Series(0.0, index=shares.index)

    for quadrant in quadrant_order:
        axis.barh(
            shares.index,
            shares[quadrant],
            left=left,
            label=quadrant,
        )
        left = left + shares[quadrant]

    set_axis_title(
        axis,
        "Composition of LLM–K-means Quadrants within Each Category\n"
        f"threshold = {HIGH_SCORE_THRESHOLD}"
    )
    axis.set_xlabel("Share within category")
    axis.set_ylabel("")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="x", alpha=0.3)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
    )

    figure.tight_layout()

    save_table(
        shares.reset_index(),
        "quadrant_composition_by_category.csv",
    )
    save_figure(figure, "quadrant_composition_by_category.png")


# =============================================================================
# 8. Relative speech length by quadrant over time
# =============================================================================

def plot_relative_word_count_by_quadrant(data: pd.DataFrame) -> None:
    quadrant_order = [
        "Both high",
        "K-means only",
        "LLM only",
        "Neither high",
    ]

    yearly = (
        data.groupby([YEAR_COLUMN, "quadrant"])
        .agg(
            n=(BASEPK_COLUMN, "size"),
            mean_relative_word_count=("relative_word_count", "mean"),
        )
        .reset_index()
    )

    figure, axis = plt.subplots(figsize=(14, 7))

    axis.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label="Yearly average speech length = 1.0",
    )

    marker_map = {
        "Both high": "o",
        "K-means only": "^",
        "LLM only": "s",
        "Neither high": "D",
    }

    for quadrant in quadrant_order:
        subset = yearly[
            yearly["quadrant"].eq(quadrant)
        ].sort_values(YEAR_COLUMN)

        if subset.empty:
            continue

        line, = axis.plot(
            subset[YEAR_COLUMN],
            subset["mean_relative_word_count"],
            linewidth=2,
            label=quadrant,
        )

        high_n = subset["n"] >= MIN_CATEGORY_YEAR_N
        low_n = subset["n"] < MIN_CATEGORY_YEAR_N

        axis.scatter(
            subset.loc[high_n, YEAR_COLUMN],
            subset.loc[high_n, "mean_relative_word_count"],
            marker=marker_map[quadrant],
            s=38,
            color=line.get_color(),
            zorder=3,
        )

        axis.scatter(
            subset.loc[low_n, YEAR_COLUMN],
            subset.loc[low_n, "mean_relative_word_count"],
            marker=marker_map[quadrant],
            s=56,
            facecolors="none",
            edgecolors=line.get_color(),
            linewidths=1.3,
            zorder=4,
        )

    set_axis_title(
        axis,
        "Mean Relative Speech Length by LLM–K-means Quadrant over Time\n"
        f"(threshold = {HIGH_SCORE_THRESHOLD})"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel(
        "Mean relative speech length\n"
        "(word count / yearly mean word count)"
    )
    axis.grid(axis="y", alpha=0.3)

    add_reference_years(axis)

    axis.legend(
        loc="upper center",
        frameon=False,
        title=f"Hollow marker: n < {MIN_CATEGORY_YEAR_N}",
    )

    figure.tight_layout()

    save_table(yearly, "relative_word_count_by_quadrant_year.csv")
    save_figure(figure, "relative_word_count_by_quadrant_year.png")


# =============================================================================
# 9. Joint distribution heatmap
# =============================================================================

def plot_joint_score_distribution(data: pd.DataFrame) -> None:
    plot_data = data[
        [PROCEDURAL_SCORE_COLUMN, CLUSTER_SCORE_COLUMN]
    ].copy()

    plot_data["procedural_discrete"] = round_to_nearest_quarter(
        plot_data[PROCEDURAL_SCORE_COLUMN]
    )
    plot_data["cluster_discrete"] = round_to_nearest_quarter(
        plot_data[CLUSTER_SCORE_COLUMN]
    )

    cross_tab = pd.crosstab(
        plot_data["procedural_discrete"],
        plot_data["cluster_discrete"],
        normalize="index",
    )

    cross_tab = cross_tab.reindex(
        index=SCORE_LEVELS,
        columns=SCORE_LEVELS,
        fill_value=0,
    )

    figure, axis = plt.subplots(figsize=(8, 7))

    values = cross_tab.to_numpy()
    norm = PowerNorm(
        gamma=0.65,
        vmin=0,
        vmax=max(values.max(), 0.01),
    )

    image = axis.imshow(
        values,
        cmap="YlGnBu",
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )

    set_axis_title(
        axis,
        "Joint Distribution of Procedural Score and Cluster Score"
    )
    axis.set_xlabel("K-means cluster_score")
    axis.set_ylabel("LLM procedural_score")

    axis.set_xticks(range(len(SCORE_LEVELS)))
    axis.set_yticks(range(len(SCORE_LEVELS)))
    axis.set_xticklabels(SCORE_LEVELS)
    axis.set_yticklabels(SCORE_LEVELS)

    max_value = values.max()

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            text_color = (
                "white"
                if value > max_value * 0.45
                else "black"
            )
            axis.text(
                column_index,
                row_index,
                f"{value:.1%}",
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
            )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Share of speeches")

    figure.tight_layout()

    save_table(
        cross_tab.reset_index(),
        "joint_score_distribution.csv",
    )
    save_figure(figure, "joint_score_distribution.png")


# =============================================================================
# 10. Yearly share of speeches high in both measures
# =============================================================================

def plot_both_high_share_by_year(data: pd.DataFrame) -> None:
    yearly = (
        data.groupby(YEAR_COLUMN, as_index=False)
        .agg(
            n=(BASEPK_COLUMN, "size"),
            both_high_share=("quadrant", lambda values: (
                values.eq("Both high").mean()
            )),
        )
        .sort_values(YEAR_COLUMN)
    )

    figure, axis = plt.subplots(figsize=(12, 6))

    axis.plot(
        yearly[YEAR_COLUMN],
        yearly["both_high_share"],
        marker="o",
        linewidth=2,
    )

    set_axis_title(
        axis,
        "Yearly Share of Speeches with High Scores in Both Measures"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Share of speeches in year")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.3)

    add_reference_years(axis)
    figure.tight_layout()

    save_table(yearly, "both_high_share_by_year.csv")
    save_figure(figure, "both_high_share_by_year.png")


# =============================================================================
# 11. Topic summary tables
# =============================================================================

def save_topic_summaries(data: pd.DataFrame) -> None:
    """Export topic-level diagnostics without producing overcrowded figures."""
    topic_summary = (
        data[data[TOPIC_COLUMN].ne("")]
        .groupby(TOPIC_COLUMN, as_index=False)
        .agg(
            n=(BASEPK_COLUMN, "size"),
            mean_procedural_score=(PROCEDURAL_SCORE_COLUMN, "mean"),
            mean_cluster_score=(CLUSTER_SCORE_COLUMN, "mean"),
            mean_abs_disagreement=("absolute_disagreement", "mean"),
            cluster_high_share=("cluster_high", "mean"),
            procedural_high_share=("procedural_high", "mean"),
            mean_word_count=(WORD_COUNT_COLUMN, "mean"),
        )
        .sort_values("n", ascending=False)
    )

    save_table(topic_summary, "topic_summary.csv")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    print(f"Input file: {INPUT_FILE}")
    print(f"Rows used: {len(data):,}")
    print(f"Years: {data[YEAR_COLUMN].min()}-{data[YEAR_COLUMN].max()}")
    print(f"Categories: {data[CATEGORY_COLUMN].nunique()}")
    print(f"Clusters: {data[CLUSTER_COLUMN].nunique()}")

    plot_cluster_category_heatmap(data)
    save_topic_summaries(data)

    print("=" * 80)
    print("All figures and tables were generated successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
