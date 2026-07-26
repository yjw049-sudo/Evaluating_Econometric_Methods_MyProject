from __future__ import annotations

"""
Check how chunking changes LLM and K-means scores.

Place this file at:
    src/pic_chunk_check.py

Expected input:
    output/Speech_final.csv

Outputs:
    output/pic/chunk_check/

Run from the project root:
    python src/pic_chunk_check.py
"""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
from statsmodels.nonparametric.smoothers_lowess import lowess


# =============================================================================
# User settings
# =============================================================================

# False is recommended for paper-ready figures.
SHOW_TITLES = False

FIGURE_DPI = 300
FONT_SIZE = 16
MIN_FONT_SIZE = 12

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 1,
})

# Main analysis keeps speeches that were actually split into at least two chunks.
MIN_CHUNK_TOTAL = 2

# Equal-frequency bins based on original speech length.
N_LENGTH_BINS = 10

# High-score definition.
HIGH_SCORE_THRESHOLD = 0.75

# Category diagnostics.
TOP_N_CATEGORIES = 10
MIN_CATEGORY_BIN_N = 30

# LOWESS settings for the continuous length analysis.
LOWESS_FRAC = 0.30

# Bootstrap settings for LOWESS confidence intervals.
LOWESS_BOOTSTRAP_REPS = 10
LOWESS_BOOTSTRAP_GRID_SIZE = 120
LOWESS_RANDOM_SEED = 2026

# Continuous LOWESS plots use the full cleaned sample range.

# Optional chunk-count groups used in one diagnostic figure.
CHUNK_TOTAL_GROUPS = [
    (2, 2, "2"),
    (3, 4, "3–4"),
    (5, 7, "5–7"),
    (8, 10, "8–10"),
    (11, np.inf, "11+"),
]


# =============================================================================
# Paths
# =============================================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

INPUT_FILE = PROJECT_ROOT / "output" / "Speech_final.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pic" / "chunk_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Column names
# =============================================================================

ID_COLUMN = "basepk"
WORD_COUNT_COLUMN = "speechtext_word_count"
CATEGORY_COLUMN = "category"

ORIGINAL_LLM_COLUMN = "procedural_score"
ORIGINAL_KMEANS_COLUMN = "cluster_score"

CHUNK_LLM_COLUMN = "chunk_procedural_score"
CHUNK_KMEANS_COLUMN = "chunk_cluster_score"

# merge_final.py creates this column.
CHUNK_ROW_COUNT_COLUMN = "chunk_row_count"

# If Speech_final.csv already contains chunk_total, it is preferred.
CHUNK_TOTAL_COLUMN = "chunk_total"


# =============================================================================
# Helpers
# =============================================================================

def set_axis_title(
    axis: plt.Axes,
    title: str,
    **kwargs,
) -> None:
    if SHOW_TITLES:
        axis.set_title(title, **kwargs)


def set_figure_suptitle(
    figure: plt.Figure,
    title: str,
    **kwargs,
) -> None:
    if SHOW_TITLES:
        figure.suptitle(title, **kwargs)


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
    data.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved table: {output_file}")


def validate_columns(
    data: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Speech_final.csv is missing required columns: {missing}\n"
            f"Available columns are: {list(data.columns)}"
        )


def mean_ci_95(
    values: pd.Series,
) -> tuple[float, float, float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = len(clean)

    if n == 0:
        return np.nan, np.nan, np.nan, 0

    mean_value = clean.mean()

    if n == 1:
        return mean_value, np.nan, np.nan, 1

    standard_error = clean.std(ddof=1) / np.sqrt(n)
    margin = 1.96 * standard_error

    return (
        mean_value,
        mean_value - margin,
        mean_value + margin,
        n,
    )


def choose_chunk_count_column(data: pd.DataFrame) -> str:
    if CHUNK_TOTAL_COLUMN in data.columns:
        return CHUNK_TOTAL_COLUMN

    if CHUNK_ROW_COUNT_COLUMN in data.columns:
        return CHUNK_ROW_COUNT_COLUMN

    raise ValueError(
        "Speech_final.csv needs either 'chunk_total' or "
        "'chunk_row_count' to identify how many chunks each speech has."
    )


def make_length_bins(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    ranks = data[WORD_COUNT_COLUMN].rank(method="first")

    data["length_bin_id"] = pd.qcut(
        ranks,
        q=N_LENGTH_BINS,
        labels=False,
        duplicates="drop",
    )

    definitions = (
        data.groupby("length_bin_id", observed=True)[WORD_COUNT_COLUMN]
        .agg(
            min_words="min",
            max_words="max",
            mean_words="mean",
            median_words="median",
            n="size",
        )
        .reset_index()
        .sort_values("length_bin_id")
    )

    definitions["length_bin_label"] = (
        definitions["min_words"].round().astype(int).astype(str)
        + "–"
        + definitions["max_words"].round().astype(int).astype(str)
    )

    data = data.merge(
        definitions[
            [
                "length_bin_id",
                "length_bin_label",
            ]
        ],
        on="length_bin_id",
        how="left",
        validate="many_to_one",
    )

    ordered_labels = definitions["length_bin_label"].tolist()

    data["length_bin_label"] = pd.Categorical(
        data["length_bin_label"],
        categories=ordered_labels,
        ordered=True,
    )

    save_table(
        definitions,
        "table_00_length_bin_definitions.csv",
    )

    return data


def assign_chunk_total_group(
    values: pd.Series,
) -> pd.Categorical:
    labels = []

    for value in values:
        assigned = None

        for lower, upper, label in CHUNK_TOTAL_GROUPS:
            if lower <= value <= upper:
                assigned = label
                break

        labels.append(assigned)

    ordered_labels = [
        label for _, _, label in CHUNK_TOTAL_GROUPS
    ]

    return pd.Categorical(
        labels,
        categories=ordered_labels,
        ordered=True,
    )


# =============================================================================
# Data loading
# =============================================================================

def load_data() -> tuple[pd.DataFrame, str]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file was not found: {INPUT_FILE}\n"
            "Run src/merge_final.py first."
        )

    data = pd.read_csv(INPUT_FILE, low_memory=False)

    required_columns = [
        ID_COLUMN,
        WORD_COUNT_COLUMN,
        ORIGINAL_LLM_COLUMN,
        ORIGINAL_KMEANS_COLUMN,
        CHUNK_LLM_COLUMN,
        CHUNK_KMEANS_COLUMN,
    ]

    validate_columns(data, required_columns)

    chunk_count_column = choose_chunk_count_column(data)

    numeric_columns = [
        WORD_COUNT_COLUMN,
        ORIGINAL_LLM_COLUMN,
        ORIGINAL_KMEANS_COLUMN,
        CHUNK_LLM_COLUMN,
        CHUNK_KMEANS_COLUMN,
        chunk_count_column,
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    original_n = len(data)

    data = data.dropna(
        subset=numeric_columns
    ).copy()

    data = data[
        data[WORD_COUNT_COLUMN].gt(0)
        & data[ORIGINAL_LLM_COLUMN].between(0, 1)
        & data[ORIGINAL_KMEANS_COLUMN].between(0, 1)
        & data[CHUNK_LLM_COLUMN].between(0, 1)
        & data[CHUNK_KMEANS_COLUMN].between(0, 1)
        & data[chunk_count_column].ge(MIN_CHUNK_TOTAL)
    ].copy()

    data["analysis_chunk_total"] = (
        data[chunk_count_column].round().astype(int)
    )

    # Main paired changes.
    data["delta_llm"] = (
        data[CHUNK_LLM_COLUMN]
        - data[ORIGINAL_LLM_COLUMN]
    )
    data["delta_kmeans"] = (
        data[CHUNK_KMEANS_COLUMN]
        - data[ORIGINAL_KMEANS_COLUMN]
    )

    data["original_gap_llm_minus_kmeans"] = (
        data[ORIGINAL_LLM_COLUMN]
        - data[ORIGINAL_KMEANS_COLUMN]
    )
    data["chunk_gap_llm_minus_kmeans"] = (
        data[CHUNK_LLM_COLUMN]
        - data[CHUNK_KMEANS_COLUMN]
    )
    data["delta_gap"] = (
        data["chunk_gap_llm_minus_kmeans"]
        - data["original_gap_llm_minus_kmeans"]
    )

    # Absolute score changes measure sensitivity to chunking regardless
    # of whether the score moves upward or downward.
    data["abs_delta_llm"] = data["delta_llm"].abs()
    data["abs_delta_kmeans"] = data["delta_kmeans"].abs()

    # High-score indicators.
    data["original_llm_high"] = (
        data[ORIGINAL_LLM_COLUMN] >= HIGH_SCORE_THRESHOLD
    )
    data["chunk_llm_high"] = (
        data[CHUNK_LLM_COLUMN] >= HIGH_SCORE_THRESHOLD
    )
    data["original_kmeans_high"] = (
        data[ORIGINAL_KMEANS_COLUMN] >= HIGH_SCORE_THRESHOLD
    )
    data["chunk_kmeans_high"] = (
        data[CHUNK_KMEANS_COLUMN] >= HIGH_SCORE_THRESHOLD
    )

    data["chunk_total_group"] = assign_chunk_total_group(
        data["analysis_chunk_total"]
    )

    data = make_length_bins(data)

    print(f"Rows loaded: {original_n:,}")
    print(
        f"Rows retained with chunk_total >= {MIN_CHUNK_TOTAL}: "
        f"{len(data):,}"
    )
    print(
        f"Chunk count column used: {chunk_count_column}"
    )
    print(
        f"Original word-count range: "
        f"{data[WORD_COUNT_COLUMN].min():.0f}–"
        f"{data[WORD_COUNT_COLUMN].max():.0f}"
    )

    return data, chunk_count_column


# =============================================================================
# Figure 1: LLM original vs chunk-aggregated score
# =============================================================================

def make_method_summary(
    data: pd.DataFrame,
    original_column: str,
    chunk_column: str,
) -> pd.DataFrame:
    rows = []

    for bin_label, group in data.groupby(
        "length_bin_label",
        observed=True,
    ):
        original_mean, original_low, original_high, n = mean_ci_95(
            group[original_column]
        )
        chunk_mean, chunk_low, chunk_high, _ = mean_ci_95(
            group[chunk_column]
        )

        rows.append(
            {
                "length_bin_label": str(bin_label),
                "n": n,
                "original_mean": original_mean,
                "original_ci_low": original_low,
                "original_ci_high": original_high,
                "chunk_mean": chunk_mean,
                "chunk_ci_low": chunk_low,
                "chunk_ci_high": chunk_high,
            }
        )

    return pd.DataFrame(rows)


def plot_before_after_method(
    data: pd.DataFrame,
    original_column: str,
    chunk_column: str,
    public_label: str,
    filename: str,
    table_filename: str,
) -> None:
    summary = make_method_summary(
        data,
        original_column,
        chunk_column,
    )

    save_table(summary, table_filename)

    x = np.arange(len(summary))

    figure, axis = plt.subplots(figsize=(12.5, 6.8))

    axis.plot(
        x,
        summary["original_mean"],
        marker="o",
        linewidth=2,
        label=f"Original {public_label}",
    )
    axis.fill_between(
        x,
        summary["original_ci_low"],
        summary["original_ci_high"],
        alpha=0.15,
    )

    axis.plot(
        x,
        summary["chunk_mean"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label=f"Chunk-aggregated {public_label}",
    )
    axis.fill_between(
        x,
        summary["chunk_ci_low"],
        summary["chunk_ci_high"],
        alpha=0.15,
    )

    axis.set_xticks(x)
    axis.set_xticklabels(
        summary["length_bin_label"],
        rotation=35,
        ha="right",
    )

    axis.set_xlabel("Original speech-length group (words)")
    axis.set_ylabel(f"Mean {public_label}")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)

    set_axis_title(
        axis,
        f"Original and Chunk-Aggregated {public_label} "
        "by Original Speech Length",
    )

    figure.tight_layout()
    save_figure(figure, filename)


# =============================================================================
# Figure 3: score changes after chunking
# =============================================================================

def make_delta_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for bin_label, group in data.groupby(
        "length_bin_label",
        observed=True,
    ):
        llm_mean, llm_low, llm_high, n = mean_ci_95(
            group["delta_llm"]
        )
        km_mean, km_low, km_high, _ = mean_ci_95(
            group["delta_kmeans"]
        )

        rows.append(
            {
                "length_bin_label": str(bin_label),
                "n": n,
                "mean_delta_llm": llm_mean,
                "llm_ci_low": llm_low,
                "llm_ci_high": llm_high,
                "mean_delta_kmeans": km_mean,
                "kmeans_ci_low": km_low,
                "kmeans_ci_high": km_high,
            }
        )

    return pd.DataFrame(rows)


def plot_score_changes(data: pd.DataFrame) -> None:
    summary = make_delta_summary(data)

    save_table(
        summary,
        "table_03_score_changes_by_length.csv",
    )

    x = np.arange(len(summary))

    figure, axis = plt.subplots(figsize=(12.5, 6.8))

    axis.axhline(
        0,
        linestyle="--",
        linewidth=1.2,
        label="No change after chunking",
    )

    axis.plot(
        x,
        summary["mean_delta_llm"],
        marker="o",
        linewidth=2,
        label="LLM score change",
    )
    axis.fill_between(
        x,
        summary["llm_ci_low"],
        summary["llm_ci_high"],
        alpha=0.15,
    )

    axis.plot(
        x,
        summary["mean_delta_kmeans"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="K-means score change",
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

    axis.set_xlabel("Original speech-length group (words)")
    axis.set_ylabel(
        "Mean change: chunk-aggregated score − original score"
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)

    set_axis_title(
        axis,
        "Change in LLM and K-means Scores after Chunking",
    )

    figure.tight_layout()
    save_figure(
        figure,
        "fig_03_score_changes_after_chunking.png",
    )


# =============================================================================
# Figure 4: original vs chunk high-score probability
# =============================================================================

def make_high_probability_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for bin_label, group in data.groupby(
        "length_bin_label",
        observed=True,
    ):
        row = {
            "length_bin_label": str(bin_label),
            "n": len(group),
        }

        for variable in [
            "original_llm_high",
            "chunk_llm_high",
            "original_kmeans_high",
            "chunk_kmeans_high",
        ]:
            mean_value, ci_low, ci_high, _ = mean_ci_95(
                group[variable].astype(float)
            )

            row[f"{variable}_probability"] = mean_value
            row[f"{variable}_ci_low"] = max(0, ci_low)
            row[f"{variable}_ci_high"] = min(1, ci_high)

        rows.append(row)

    return pd.DataFrame(rows)


def plot_high_score_probabilities(
    data: pd.DataFrame,
) -> None:
    summary = make_high_probability_summary(data)

    save_table(
        summary,
        "table_04_high_score_probabilities_by_length.csv",
    )

    x = np.arange(len(summary))

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6.2),
        sharey=True,
    )

    specifications = [
        (
            axes[0],
            "LLM score",
            "original_llm_high",
            "chunk_llm_high",
        ),
        (
            axes[1],
            "K-means score",
            "original_kmeans_high",
            "chunk_kmeans_high",
        ),
    ]

    for axis, public_label, original_prefix, chunk_prefix in specifications:
        axis.plot(
            x,
            summary[f"{original_prefix}_probability"],
            marker="o",
            linewidth=2,
            label="Original score",
        )
        axis.fill_between(
            x,
            summary[f"{original_prefix}_ci_low"],
            summary[f"{original_prefix}_ci_high"],
            alpha=0.15,
        )

        axis.plot(
            x,
            summary[f"{chunk_prefix}_probability"],
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Chunk-aggregated score",
        )
        axis.fill_between(
            x,
            summary[f"{chunk_prefix}_ci_low"],
            summary[f"{chunk_prefix}_ci_high"],
            alpha=0.15,
        )

        axis.set_xticks(x)
        axis.set_xticklabels(
            summary["length_bin_label"],
            rotation=40,
            ha="right",
        )
        axis.set_xlabel("Original speech-length group (words)")
        axis.yaxis.set_major_formatter(
            PercentFormatter(1.0)
        )
        axis.grid(axis="y", alpha=0.3)
        axis.legend(frameon=False)

        set_axis_title(axis, public_label)

    axes[0].set_ylabel(
        f"Probability that score ≥ {HIGH_SCORE_THRESHOLD}"
    )

    set_figure_suptitle(
        figure,
        "High-Score Probability before and after Chunk Aggregation",
        fontsize=15,
        y=1.02,
    )

    figure.tight_layout()
    save_figure(
        figure,
        "fig_04_high_score_probability_before_after.png",
    )


# =============================================================================
# Figure 5: LLM–K-means score gap before and after chunking
# =============================================================================

def make_gap_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for bin_label, group in data.groupby(
        "length_bin_label",
        observed=True,
    ):
        original_mean, original_low, original_high, n = mean_ci_95(
            group["original_gap_llm_minus_kmeans"]
        )
        chunk_mean, chunk_low, chunk_high, _ = mean_ci_95(
            group["chunk_gap_llm_minus_kmeans"]
        )
        change_mean, change_low, change_high, _ = mean_ci_95(
            group["delta_gap"]
        )

        rows.append(
            {
                "length_bin_label": str(bin_label),
                "n": n,
                "original_gap_mean": original_mean,
                "original_gap_ci_low": original_low,
                "original_gap_ci_high": original_high,
                "chunk_gap_mean": chunk_mean,
                "chunk_gap_ci_low": chunk_low,
                "chunk_gap_ci_high": chunk_high,
                "gap_change_mean": change_mean,
                "gap_change_ci_low": change_low,
                "gap_change_ci_high": change_high,
            }
        )

    return pd.DataFrame(rows)


def plot_score_gap_before_after(data: pd.DataFrame) -> None:
    summary = make_gap_summary(data)

    save_table(
        summary,
        "table_05_llm_kmeans_gap_before_after.csv",
    )

    x = np.arange(len(summary))

    figure, axis = plt.subplots(figsize=(12.5, 6.8))

    axis.axhline(
        0,
        linestyle="--",
        linewidth=1.2,
        label="LLM score = K-means score",
    )

    axis.plot(
        x,
        summary["original_gap_mean"],
        marker="o",
        linewidth=2,
        label="Original scores",
    )
    axis.fill_between(
        x,
        summary["original_gap_ci_low"],
        summary["original_gap_ci_high"],
        alpha=0.15,
    )

    axis.plot(
        x,
        summary["chunk_gap_mean"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Chunk-aggregated scores",
    )
    axis.fill_between(
        x,
        summary["chunk_gap_ci_low"],
        summary["chunk_gap_ci_high"],
        alpha=0.15,
    )

    axis.set_xticks(x)
    axis.set_xticklabels(
        summary["length_bin_label"],
        rotation=35,
        ha="right",
    )

    axis.set_xlabel("Original speech-length group (words)")
    axis.set_ylabel(
        "Mean score difference: LLM score − K-means score"
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)

    set_axis_title(
        axis,
        "LLM–K-means Score Difference before and after Chunking",
    )

    figure.tight_layout()
    save_figure(
        figure,
        "fig_05_llm_kmeans_gap_before_after.png",
    )


# =============================================================================
# Figure 6: changes by chunk count
# =============================================================================

def make_chunk_total_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for group_label, group in data.groupby(
        "chunk_total_group",
        observed=True,
    ):
        llm_mean, llm_low, llm_high, n = mean_ci_95(
            group["delta_llm"]
        )
        km_mean, km_low, km_high, _ = mean_ci_95(
            group["delta_kmeans"]
        )

        rows.append(
            {
                "chunk_total_group": str(group_label),
                "n": n,
                "mean_chunk_total": group[
                    "analysis_chunk_total"
                ].mean(),
                "mean_delta_llm": llm_mean,
                "llm_ci_low": llm_low,
                "llm_ci_high": llm_high,
                "mean_delta_kmeans": km_mean,
                "kmeans_ci_low": km_low,
                "kmeans_ci_high": km_high,
            }
        )

    return pd.DataFrame(rows)


def plot_changes_by_chunk_total(data: pd.DataFrame) -> None:
    summary = make_chunk_total_summary(data)

    save_table(
        summary,
        "table_06_score_changes_by_chunk_total.csv",
    )

    x = np.arange(len(summary))

    figure, axis = plt.subplots(figsize=(10.5, 6.3))

    axis.axhline(
        0,
        linestyle="--",
        linewidth=1.2,
        label="No change after chunking",
    )

    axis.errorbar(
        x - 0.06,
        summary["mean_delta_llm"],
        yerr=np.vstack(
            [
                summary["mean_delta_llm"]
                - summary["llm_ci_low"],
                summary["llm_ci_high"]
                - summary["mean_delta_llm"],
            ]
        ),
        fmt="o",
        capsize=4,
        linewidth=1.5,
        label="LLM score change",
    )

    axis.errorbar(
        x + 0.06,
        summary["mean_delta_kmeans"],
        yerr=np.vstack(
            [
                summary["mean_delta_kmeans"]
                - summary["kmeans_ci_low"],
                summary["kmeans_ci_high"]
                - summary["mean_delta_kmeans"],
            ]
        ),
        fmt="s",
        capsize=4,
        linewidth=1.5,
        label="K-means score change",
    )

    axis.set_xticks(x)
    axis.set_xticklabels(
        summary["chunk_total_group"]
    )

    axis.set_xlabel("Number of chunks per original speech")
    axis.set_ylabel(
        "Mean change: chunk-aggregated score − original score"
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)

    set_axis_title(
        axis,
        "Score Changes after Chunking by Number of Chunks",
    )

    figure.tight_layout()
    save_figure(
        figure,
        "fig_06_score_changes_by_chunk_total.png",
    )


# =============================================================================
# Figure 7: within-category changes
# =============================================================================

def plot_within_category_changes(data: pd.DataFrame) -> None:
    if CATEGORY_COLUMN not in data.columns:
        print(
            "Skipped category figure because category is not present."
        )
        return

    top_categories = (
        data[CATEGORY_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(TOP_N_CATEGORIES)
        .index
        .tolist()
    )

    if not top_categories:
        print(
            "Skipped category figure because no valid categories were found."
        )
        return

    rows = []

    for category in top_categories:
        category_data = data[
            data[CATEGORY_COLUMN].eq(category)
        ]

        for bin_label, group in category_data.groupby(
            "length_bin_label",
            observed=True,
        ):
            llm_mean, llm_low, llm_high, n = mean_ci_95(
                group["delta_llm"]
            )
            km_mean, km_low, km_high, _ = mean_ci_95(
                group["delta_kmeans"]
            )

            rows.append(
                {
                    "category": category,
                    "length_bin_label": str(bin_label),
                    "n": n,
                    "mean_delta_llm": llm_mean,
                    "llm_ci_low": llm_low,
                    "llm_ci_high": llm_high,
                    "mean_delta_kmeans": km_mean,
                    "kmeans_ci_low": km_low,
                    "kmeans_ci_high": km_high,
                }
            )

    summary = pd.DataFrame(rows)
    summary["keep_for_plot"] = (
        summary["n"] >= MIN_CATEGORY_BIN_N
    )

    save_table(
        summary,
        "table_07_score_changes_by_length_and_category.csv",
    )

    n_columns = 2
    n_rows = math.ceil(
        len(top_categories) / n_columns
    )

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(15, 3.7 * n_rows),
        sharex=True,
        sharey=True,
    )

    axes = np.asarray(axes).reshape(-1)

    bin_order = [
        str(value)
        for value in data["length_bin_label"].cat.categories
    ]

    x = np.arange(len(bin_order))

    for axis, category in zip(
        axes,
        top_categories,
    ):
        category_summary = (
            summary[
                summary["category"].eq(category)
            ]
            .set_index("length_bin_label")
            .reindex(bin_order)
            .reset_index()
        )

        llm_values = category_summary[
            "mean_delta_llm"
        ].where(
            category_summary["keep_for_plot"]
        )
        km_values = category_summary[
            "mean_delta_kmeans"
        ].where(
            category_summary["keep_for_plot"]
        )

        axis.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        axis.plot(
            x,
            llm_values,
            marker="o",
            linewidth=1.6,
            label="LLM",
        )

        axis.plot(
            x,
            km_values,
            marker="s",
            linestyle="--",
            linewidth=1.6,
            label="K-means",
        )

        set_axis_title(axis, category)
        axis.grid(axis="y", alpha=0.25)

    for unused_axis in axes[len(top_categories):]:
        unused_axis.axis("off")

    for axis in axes:
        if axis.has_data():
            axis.set_xticks(x)
            axis.set_xticklabels(
                bin_order,
                rotation=45,
                ha="right",
                fontsize=7,
            )

    if len(top_categories) > 0:
        axes[0].legend(
            frameon=False,
            loc="best",
        )

    set_figure_suptitle(
        figure,
        "Score Changes after Chunking within Major Categories\n"
        f"Category–length cells with n < {MIN_CATEGORY_BIN_N} are omitted",
        fontsize=15,
        y=1.005,
    )

    figure.supxlabel(
        "Original speech-length group (words)"
    )
    figure.supylabel(
        "Mean change: chunk-aggregated score − original score"
    )

    figure.tight_layout()
    save_figure(
        figure,
        "fig_07_score_changes_within_categories.png",
    )




# =============================================================================
# Figure 8: continuous LOWESS curves with bootstrap confidence intervals
# =============================================================================

def make_lowess_grid(data: pd.DataFrame) -> np.ndarray:
    """Create a common log-word-count grid over the full cleaned range."""
    lower = float(
        data[WORD_COUNT_COLUMN].min()
    )
    upper = float(
        data[WORD_COUNT_COLUMN].max()
    )

    return np.linspace(
        np.log(lower),
        np.log(upper),
        LOWESS_BOOTSTRAP_GRID_SIZE,
    )


def fit_lowess_on_grid(
    x_log: np.ndarray,
    y: np.ndarray,
    grid_log: np.ndarray,
) -> np.ndarray:
    """Fit LOWESS and interpolate fitted values onto a common grid."""
    fitted = lowess(
        endog=y,
        exog=x_log,
        frac=LOWESS_FRAC,
        return_sorted=True,
    )

    return np.interp(
        grid_log,
        fitted[:, 0],
        fitted[:, 1],
    )


def bootstrap_lowess_summary(
    data: pd.DataFrame,
    value_column: str,
    measure_label: str,
) -> pd.DataFrame:
    """Estimate LOWESS with a percentile-bootstrap 95% confidence band."""
    clean = data[
        [WORD_COUNT_COLUMN, value_column]
    ].dropna().copy()

    x_log = np.log(
        clean[WORD_COUNT_COLUMN].to_numpy()
    )
    y = clean[value_column].to_numpy()

    grid_log = make_lowess_grid(clean)

    fitted_mean = fit_lowess_on_grid(
        x_log=x_log,
        y=y,
        grid_log=grid_log,
    )

    rng = np.random.default_rng(
        LOWESS_RANDOM_SEED
    )

    n = len(clean)
    bootstrap_curves = np.empty(
        (
            LOWESS_BOOTSTRAP_REPS,
            len(grid_log),
        ),
        dtype=float,
    )

    for repetition in range(
        LOWESS_BOOTSTRAP_REPS
    ):
        sample_indices = rng.integers(
            0,
            n,
            size=n,
        )

        bootstrap_curves[repetition] = (
            fit_lowess_on_grid(
                x_log=x_log[sample_indices],
                y=y[sample_indices],
                grid_log=grid_log,
            )
        )

    ci_low = np.percentile(
        bootstrap_curves,
        2.5,
        axis=0,
    )
    ci_high = np.percentile(
        bootstrap_curves,
        97.5,
        axis=0,
    )

    return pd.DataFrame(
        {
            "measure": measure_label,
            "log_word_count": grid_log,
            "word_count": np.exp(grid_log),
            "smoothed_value": fitted_mean,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "bootstrap_repetitions": LOWESS_BOOTSTRAP_REPS,
            "lowess_fraction": LOWESS_FRAC,
        }
    )


def make_signed_lowess_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    summaries = []

    for measure_label, value_column in [
        ("LLM score change", "delta_llm"),
        ("K-means score change", "delta_kmeans"),
    ]:
        summaries.append(
            bootstrap_lowess_summary(
                data=data,
                value_column=value_column,
                measure_label=measure_label,
            )
        )

    return pd.concat(
        summaries,
        ignore_index=True,
    )


def plot_lowess_score_changes(
    data: pd.DataFrame,
) -> None:
    """
    Plot signed score changes in separate method panels.

    Background scatter points are omitted because the discrete score changes
    are heavily overplotted and add little information.
    """
    lowess_table = make_signed_lowess_table(
        data
    )

    save_table(
        lowess_table,
        "table_09_lowess_score_changes.csv",
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6.2),
        sharex=True,
    )

    specifications = [
        (
            axes[0],
            "LLM score change",
            "LLM score",
        ),
        (
            axes[1],
            "K-means score change",
            "K-means score",
        ),
    ]

    for axis, measure_label, panel_label in specifications:
        sub = lowess_table[
            lowess_table["measure"].eq(
                measure_label
            )
        ].copy()

        axis.axhline(
            0,
            linestyle="--",
            linewidth=1.2,
            label="No change after chunking",
        )

        line, = axis.plot(
            sub["word_count"],
            sub["smoothed_value"],
            linewidth=2.4,
            label=measure_label,
        )

        axis.fill_between(
            sub["word_count"],
            sub["ci_95_low"],
            sub["ci_95_high"],
            alpha=0.18,
            color=line.get_color(),
            label="95% bootstrap confidence interval",
        )

        axis.set_xscale("log")
        axis.set_xlabel(
            "Original speech length (words, log scale)"
        )
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)

        set_axis_title(axis, panel_label)

    axes[0].set_ylabel(
        "Smoothed change: chunk-aggregated score − original score"
    )

    set_figure_suptitle(
        figure,
        "Continuous Relationship between Speech Length "
        "and Score Changes after Chunking",
        fontsize=15,
        y=1.02,
    )

    figure.tight_layout()
    save_figure(
        figure,
        "fig_08_lowess_score_changes_by_length.png",
    )


# =============================================================================
# Figure 9: continuous absolute score changes
# =============================================================================

def make_absolute_lowess_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    summaries = []

    for measure_label, value_column in [
        (
            "Absolute LLM score change",
            "abs_delta_llm",
        ),
        (
            "Absolute K-means score change",
            "abs_delta_kmeans",
        ),
    ]:
        summaries.append(
            bootstrap_lowess_summary(
                data=data,
                value_column=value_column,
                measure_label=measure_label,
            )
        )

    return pd.concat(
        summaries,
        ignore_index=True,
    )


def plot_lowess_absolute_changes(
    data: pd.DataFrame,
) -> None:
    """Plot chunking sensitivity irrespective of the direction of change."""
    lowess_table = make_absolute_lowess_table(
        data
    )

    save_table(
        lowess_table,
        "table_10_lowess_absolute_score_changes.csv",
    )

    figure, axis = plt.subplots(
        figsize=(11, 6.5)
    )

    for measure_label in [
        "Absolute LLM score change",
        "Absolute K-means score change",
    ]:
        sub = lowess_table[
            lowess_table["measure"].eq(
                measure_label
            )
        ].copy()

        line, = axis.plot(
            sub["word_count"],
            sub["smoothed_value"],
            linewidth=2.4,
            label=measure_label,
        )

        axis.fill_between(
            sub["word_count"],
            sub["ci_95_low"],
            sub["ci_95_high"],
            alpha=0.18,
            color=line.get_color(),
        )

    axis.set_xscale("log")
    axis.set_xlabel(
        "Original speech length (words, log scale)"
    )
    axis.set_ylabel(
        "Smoothed absolute change in score"
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    set_axis_title(
        axis,
        "Absolute Score Changes after Chunking "
        "by Original Speech Length",
    )

    figure.tight_layout()
    save_figure(
        figure,
        "fig_09_lowess_absolute_score_changes.png",
    )


# =============================================================================
# Paired summary table
# =============================================================================

def save_overall_paired_summary(data: pd.DataFrame) -> None:
    rows = []

    for label, original_column, chunk_column in [
        (
            "LLM score",
            ORIGINAL_LLM_COLUMN,
            CHUNK_LLM_COLUMN,
        ),
        (
            "K-means score",
            ORIGINAL_KMEANS_COLUMN,
            CHUNK_KMEANS_COLUMN,
        ),
        (
            "LLM − K-means score difference",
            "original_gap_llm_minus_kmeans",
            "chunk_gap_llm_minus_kmeans",
        ),
    ]:
        change = (
            data[chunk_column] - data[original_column]
        )

        change_mean, change_low, change_high, n = mean_ci_95(
            change
        )

        rows.append(
            {
                "measure": label,
                "n": n,
                "mean_original": data[original_column].mean(),
                "mean_chunk_aggregated": data[chunk_column].mean(),
                "mean_change": change_mean,
                "change_ci_95_low": change_low,
                "change_ci_95_high": change_high,
                "median_change": change.median(),
                "share_increased": (change > 0).mean(),
                "share_unchanged": (change == 0).mean(),
                "share_decreased": (change < 0).mean(),
            }
        )

    save_table(
        pd.DataFrame(rows),
        "table_08_overall_paired_summary.csv",
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    data, chunk_count_column = load_data()

    # Retained core figures only.
    plot_score_changes(data)
    plot_within_category_changes(data)

    save_overall_paired_summary(data)

    print("\nAll chunk-check outputs completed.")
    print(f"Input: {INPUT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Chunk-count column used: {chunk_count_column}")
    print(
        f"Minimum chunk total in main analysis: "
        f"{MIN_CHUNK_TOTAL}"
    )
    print(f"Figure titles enabled: {SHOW_TITLES}")


if __name__ == "__main__":
    main()
