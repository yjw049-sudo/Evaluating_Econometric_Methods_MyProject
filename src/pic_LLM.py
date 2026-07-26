from __future__ import annotations

"""
LLM chunk-score diagnostics

Place this file at:
    src/pic_LLM.py

Run from the project root:
    python src/pic_LLM.py

Inputs:
    output/Speech_sample.csv
    output/Speech_chunk.csv

Outputs:
    output/pic/LLM/fig_01_full_to_chunk_mean_transition_heatmap.png
    output/pic/LLM/fig_02_delta_by_full_score.png
    output/pic/LLM/fig_03_full_score_distance_to_chunk_aggregations.png
    output/pic/LLM/fig_04_delta_by_chunk_score_heterogeneity.png
    output/pic/LLM/fig_05_signed_aggregation_difference_by_full_score.png
    output/pic/LLM/fig_06_delta_by_zero_vs_positive_heterogeneity.png

The script studies why the speech-level mean chunk score may be lower than
the full-speech LLM score. It does not run regressions and does not output CSVs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text


# ---------------------------------------------------------------------
# Paths and analysis settings
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FULL_FILE = PROJECT_ROOT / "output" / "Speech_sample.csv"
CHUNK_FILE = PROJECT_ROOT / "output" / "Speech_chunk.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pic" / "LLM"

MIN_CHUNK_TOTAL = 2
HIGH_SCORE_THRESHOLD = 0.75
SCORE_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
HETEROGENEITY_BIN_COUNT = 10
POSITIVE_HETEROGENEITY_BIN_COUNT = 4
ZERO_TOLERANCE = 1e-9

FIGSIZE = (12, 7)
DPI = 300
FONT_SIZE = 16
MIN_FONT_SIZE = 12
SHOW_TITLES = False

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 1,
})


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def set_axis_title(
    axis: plt.Axes,
    title: str,
    **kwargs,
) -> None:
    """Add an axis title only when SHOW_TITLES is enabled."""
    if SHOW_TITLES:
        axis.set_title(title, **kwargs)


@dataclass(frozen=True)
class MeanCI:
    mean: float
    low: float
    high: float
    n: int


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file was not found: {path}")


def read_required_columns(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    required = list(required_columns)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]

    if missing:
        raise KeyError(
            f"{path.name} is missing required columns: {missing}\n"
            f"Available columns include: {header[:30]}"
        )

    return pd.read_csv(path, usecols=required, low_memory=False)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def mean_ci(values: pd.Series) -> MeanCI:
    clean = numeric(values).dropna()
    n = int(clean.shape[0])

    if n == 0:
        return MeanCI(np.nan, np.nan, np.nan, 0)

    mean_value = float(clean.mean())

    if n == 1:
        return MeanCI(mean_value, mean_value, mean_value, 1)

    standard_error = float(clean.std(ddof=1) / np.sqrt(n))
    margin = 1.96 * standard_error

    return MeanCI(
        mean=mean_value,
        low=mean_value - margin,
        high=mean_value + margin,
        n=n,
    )


def upper_quartile_mean(values: pd.Series) -> float:
    clean = numeric(values).dropna()

    if clean.empty:
        return np.nan

    threshold = clean.quantile(0.75)
    upper_values = clean.loc[clean >= threshold]

    if upper_values.empty:
        return float(clean.max())

    return float(upper_values.mean())


def make_score_bins(series: pd.Series) -> pd.Series:
    """Group scores into five fixed intervals: [0,.2), [.2,.4), ..., [.8,1]."""
    display_edges = np.array(SCORE_BINS, dtype=float)
    cut_edges = display_edges.copy()

    # pd.cut uses right-open intervals below; extend the final edge slightly
    # so that an exact score of 1.0 is included in the last group.
    cut_edges[-1] += 1e-10

    labels = [
        f"{display_edges[i]:.1f}–{display_edges[i + 1]:.1f}"
        for i in range(len(display_edges) - 1)
    ]

    return pd.cut(
        series.clip(0, 1),
        bins=cut_edges,
        labels=labels,
        include_lowest=True,
        right=False,
        ordered=True,
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    output_path = OUTPUT_DIR / filename
    enlarge_figure_fonts(fig)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def enlarge_figure_fonts(figure: plt.Figure) -> None:
    """Ensure every text element is large enough for paper figures."""
    for text in figure.findobj(match=Text):
        text.set_fontsize(max(text.get_fontsize(), MIN_FONT_SIZE))


# ---------------------------------------------------------------------
# Data cleaning and speech-level construction
# ---------------------------------------------------------------------

def load_and_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    require_file(FULL_FILE)
    require_file(CHUNK_FILE)

    full = read_required_columns(
        FULL_FILE,
        [
            "basepk",
            "procedural_score",
            "speechtext_word_count",
        ],
    ).rename(
        columns={
            "procedural_score": "full_score",
            "speechtext_word_count": "original_word_count",
        }
    )

    chunks = read_required_columns(
        CHUNK_FILE,
        [
            "basepk",
            "chunk_index",
            "chunk_total",
            "procedural_score",
        ],
    ).rename(columns={"procedural_score": "chunk_score"})

    # Numeric conversion
    full["full_score"] = numeric(full["full_score"])
    full["original_word_count"] = numeric(full["original_word_count"])

    chunks["chunk_index"] = numeric(chunks["chunk_index"])
    chunks["chunk_total"] = numeric(chunks["chunk_total"])
    chunks["chunk_score"] = numeric(chunks["chunk_score"])

    # Full-speech cleaning
    full = full.dropna(
        subset=["basepk", "full_score", "original_word_count"]
    ).copy()

    full = full.loc[
        full["full_score"].between(0, 1, inclusive="both")
        & (full["original_word_count"] > 0)
    ].copy()

    duplicate_full_ids = full.loc[
        full.duplicated("basepk", keep=False), "basepk"
    ].unique()

    if len(duplicate_full_ids) > 0:
        print(
            f"Dropping {len(duplicate_full_ids):,} basepk values duplicated "
            "in Speech_sample.csv."
        )
        full = full.loc[~full["basepk"].isin(duplicate_full_ids)].copy()

    # Chunk row cleaning
    chunks = chunks.dropna(
        subset=[
            "basepk",
            "chunk_index",
            "chunk_total",
            "chunk_score",
        ]
    ).copy()

    chunks = chunks.loc[
        chunks["chunk_score"].between(0, 1, inclusive="both")
        & (chunks["chunk_total"] >= MIN_CHUNK_TOTAL)
    ].copy()

    chunks["chunk_index"] = chunks["chunk_index"].astype(int)
    chunks["chunk_total"] = chunks["chunk_total"].astype(int)

    invalid_basepks: set = set()

    # Duplicate (basepk, chunk_index)
    duplicate_pairs = chunks.duplicated(
        ["basepk", "chunk_index"], keep=False
    )
    invalid_basepks.update(chunks.loc[duplicate_pairs, "basepk"].tolist())

    # Inconsistent chunk_total within a speech
    inconsistent_total = (
        chunks.groupby("basepk")["chunk_total"].nunique().loc[lambda x: x != 1]
    )
    invalid_basepks.update(inconsistent_total.index.tolist())

    # Invalid chunk_index range
    invalid_index = (
        (chunks["chunk_index"] < 0)
        | (chunks["chunk_index"] >= chunks["chunk_total"])
    )
    invalid_basepks.update(chunks.loc[invalid_index, "basepk"].tolist())

    if invalid_basepks:
        print(
            f"Dropping {len(invalid_basepks):,} speeches with duplicate, "
            "inconsistent, or invalid chunk records."
        )
        chunks = chunks.loc[~chunks["basepk"].isin(invalid_basepks)].copy()

    # Require a complete set of chunks: 0, ..., chunk_total - 1
    structure = chunks.groupby("basepk").agg(
        observed_chunk_rows=("chunk_index", "size"),
        observed_unique_indices=("chunk_index", "nunique"),
        minimum_chunk_index=("chunk_index", "min"),
        maximum_chunk_index=("chunk_index", "max"),
        declared_chunk_total=("chunk_total", "first"),
    )

    complete_mask = (
        (structure["observed_chunk_rows"] == structure["declared_chunk_total"])
        & (
            structure["observed_unique_indices"]
            == structure["declared_chunk_total"]
        )
        & (structure["minimum_chunk_index"] == 0)
        & (
            structure["maximum_chunk_index"]
            == structure["declared_chunk_total"] - 1
        )
    )

    incomplete_basepks = structure.index[~complete_mask]

    if len(incomplete_basepks) > 0:
        print(
            f"Dropping {len(incomplete_basepks):,} speeches with incomplete "
            "chunk sequences."
        )
        chunks = chunks.loc[
            ~chunks["basepk"].isin(incomplete_basepks)
        ].copy()

    valid_ids = np.intersect1d(
        full["basepk"].unique(),
        chunks["basepk"].unique(),
    )

    full = full.loc[full["basepk"].isin(valid_ids)].copy()
    chunks = chunks.loc[chunks["basepk"].isin(valid_ids)].copy()

    print(f"Valid full speeches: {full['basepk'].nunique():,}")
    print(f"Valid chunk rows:     {len(chunks):,}")

    return full, chunks


def build_speech_level_data(
    full: pd.DataFrame,
    chunks: pd.DataFrame,
) -> pd.DataFrame:
    grouped = chunks.groupby("basepk")["chunk_score"]

    speech = grouped.agg(
        chunk_mean="mean",
        chunk_median="median",
        chunk_max="max",
        chunk_min="min",
        chunk_sd=lambda values: float(
            numeric(values).std(ddof=0)
        ),
        chunk_n="size",
        chunk_unique_scores="nunique",
    ).reset_index()

    upper_means = (
        grouped.apply(upper_quartile_mean)
        .rename("chunk_upper_quartile_mean")
        .reset_index()
    )

    high_share = (
        chunks.assign(
            chunk_high=chunks["chunk_score"] >= HIGH_SCORE_THRESHOLD
        )
        .groupby("basepk")["chunk_high"]
        .mean()
        .rename("high_chunk_share")
        .reset_index()
    )

    speech = speech.merge(upper_means, on="basepk", how="left")
    speech = speech.merge(high_share, on="basepk", how="left")
    speech = speech.merge(full, on="basepk", how="inner", validate="one_to_one")

    speech["chunk_range"] = speech["chunk_max"] - speech["chunk_min"]
    speech["delta_llm"] = speech["chunk_mean"] - speech["full_score"]

    speech["change_direction"] = np.select(
        [
            speech["delta_llm"] < -ZERO_TOLERANCE,
            speech["delta_llm"] > ZERO_TOLERANCE,
        ],
        [
            "Decrease",
            "Increase",
        ],
        default="Unchanged",
    )

    for variable in [
        "chunk_mean",
        "chunk_median",
        "chunk_max",
        "chunk_upper_quartile_mean",
    ]:
        speech[f"signed_difference_{variable}"] = (
            speech[variable] - speech["full_score"]
        )
        speech[f"abs_distance_{variable}"] = (
            speech[f"signed_difference_{variable}"]
        ).abs()

    speech["full_score_bin"] = make_score_bins(speech["full_score"])
    speech["chunk_mean_bin"] = make_score_bins(speech["chunk_mean"])

    return speech


# ---------------------------------------------------------------------
# Figure 1: Full-score to mean-chunk-score transition heatmap
# ---------------------------------------------------------------------

def plot_transition_heatmap(speech: pd.DataFrame) -> None:
    score_labels = list(speech["full_score_bin"].cat.categories)

    counts = pd.crosstab(
        speech["full_score_bin"],
        speech["chunk_mean_bin"],
        dropna=False,
    ).reindex(
        index=score_labels,
        columns=score_labels,
        fill_value=0,
    )

    row_totals = counts.sum(axis=1).to_numpy()
    shares = np.divide(
        counts.to_numpy(dtype=float),
        row_totals[:, None],
        out=np.zeros_like(counts.to_numpy(dtype=float)),
        where=row_totals[:, None] != 0,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    image = ax.imshow(
        shares,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=max(float(np.nanmax(shares)), 0.01),
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Row-normalized share")

    ax.set_xticks(np.arange(len(score_labels)))
    ax.set_yticks(np.arange(len(score_labels)))
    ax.set_xticklabels(score_labels, rotation=35, ha="right")
    ax.set_yticklabels(
        [
            f"{label} (n={int(row_totals[i])})"
            for i, label in enumerate(score_labels)
        ]
    )

    ax.set_xlabel("Mean chunk procedural score")
    ax.set_ylabel("Full-speech procedural score")
    set_axis_title(
        ax,
        "Full-Speech Score to Mean Chunk Score Transition"
    )

    # Diagonal reference
    ax.plot(
        [-0.5, len(score_labels) - 0.5],
        [-0.5, len(score_labels) - 0.5],
        linestyle="--",
        linewidth=1.2,
    )

    for row in range(shares.shape[0]):
        for column in range(shares.shape[1]):
            value = shares[row, column]
            if value >= 0.01:
                ax.text(
                    column,
                    row,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    save_figure(
        fig,
        "fig_01_full_to_chunk_mean_transition_heatmap.png",
    )


# ---------------------------------------------------------------------
# Figure 2: Delta and direction shares by full-speech score
# ---------------------------------------------------------------------

def plot_delta_by_full_score(speech: pd.DataFrame) -> None:
    rows: list[dict] = []

    for score_bin, group in speech.groupby(
        "full_score_bin",
        observed=False,
        sort=True,
    ):
        if group.empty:
            continue

        delta_summary = mean_ci(group["delta_llm"])

        direction_share = (
            group["change_direction"]
            .value_counts(normalize=True)
            .reindex(
                ["Decrease", "Unchanged", "Increase"],
                fill_value=0.0,
            )
        )

        rows.append(
            {
                "score_bin": str(score_bin),
                "n": len(group),
                "mean_delta": delta_summary.mean,
                "ci_low": delta_summary.low,
                "ci_high": delta_summary.high,
                "share_decrease": direction_share["Decrease"],
                "share_unchanged": direction_share["Unchanged"],
                "share_increase": direction_share["Increase"],
            }
        )

    summary = pd.DataFrame(rows)
    x = np.arange(len(summary))

    fig, share_axis = plt.subplots(figsize=FIGSIZE)

    bottom = np.zeros(len(summary))
    for column, label in [
        ("share_decrease", "Decrease"),
        ("share_unchanged", "Unchanged"),
        ("share_increase", "Increase"),
    ]:
        values = summary[column].to_numpy()
        share_axis.bar(
            x,
            values,
            bottom=bottom,
            label=label,
            alpha=0.55,
        )
        bottom += values

    share_axis.set_ylim(0, 1)
    share_axis.set_ylabel("Share of speeches")
    share_axis.set_xlabel("Full-speech procedural-score group")
    share_axis.set_xticks(x)
    share_axis.set_xticklabels(
        [
            f"{label}\n(n={n})"
            for label, n in zip(summary["score_bin"], summary["n"])
        ],
        rotation=35,
        ha="right",
    )

    delta_axis = share_axis.twinx()

    y_error = np.vstack(
        [
            summary["mean_delta"] - summary["ci_low"],
            summary["ci_high"] - summary["mean_delta"],
        ]
    )

    delta_axis.errorbar(
        x,
        summary["mean_delta"],
        yerr=y_error,
        marker="o",
        linestyle="-",
        capsize=4,
        label="Mean chunk score − full score",
    )
    delta_axis.axhline(0, linestyle="--", linewidth=1)
    delta_axis.set_ylabel("Mean score change")

    handles_1, labels_1 = share_axis.get_legend_handles_labels()
    handles_2, labels_2 = delta_axis.get_legend_handles_labels()

    share_axis.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="best",
    )

    set_axis_title(
        share_axis,
        "Direction and Magnitude of LLM Score Change by Full-Speech Score"
    )
    share_axis.grid(axis="y", alpha=0.25)

    save_figure(fig, "fig_02_delta_by_full_score.png")


# ---------------------------------------------------------------------
# Figure 3: Which chunk aggregation best matches the full score?
# ---------------------------------------------------------------------

def plot_aggregation_distance(speech: pd.DataFrame) -> None:
    aggregation_variables = [
        (
            "Mean",
            "abs_distance_chunk_mean",
        ),
        (
            "Median",
            "abs_distance_chunk_median",
        ),
        (
            "Maximum",
            "abs_distance_chunk_max",
        ),
        (
            "Upper-quartile mean",
            "abs_distance_chunk_upper_quartile_mean",
        ),
    ]

    rows: list[dict] = []

    for label, variable in aggregation_variables:
        summary = mean_ci(speech[variable])
        rows.append(
            {
                "aggregation": label,
                "mean_distance": summary.mean,
                "ci_low": summary.low,
                "ci_high": summary.high,
                "n": summary.n,
            }
        )

    results = pd.DataFrame(rows)
    x = np.arange(len(results))

    y_error = np.vstack(
        [
            results["mean_distance"] - results["ci_low"],
            results["ci_high"] - results["mean_distance"],
        ]
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(
        x,
        results["mean_distance"],
        yerr=y_error,
        marker="o",
        linestyle="none",
        capsize=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(results["aggregation"], rotation=20, ha="right")
    ax.set_ylabel(
        "Mean absolute distance from full-speech score"
    )
    ax.set_xlabel("Chunk-score aggregation rule")
    set_axis_title(
        ax,
        "Which Chunk Aggregation Most Closely Matches the Full-Speech Score?"
    )
    ax.grid(axis="y", alpha=0.25)

    for i, value in enumerate(results["mean_distance"]):
        ax.annotate(
            f"{value:.3f}",
            xy=(i, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    save_figure(
        fig,
        "fig_03_full_score_distance_to_chunk_aggregations.png",
    )

    closest_row = results.loc[results["mean_distance"].idxmin()]
    print(
        "Closest aggregation to the full-speech score: "
        f"{closest_row['aggregation']} "
        f"(mean absolute distance = {closest_row['mean_distance']:.4f})"
    )


# ---------------------------------------------------------------------
# Figure 4: Delta by within-speech chunk-score heterogeneity
# ---------------------------------------------------------------------

def plot_delta_by_heterogeneity(speech: pd.DataFrame) -> None:
    working = speech.dropna(subset=["chunk_sd", "delta_llm"]).copy()

    try:
        working["heterogeneity_bin"] = pd.qcut(
            working["chunk_sd"],
            q=HETEROGENEITY_BIN_COUNT,
            duplicates="drop",
        )
    except ValueError:
        working["heterogeneity_bin"] = pd.cut(
            working["chunk_sd"],
            bins=min(
                HETEROGENEITY_BIN_COUNT,
                max(1, working["chunk_sd"].nunique()),
            ),
            duplicates="drop",
        )

    rows: list[dict] = []

    for _, group in working.groupby(
        "heterogeneity_bin",
        observed=True,
        sort=True,
    ):
        if group.empty:
            continue

        delta_summary = mean_ci(group["delta_llm"])
        minimum_sd = float(group["chunk_sd"].min())
        maximum_sd = float(group["chunk_sd"].max())
        share_decrease = float(
            (group["change_direction"] == "Decrease").mean()
        )

        rows.append(
            {
                "label": f"{minimum_sd:.3f}–{maximum_sd:.3f}",
                "n": len(group),
                "mean_delta": delta_summary.mean,
                "ci_low": delta_summary.low,
                "ci_high": delta_summary.high,
                "share_decrease": share_decrease,
            }
        )

    summary = pd.DataFrame(rows)
    x = np.arange(len(summary))

    y_error = np.vstack(
        [
            summary["mean_delta"] - summary["ci_low"],
            summary["ci_high"] - summary["mean_delta"],
        ]
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(
        x,
        summary["mean_delta"],
        yerr=y_error,
        marker="o",
        linestyle="-",
        capsize=4,
    )
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{label}\n(n={n})"
            for label, n in zip(summary["label"], summary["n"])
        ],
        rotation=35,
        ha="right",
    )
    ax.set_xlabel(
        "Within-speech standard deviation of chunk scores"
    )
    ax.set_ylabel("Mean chunk score − full-speech score")
    set_axis_title(
        ax,
        "LLM Score Change by Within-Speech Chunk-Score Heterogeneity"
    )
    ax.grid(axis="y", alpha=0.25)

    for i, row in summary.iterrows():
        ax.annotate(
            f"Decrease: {row['share_decrease']:.0%}",
            xy=(i, row["mean_delta"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    save_figure(
        fig,
        "fig_04_delta_by_chunk_score_heterogeneity.png",
    )



# ---------------------------------------------------------------------
# Figure 5: Signed aggregation differences by full-speech score
# ---------------------------------------------------------------------

def plot_signed_aggregation_difference_by_full_score(
    speech: pd.DataFrame,
) -> None:
    """
    Compare aggregation score minus full-speech score.

    Values below zero mean that the chunk aggregation is systematically
    lower than the full-speech score. Values above zero mean that it is
    systematically higher.
    """
    aggregation_variables = [
        ("Mean", "signed_difference_chunk_mean"),
        ("Median", "signed_difference_chunk_median"),
        ("Maximum", "signed_difference_chunk_max"),
        (
            "Upper-quartile mean",
            "signed_difference_chunk_upper_quartile_mean",
        ),
    ]

    score_labels = list(speech["full_score_bin"].cat.categories)
    nonempty_labels = [
        label
        for label in score_labels
        if (speech["full_score_bin"] == label).any()
    ]

    rows: list[dict] = []

    for score_bin in nonempty_labels:
        group = speech.loc[speech["full_score_bin"] == score_bin]

        for aggregation_label, variable in aggregation_variables:
            summary = mean_ci(group[variable])
            rows.append(
                {
                    "score_bin": str(score_bin),
                    "aggregation": aggregation_label,
                    "mean_difference": summary.mean,
                    "ci_low": summary.low,
                    "ci_high": summary.high,
                    "n": summary.n,
                }
            )

    results = pd.DataFrame(rows)
    x = np.arange(len(nonempty_labels))

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for aggregation_label, _ in aggregation_variables:
        subset = (
            results.loc[results["aggregation"] == aggregation_label]
            .set_index("score_bin")
            .reindex([str(label) for label in nonempty_labels])
            .reset_index()
        )

        y_error = np.vstack(
            [
                subset["mean_difference"] - subset["ci_low"],
                subset["ci_high"] - subset["mean_difference"],
            ]
        )

        ax.errorbar(
            x,
            subset["mean_difference"],
            yerr=y_error,
            marker="o",
            linestyle="-",
            capsize=4,
            label=aggregation_label,
        )

    group_sizes = (
        speech.groupby("full_score_bin", observed=False)
        .size()
        .reindex(nonempty_labels)
        .astype(int)
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{label}\n(n={group_sizes.loc[label]})"
            for label in nonempty_labels
        ],
        rotation=35,
        ha="right",
    )
    ax.set_xlabel("Full-speech procedural-score group")
    ax.set_ylabel("Chunk aggregation score − full-speech score")
    set_axis_title(
        ax,
        "Signed Difference between Chunk Aggregations and Full-Speech Score"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Chunk aggregation")

    save_figure(
        fig,
        "fig_05_signed_aggregation_difference_by_full_score.png",
    )


# ---------------------------------------------------------------------
# Figure 6: Zero heterogeneity versus positive-SD quantile groups
# ---------------------------------------------------------------------

def plot_delta_by_zero_vs_positive_heterogeneity(
    speech: pd.DataFrame,
) -> None:
    """
    Separate speeches with exactly zero chunk-score heterogeneity from
    speeches with positive heterogeneity. Positive SD values are then divided
    into approximately equal-frequency groups.
    """
    working = speech.dropna(subset=["chunk_sd", "delta_llm"]).copy()

    zero_mask = working["chunk_sd"].abs() <= ZERO_TOLERANCE
    zero_group = working.loc[zero_mask].copy()
    positive = working.loc[~zero_mask].copy()

    grouped_data: list[tuple[str, pd.DataFrame]] = []

    if not zero_group.empty:
        grouped_data.append(("SD = 0", zero_group))

    if not positive.empty:
        try:
            positive["positive_sd_bin"] = pd.qcut(
                positive["chunk_sd"],
                q=POSITIVE_HETEROGENEITY_BIN_COUNT,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            ranks = positive["chunk_sd"].rank(method="first")
            positive["positive_sd_bin"] = pd.qcut(
                ranks,
                q=min(
                    POSITIVE_HETEROGENEITY_BIN_COUNT,
                    len(positive),
                ),
                labels=False,
                duplicates="drop",
            )

        positive = positive.dropna(subset=["positive_sd_bin"]).copy()
        positive["positive_sd_bin"] = positive["positive_sd_bin"].astype(int)

        for bin_id in sorted(positive["positive_sd_bin"].unique()):
            group = positive.loc[positive["positive_sd_bin"] == bin_id].copy()
            minimum_sd = float(group["chunk_sd"].min())
            maximum_sd = float(group["chunk_sd"].max())
            label = (
                f"Positive SD Q{bin_id + 1}\n"
                f"{minimum_sd:.3f}–{maximum_sd:.3f}"
            )
            grouped_data.append((label, group))

    rows: list[dict] = []

    for label, group in grouped_data:
        delta_summary = mean_ci(group["delta_llm"])
        share_decrease = float(
            (group["change_direction"] == "Decrease").mean()
        )

        rows.append(
            {
                "label": label,
                "n": len(group),
                "mean_delta": delta_summary.mean,
                "ci_low": delta_summary.low,
                "ci_high": delta_summary.high,
                "share_decrease": share_decrease,
            }
        )

    summary = pd.DataFrame(rows)
    x = np.arange(len(summary))

    y_error = np.vstack(
        [
            summary["mean_delta"] - summary["ci_low"],
            summary["ci_high"] - summary["mean_delta"],
        ]
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(
        x,
        summary["mean_delta"],
        yerr=y_error,
        marker="o",
        linestyle="-",
        capsize=4,
    )
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{label}\n(n={n})"
            for label, n in zip(summary["label"], summary["n"])
        ],
        rotation=30,
        ha="right",
    )
    ax.set_xlabel("Within-speech chunk-score heterogeneity group")
    ax.set_ylabel("Mean chunk score − full-speech score")
    set_axis_title(
        ax,
        "LLM Score Change for Zero versus Positive Chunk-Score Heterogeneity"
    )
    ax.grid(axis="y", alpha=0.25)

    for i, row in summary.iterrows():
        ax.annotate(
            f"Decrease: {row['share_decrease']:.0%}",
            xy=(i, row["mean_delta"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    save_figure(
        fig,
        "fig_06_delta_by_zero_vs_positive_heterogeneity.png",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Full file:    {FULL_FILE}")
    print(f"Chunk file:   {CHUNK_FILE}")
    print(f"Output dir:   {OUTPUT_DIR}")

    full, chunks = load_and_clean_data()
    speech = build_speech_level_data(full, chunks)

    print(f"Speech-level analysis sample: {len(speech):,}")
    print(
        "Mean full-speech score: "
        f"{speech['full_score'].mean():.4f}"
    )
    print(
        "Mean speech-level chunk score: "
        f"{speech['chunk_mean'].mean():.4f}"
    )
    print(
        "Mean score change (chunk mean − full): "
        f"{speech['delta_llm'].mean():.4f}"
    )

    direction_shares = (
        speech["change_direction"]
        .value_counts(normalize=True)
        .reindex(["Decrease", "Unchanged", "Increase"], fill_value=0)
    )

    print("Direction shares:")
    for label, value in direction_shares.items():
        print(f"  {label}: {value:.2%}")

    plot_transition_heatmap(speech)
    plot_signed_aggregation_difference_by_full_score(speech)
    plot_delta_by_zero_vs_positive_heterogeneity(speech)

    print("Completed LLM chunk-score diagnostic figures.")


if __name__ == "__main__":
    main()
