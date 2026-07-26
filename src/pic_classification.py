"""Analyze entropy, retention, and classification transitions after chunking.

Inputs:
    output/Speech_sample.csv
    output/Speech_chunk.csv

Outputs:
    output/pic/classification/

Run:
    python src/pic_classification.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


# =============================================================================
# User settings
# =============================================================================

MIN_CHUNK_TOTAL = 2
TRANSITION_INTERVALS = [
    (2, 6, "2-6"),
    (6, 10, "6-10"),
    (10, None, "10+"),
]

FIGURE_DPI = 300
FONT_SIZE = 16
MIN_FONT_SIZE = 12
SHOW_TITLES = False
SHOW_FIGURES = False
REQUIRE_COMPLETE_CHUNKS = True
HEATMAP_ANNOTATION_THRESHOLD = 0.05
CSV_SEPARATOR = ","
SANKEY_TOP_N_FLOWS = 8
TOP_DESTINATION_N = 10

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
ORIGINAL_INPUT_FILE = PROJECT_DIR / "output" / "Speech_sample.csv"
CHUNK_INPUT_FILE = PROJECT_DIR / "output" / "Speech_chunk.csv"
OUTPUT_DIR = PROJECT_DIR / "output" / "pic" / "classification"

BASEPK = "basepk"
CATEGORY = "category"
CLUSTER = "cluster"
CHUNK_TOTAL = "chunk_total"
TOPIC_ERROR = "topic_error"
CATEGORY_ERROR = "category_error"
WEIGHT = "speech_weight"
INTERVAL = "transition_interval"

ORIGINAL_COLUMNS = [
    BASEPK,
    CATEGORY,
    CLUSTER,
    TOPIC_ERROR,
    CATEGORY_ERROR,
]
CHUNK_COLUMNS = [
    *ORIGINAL_COLUMNS,
    CHUNK_TOTAL,
]


# =============================================================================
# Utilities
# =============================================================================

def set_title(axis: plt.Axes, title: str) -> None:
    if SHOW_TITLES:
        axis.set_title(title)


def save_figure(fig: plt.Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    enlarge_figure_fonts(fig)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    print(f"Saved figure: {path}")


def enlarge_figure_fonts(figure: plt.Figure) -> None:
    """Ensure every text element is large enough for paper figures."""
    for text in figure.findobj(match=Text):
        text.set_fontsize(max(text.get_fontsize(), MIN_FONT_SIZE))


def save_table(df: pd.DataFrame, filename: str) -> None:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved table: {path}")


def validate_columns(
    header: pd.DataFrame,
    filename: Path,
    required: list[str],
) -> None:
    missing = [c for c in required if c not in header.columns]
    if missing:
        raise ValueError(
            f"{filename} is missing columns: {missing}\n"
            f"Available columns: {list(header.columns)}"
        )


def normalize_basepk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[BASEPK] = pd.to_numeric(df[BASEPK], errors="coerce").astype("Int64")
    return df.dropna(subset=[BASEPK]).copy()


def remove_error_rows(df: pd.DataFrame) -> pd.DataFrame:
    topic_bad = df[TOPIC_ERROR].fillna("").astype(str).str.strip().ne("")
    category_bad = (
        df[CATEGORY_ERROR].fillna("").astype(str).str.strip().ne("")
    )
    return df.loc[~(topic_bad | category_bad)].copy()


def assign_interval(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        values,
        bins=[2, 6, 10, np.inf],
        labels=["2-6", "6-10", "10+"],
        right=False,
        include_lowest=True,
        ordered=True,
    )


# =============================================================================
# Load and clean
# =============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        ORIGINAL_INPUT_FILE: ORIGINAL_COLUMNS,
        CHUNK_INPUT_FILE: CHUNK_COLUMNS,
    }

    for filename, columns in required.items():
        if not filename.exists():
            raise FileNotFoundError(f"Input file not found: {filename}")
        header = pd.read_csv(filename, sep=CSV_SEPARATOR, nrows=0)
        validate_columns(header, filename, columns)

    original = pd.read_csv(
        ORIGINAL_INPUT_FILE,
        sep=CSV_SEPARATOR,
        usecols=ORIGINAL_COLUMNS,
        low_memory=False,
    )
    chunks = pd.read_csv(
        CHUNK_INPUT_FILE,
        sep=CSV_SEPARATOR,
        usecols=CHUNK_COLUMNS,
        low_memory=False,
    )

    initial_original_rows = len(original)
    initial_chunk_rows = len(chunks)

    original = remove_error_rows(normalize_basepk(original))
    chunks = remove_error_rows(normalize_basepk(chunks))

    for df in [original, chunks]:
        df[CATEGORY] = df[CATEGORY].fillna("").astype(str).str.strip()
        df[CLUSTER] = pd.to_numeric(df[CLUSTER], errors="coerce")

    chunks[CHUNK_TOTAL] = pd.to_numeric(
        chunks[CHUNK_TOTAL],
        errors="coerce",
    )

    original = original.dropna(subset=[BASEPK, CATEGORY, CLUSTER]).copy()
    chunks = chunks.dropna(
        subset=[BASEPK, CATEGORY, CLUSTER, CHUNK_TOTAL]
    ).copy()

    original = original.loc[
        original[CATEGORY].ne("") & original[CLUSTER].ge(0)
    ].copy()
    chunks = chunks.loc[
        chunks[CATEGORY].ne("")
        & chunks[CLUSTER].ge(0)
        & chunks[CHUNK_TOTAL].ge(MIN_CHUNK_TOTAL)
    ].copy()

    original[CLUSTER] = original[CLUSTER].astype(int)
    chunks[CLUSTER] = chunks[CLUSTER].astype(int)
    chunks[CHUNK_TOTAL] = chunks[CHUNK_TOTAL].astype(int)

    if original.duplicated(BASEPK).any():
        duplicate_n = int(original.duplicated(BASEPK, keep=False).sum())
        print(
            "Warning: duplicate original basepk rows; keeping first. "
            f"Rows involved: {duplicate_n:,}"
        )
        original = original.drop_duplicates(BASEPK, keep="first").copy()

    common = set(original[BASEPK]).intersection(set(chunks[BASEPK]))
    original = original.loc[original[BASEPK].isin(common)].copy()
    chunks = chunks.loc[chunks[BASEPK].isin(common)].copy()

    chunks[WEIGHT] = 1.0 / chunks[CHUNK_TOTAL]

    diagnostics = (
        chunks.groupby(BASEPK, observed=True)
        .agg(
            rows_found=(BASEPK, "size"),
            chunk_total_min=(CHUNK_TOTAL, "min"),
            chunk_total_max=(CHUNK_TOTAL, "max"),
            total_chunk_weight=(WEIGHT, "sum"),
        )
        .reset_index()
    )
    diagnostics["chunk_total_consistent"] = diagnostics[
        "chunk_total_min"
    ].eq(diagnostics["chunk_total_max"])
    diagnostics["all_chunks_present"] = diagnostics[
        "rows_found"
    ].eq(diagnostics["chunk_total_max"])

    if not diagnostics["chunk_total_consistent"].all():
        raise ValueError("chunk_total is inconsistent within some speeches.")

    incomplete_n = int((~diagnostics["all_chunks_present"]).sum())

    if REQUIRE_COMPLETE_CHUNKS:
        complete_ids = set(
            diagnostics.loc[diagnostics["all_chunks_present"], BASEPK]
        )
        original = original.loc[original[BASEPK].isin(complete_ids)].copy()
        chunks = chunks.loc[chunks[BASEPK].isin(complete_ids)].copy()
        diagnostics = diagnostics.loc[
            diagnostics[BASEPK].isin(complete_ids)
        ].copy()

    if original.empty or chunks.empty:
        raise ValueError("No matched speeches remain after cleaning.")

    if REQUIRE_COMPLETE_CHUNKS and not np.allclose(
        diagnostics["total_chunk_weight"],
        1.0,
        atol=1e-10,
    ):
        raise ValueError("Some retained speech weights do not sum to one.")

    chunks[INTERVAL] = assign_interval(chunks[CHUNK_TOTAL])

    filter_summary = pd.DataFrame(
        {
            "metric": [
                "initial_original_rows",
                "initial_chunk_rows",
                "matched_speeches_before_completeness_filter",
                "incomplete_speeches_after_cleaning",
                "require_complete_chunks",
                "final_original_rows",
                "final_chunk_rows",
                "final_unique_speeches",
                "final_sum_chunk_weights",
            ],
            "value": [
                initial_original_rows,
                initial_chunk_rows,
                len(common),
                incomplete_n,
                REQUIRE_COMPLETE_CHUNKS,
                len(original),
                len(chunks),
                original[BASEPK].nunique(),
                chunks[WEIGHT].sum(),
            ],
        }
    )

    return original, chunks, diagnostics, filter_summary


# =============================================================================
# Entropy and retention
# =============================================================================

def entropy(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True).to_numpy(float)
    if len(probabilities) <= 1:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)))


def adjusted_entropy(
    values: pd.Series,
    chunk_total: int,
    class_count: int,
) -> tuple[float, float]:
    raw = entropy(values)
    maximum_distinct = min(int(chunk_total), int(class_count))
    if maximum_distinct <= 1:
        return raw, 0.0
    adjusted = raw / np.log(maximum_distinct)
    return raw, float(np.clip(adjusted, 0, 1))


def build_speech_metrics(
    original: pd.DataFrame,
    chunks: pd.DataFrame,
) -> pd.DataFrame:
    lookup = original[[BASEPK, CATEGORY, CLUSTER]].rename(
        columns={
            CATEGORY: "original_category",
            CLUSTER: "original_cluster",
        }
    )

    category_count = int(
        pd.concat([original[CATEGORY], chunks[CATEGORY]]).nunique()
    )
    cluster_count = int(
        pd.concat([original[CLUSTER], chunks[CLUSTER]]).nunique()
    )

    lookup = lookup.set_index(BASEPK)
    rows: list[dict[str, object]] = []

    for basepk, group in chunks.groupby(BASEPK, observed=True):
        original_row = lookup.loc[basepk]
        chunk_total = int(group[CHUNK_TOTAL].iloc[0])

        cat_raw, cat_adjusted = adjusted_entropy(
            group[CATEGORY],
            chunk_total,
            category_count,
        )
        clu_raw, clu_adjusted = adjusted_entropy(
            group[CLUSTER],
            chunk_total,
            cluster_count,
        )

        rows.append(
            {
                BASEPK: basepk,
                CHUNK_TOTAL: chunk_total,
                "original_category": original_row["original_category"],
                "original_cluster": int(original_row["original_cluster"]),
                "category_entropy_raw": cat_raw,
                "category_entropy_adjusted": cat_adjusted,
                "cluster_entropy_raw": clu_raw,
                "cluster_entropy_adjusted": clu_adjusted,
                "category_retention": float(
                    group[CATEGORY]
                    .eq(original_row["original_category"])
                    .mean()
                ),
                "cluster_retention": float(
                    group[CLUSTER]
                    .eq(int(original_row["original_cluster"]))
                    .mean()
                ),
                "category_distinct_count": int(group[CATEGORY].nunique()),
                "cluster_distinct_count": int(group[CLUSTER].nunique()),
                "category_dominant_share": float(
                    group[CATEGORY].value_counts(normalize=True).iloc[0]
                ),
                "cluster_dominant_share": float(
                    group[CLUSTER].value_counts(normalize=True).iloc[0]
                ),
            }
        )

    return pd.DataFrame(rows).sort_values([CHUNK_TOTAL, BASEPK])


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan, np.nan, np.nan
    mean = float(clean.mean())
    if len(clean) == 1:
        return mean, np.nan, np.nan
    margin = 1.96 * float(clean.std(ddof=1) / np.sqrt(len(clean)))
    return mean, mean - margin, mean + margin


def summarize_exact(
    metrics: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    rows = []

    for chunk_total, group in metrics.groupby(CHUNK_TOTAL, observed=True):
        row = {
            CHUNK_TOTAL: int(chunk_total),
            "speech_n": int(len(group)),
        }
        for variable in variables:
            mean, low, high = mean_ci(group[variable])
            row[f"mean_{variable}"] = mean
            row[f"{variable}_ci_95_low"] = low
            row[f"{variable}_ci_95_high"] = high
        rows.append(row)

    return pd.DataFrame(rows).sort_values(CHUNK_TOTAL)


def plot_entropy(summary: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(12, 6.8))
    x = summary[CHUNK_TOTAL]

    axis.plot(
        x,
        summary["mean_category_entropy_adjusted"],
        marker="o",
        linewidth=2,
        label="LLM category",
    )
    axis.fill_between(
        x,
        summary["category_entropy_adjusted_ci_95_low"],
        summary["category_entropy_adjusted_ci_95_high"],
        alpha=0.16,
    )

    axis.plot(
        x,
        summary["mean_cluster_entropy_adjusted"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="K-means cluster",
    )
    axis.fill_between(
        x,
        summary["cluster_entropy_adjusted_ci_95_low"],
        summary["cluster_entropy_adjusted_ci_95_high"],
        alpha=0.16,
    )

    axis.set_xlabel("Exact number of chunks per original speech")
    axis.set_ylabel("Mean adjusted within-speech entropy")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)
    set_title(axis, "Within-Speech Entropy by Exact Chunk Total")

    fig.tight_layout()
    save_figure(fig, "entropy_by_exact_chunk_total.png")


def plot_retention(summary: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(12, 6.8))
    x = summary[CHUNK_TOTAL]

    axis.plot(
        x,
        summary["mean_category_retention"],
        marker="o",
        linewidth=2,
        label="LLM category retention",
    )
    axis.fill_between(
        x,
        summary["category_retention_ci_95_low"],
        summary["category_retention_ci_95_high"],
        alpha=0.16,
    )

    axis.plot(
        x,
        summary["mean_cluster_retention"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="K-means cluster retention",
    )
    axis.fill_between(
        x,
        summary["cluster_retention_ci_95_low"],
        summary["cluster_retention_ci_95_high"],
        alpha=0.16,
    )

    axis.set_xlabel("Exact number of chunks per original speech")
    axis.set_ylabel("Mean share retaining original classification")
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)
    set_title(axis, "Original-Classification Retention by Exact Chunk Total")

    fig.tight_layout()
    save_figure(fig, "retention_by_exact_chunk_total.png")


# =============================================================================
# Transition matrices
# =============================================================================

def transition_matrix(
    original: pd.DataFrame,
    chunks: pd.DataFrame,
    classification: str,
    interval_label: str,
) -> pd.DataFrame:
    subset = chunks.loc[chunks[INTERVAL].astype(str).eq(interval_label)].copy()
    if subset.empty:
        raise ValueError(f"No rows for transition interval {interval_label}.")

    original_col = f"original_{classification}"
    chunk_col = f"chunk_{classification}"

    lookup = original[[BASEPK, classification]].rename(
        columns={classification: original_col}
    )
    transition = subset[[BASEPK, classification, WEIGHT]].rename(
        columns={classification: chunk_col}
    ).merge(
        lookup,
        on=BASEPK,
        how="inner",
        validate="many_to_one",
    )

    counts = transition.pivot_table(
        index=original_col,
        columns=chunk_col,
        values=WEIGHT,
        aggfunc="sum",
        fill_value=0,
        observed=True,
    )

    labels = sorted(set(counts.index).union(set(counts.columns)))
    counts = counts.reindex(index=labels, columns=labels, fill_value=0)

    return counts.div(
        counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0)


def plot_heatmap(
    shares: pd.DataFrame,
    classification_label: str,
    interval_label: str,
    filename: str,
) -> None:
    labels = shares.index.tolist()
    values = shares.to_numpy(float)

    fig, axis = plt.subplots(
        figsize=(
            max(10, 0.58 * len(labels)),
            max(8, 0.54 * len(labels)),
        )
    )

    norm = PowerNorm(
        gamma=0.65,
        vmin=0,
        vmax=max(float(values.max()), 1e-9),
    )
    image = axis.imshow(
        values,
        cmap="YlGnBu",
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )

    axis.set_xlabel(f"Chunk {classification_label.lower()}")
    axis.set_ylabel(f"Original-speech {classification_label.lower()}")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(labels, fontsize=8)

    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            value = values[r, c]
            if value >= HEATMAP_ANNOTATION_THRESHOLD:
                axis.text(
                    c,
                    r,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=(
                        "white"
                        if value > values.max() * 0.5
                        else "black"
                    ),
                )

    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Weighted row share")
    set_title(
        axis,
        f"{classification_label} Transition after Chunking\n"
        f"chunk_total interval: {interval_label}",
    )

    fig.tight_layout()
    save_figure(fig, filename)


def create_transition_outputs(
    original: pd.DataFrame,
    chunks: pd.DataFrame,
) -> None:
    category_tables = []
    cluster_tables = []

    for _, _, interval_label in TRANSITION_INTERVALS:
        category_shares = transition_matrix(
            original,
            chunks,
            CATEGORY,
            interval_label,
        )
        category_long = (
            category_shares.rename_axis(
                index="original_category",
                columns="chunk_category",
            )
            .stack()
            .rename("weighted_row_share")
            .reset_index()
        )
        category_long.insert(0, "interval_label", interval_label)
        category_tables.append(category_long)

        plot_heatmap(
            category_shares,
            "LLM category",
            interval_label,
            (
                "category_transition_heatmap_"
                f"{interval_label.replace('+', 'plus').replace('-', '_')}.png"
            ),
        )

        cluster_shares = transition_matrix(
            original,
            chunks,
            CLUSTER,
            interval_label,
        )
        cluster_long = (
            cluster_shares.rename_axis(
                index="original_cluster",
                columns="chunk_cluster",
            )
            .stack()
            .rename("weighted_row_share")
            .reset_index()
        )
        cluster_long.insert(0, "interval_label", interval_label)
        cluster_tables.append(cluster_long)

        plot_heatmap(
            cluster_shares,
            "K-means cluster",
            interval_label,
            (
                "cluster_transition_heatmap_"
                f"{interval_label.replace('+', 'plus').replace('-', '_')}.png"
            ),
        )

    save_table(
        pd.concat(category_tables, ignore_index=True),
        "category_transition_all_intervals.csv",
    )
    save_table(
        pd.concat(cluster_tables, ignore_index=True),
        "cluster_transition_all_intervals.csv",
    )


# =============================================================================
# Global cluster-flow diagnostics (all retained chunk_total >= MIN_CHUNK_TOTAL)
# =============================================================================

def build_cluster_transition_long(
    original: pd.DataFrame,
    chunks: pd.DataFrame,
) -> pd.DataFrame:
    """Build weighted original-cluster to chunk-cluster flows for all speeches."""
    lookup = original[[BASEPK, CLUSTER]].rename(
        columns={CLUSTER: "original_cluster"}
    )

    transition = chunks[[BASEPK, CLUSTER, WEIGHT]].rename(
        columns={CLUSTER: "chunk_cluster"}
    ).merge(
        lookup,
        on=BASEPK,
        how="inner",
        validate="many_to_one",
    )

    flow = (
        transition.groupby(
            ["original_cluster", "chunk_cluster"],
            observed=True,
            as_index=False,
        )
        .agg(
            weighted_flow=(WEIGHT, "sum"),
            contributing_speech_n=(BASEPK, "nunique"),
            raw_chunk_n=(BASEPK, "size"),
        )
    )

    flow["is_retained"] = flow["original_cluster"].eq(
        flow["chunk_cluster"]
    )
    return flow.sort_values("weighted_flow", ascending=False)


def plot_top_destination_clusters(flow: pd.DataFrame) -> None:
    """Rank destination clusters by weighted inflow from other clusters."""
    switched = flow.loc[~flow["is_retained"]].copy()

    destination = (
        switched.groupby("chunk_cluster", as_index=False)
        .agg(
            weighted_inflow=("weighted_flow", "sum"),
            source_cluster_n=("original_cluster", "nunique"),
            contributing_speech_n=("contributing_speech_n", "sum"),
        )
        .sort_values("weighted_inflow", ascending=False)
    )

    total_switched = destination["weighted_inflow"].sum()
    destination["share_of_all_switched_weight"] = np.where(
        total_switched > 0,
        destination["weighted_inflow"] / total_switched,
        np.nan,
    )
    destination["rank"] = np.arange(1, len(destination) + 1)

    save_table(destination, "top_destination_clusters.csv")

    plot_df = destination.head(TOP_DESTINATION_N).sort_values(
        "weighted_inflow",
        ascending=True,
    )

    fig, axis = plt.subplots(
        figsize=(11, max(6, 0.52 * len(plot_df)))
    )
    axis.barh(
        plot_df["chunk_cluster"].astype(str),
        plot_df["weighted_inflow"],
    )

    for y_pos, (_, row) in enumerate(plot_df.iterrows()):
        axis.text(
            row["weighted_inflow"],
            y_pos,
            (
                f"  {row['share_of_all_switched_weight']:.1%}; "
                f"{int(row['source_cluster_n'])} sources"
            ),
            va="center",
            fontsize=9,
        )

    axis.set_xlabel("Weighted inflow from other original clusters")
    axis.set_ylabel("Destination chunk cluster")
    axis.grid(axis="x", alpha=0.3)
    set_title(
        axis,
        "Top Destination Clusters after Chunking\n"
        "All retained speeches; diagonal retention flows excluded",
    )
    fig.tight_layout()
    save_figure(fig, "top_destination_clusters.png")


def plot_cluster_net_inflow_outflow(flow: pd.DataFrame) -> None:
    """Calculate gross inflow, gross outflow, and net flow by cluster."""
    switched = flow.loc[~flow["is_retained"]].copy()
    retained = flow.loc[flow["is_retained"]].copy()

    inflow = (
        switched.groupby("chunk_cluster")["weighted_flow"]
        .sum()
        .rename("gross_inflow")
    )
    outflow = (
        switched.groupby("original_cluster")["weighted_flow"]
        .sum()
        .rename("gross_outflow")
    )
    retained_weight = (
        retained.groupby("original_cluster")["weighted_flow"]
        .sum()
        .rename("retained_weight")
    )

    clusters = sorted(
        set(flow["original_cluster"]).union(set(flow["chunk_cluster"]))
    )

    summary = pd.DataFrame({"cluster": clusters})
    summary = summary.merge(
        inflow,
        left_on="cluster",
        right_index=True,
        how="left",
    )
    summary = summary.merge(
        outflow,
        left_on="cluster",
        right_index=True,
        how="left",
    )
    summary = summary.merge(
        retained_weight,
        left_on="cluster",
        right_index=True,
        how="left",
    )

    for column in ["gross_inflow", "gross_outflow", "retained_weight"]:
        summary[column] = summary[column].fillna(0.0)

    summary["net_inflow"] = (
        summary["gross_inflow"] - summary["gross_outflow"]
    )
    summary["absolute_net_flow"] = summary["net_inflow"].abs()
    summary = summary.sort_values("net_inflow", ascending=True)

    save_table(summary, "cluster_net_inflow_outflow.csv")

    fig, axis = plt.subplots(
        figsize=(12, max(7, 0.42 * len(summary)))
    )
    axis.barh(
        summary["cluster"].astype(str),
        summary["net_inflow"],
    )
    axis.axvline(0, linewidth=1)
    axis.set_xlabel(
        "Net weighted flow = inflow from other clusters − outflow to other clusters"
    )
    axis.set_ylabel("Cluster")
    axis.grid(axis="x", alpha=0.3)
    set_title(
        axis,
        "Net Cluster Inflow / Outflow after Chunking\n"
        "All retained speeches; diagonal retention flows excluded",
    )
    fig.tight_layout()
    save_figure(fig, "cluster_net_inflow_outflow.png")


def create_cluster_sankey(flow: pd.DataFrame) -> None:
    """Create an HTML Sankey for the largest non-diagonal cluster flows."""
    switched = flow.loc[~flow["is_retained"]].copy()
    switched = switched.sort_values(
        "weighted_flow",
        ascending=False,
    )

    save_table(switched, "cluster_transition_flows_all.csv")

    top_flows = switched.head(SANKEY_TOP_N_FLOWS).copy()
    save_table(
        top_flows,
        f"cluster_transition_top_{SANKEY_TOP_N_FLOWS}_flows.csv",
    )

    if go is None:
        print(
            "Plotly is not installed; Sankey HTML was skipped. "
            "Install it with: pip install plotly"
        )
        return

    source_labels = [
        f"Original {int(value)}"
        for value in top_flows["original_cluster"].tolist()
    ]
    target_labels = [
        f"Chunk {int(value)}"
        for value in top_flows["chunk_cluster"].tolist()
    ]
    node_labels = list(dict.fromkeys(source_labels + target_labels))
    node_index = {
        label: index for index, label in enumerate(node_labels)
    }

    figure = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={
                    "label": node_labels,
                    "pad": 18,
                    "thickness": 18,
                },
                link={
                    "source": [
                        node_index[label] for label in source_labels
                    ],
                    "target": [
                        node_index[label] for label in target_labels
                    ],
                    "value": top_flows["weighted_flow"].tolist(),
                    "customdata": top_flows[
                        "contributing_speech_n"
                    ].tolist(),
                    "hovertemplate": (
                        "%{source.label} → %{target.label}<br>"
                        "Weighted flow: %{value:.3f}<br>"
                        "Contributing speeches: %{customdata}"
                        "<extra></extra>"
                    ),
                },
            )
        ]
    )

    figure.update_layout(
        title=(
            (
                f"Top {SANKEY_TOP_N_FLOWS} Non-retention Cluster Flows "
                "after Chunking"
            )
            if SHOW_TITLES
            else None
        ),
        font_size=FONT_SIZE,
        height=650,
    )

    output_file = (
        OUTPUT_DIR
        / f"cluster_transition_sankey_top_{SANKEY_TOP_N_FLOWS}.html"
    )
    figure.write_html(
        output_file,
        include_plotlyjs=True,
        full_html=True,
    )
    print(f"Saved Sankey diagram: {output_file}")


# =============================================================================
# Original-category comparisons (all retained chunk_total >= MIN_CHUNK_TOTAL)
# =============================================================================

def summarize_metric_by_original_category(
    metrics: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:
    """Summarize one speech-level metric by original LLM category."""
    rows: list[dict[str, object]] = []

    for category, group in metrics.groupby(
        "original_category",
        observed=True,
    ):
        clean = pd.to_numeric(group[variable], errors="coerce").dropna()
        n = int(len(clean))

        if n == 0:
            continue

        mean = float(clean.mean())
        std = float(clean.std(ddof=1)) if n > 1 else np.nan
        se = std / np.sqrt(n) if n > 1 else np.nan
        margin = 1.96 * se if n > 1 else np.nan

        rows.append(
            {
                "original_category": category,
                "speech_n": n,
                f"mean_{variable}": mean,
                f"std_{variable}": std,
                f"se_{variable}": se,
                "ci95_low": (
                    max(0.0, mean - margin)
                    if n > 1 else np.nan
                ),
                "ci95_high": (
                    min(1.0, mean + margin)
                    if n > 1 else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_kmeans_entropy_by_original_category(
    metrics: pd.DataFrame,
) -> None:
    """Plot mean adjusted K-means entropy by original speech category."""
    summary = summarize_metric_by_original_category(
        metrics,
        "cluster_entropy_adjusted",
    )
    summary = summary.sort_values(
        "mean_cluster_entropy_adjusted",
        ascending=True,
    )

    save_table(
        summary,
        "kmeans_entropy_by_original_category.csv",
    )

    x = summary["mean_cluster_entropy_adjusted"]
    xerr = np.vstack(
        [
            x - summary["ci95_low"],
            summary["ci95_high"] - x,
        ]
    )

    fig, axis = plt.subplots(
        figsize=(12, max(7, 0.62 * len(summary)))
    )
    y = np.arange(len(summary))

    axis.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        capsize=4,
        linewidth=1.4,
    )
    axis.set_yticks(y)
    axis.set_yticklabels(summary["original_category"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Mean adjusted K-means within-speech entropy")
    axis.set_ylabel("Original speech category")
    axis.grid(axis="x", alpha=0.3)

    for y_pos, (_, row) in enumerate(summary.iterrows()):
        axis.text(
            min(
                0.99,
                row["mean_cluster_entropy_adjusted"] + 0.025,
            ),
            y_pos,
            f"n={int(row['speech_n']):,}",
            va="center",
            fontsize=9,
        )

    set_title(
        axis,
        "Adjusted K-means Entropy by Original LLM Category\n"
        "Points are category means; error bars are 95% confidence intervals",
    )
    fig.tight_layout()
    save_figure(fig, "kmeans_entropy_by_original_category.png")


def plot_cluster_retention_by_original_category(
    metrics: pd.DataFrame,
) -> None:
    """Plot mean K-means cluster retention by original speech category."""
    summary = summarize_metric_by_original_category(
        metrics,
        "cluster_retention",
    )
    summary = summary.sort_values(
        "mean_cluster_retention",
        ascending=True,
    )

    save_table(
        summary,
        "cluster_retention_by_original_category.csv",
    )

    x = summary["mean_cluster_retention"]
    xerr = np.vstack(
        [
            x - summary["ci95_low"],
            summary["ci95_high"] - x,
        ]
    )

    fig, axis = plt.subplots(
        figsize=(12, max(7, 0.62 * len(summary)))
    )
    y = np.arange(len(summary))

    axis.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        capsize=4,
        linewidth=1.4,
    )
    axis.set_yticks(y)
    axis.set_yticklabels(summary["original_category"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Mean share of chunks retaining original K-means cluster")
    axis.set_ylabel("Original speech category")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="x", alpha=0.3)

    for y_pos, (_, row) in enumerate(summary.iterrows()):
        axis.text(
            min(0.99, row["mean_cluster_retention"] + 0.025),
            y_pos,
            f"n={int(row['speech_n']):,}",
            va="center",
            fontsize=9,
        )

    set_title(
        axis,
        "K-means Cluster Retention by Original LLM Category\n"
        "Points are category means; error bars are 95% confidence intervals",
    )
    fig.tight_layout()
    save_figure(fig, "cluster_retention_by_original_category.png")


def create_global_fragmentation_outputs(
    original: pd.DataFrame,
    chunks: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Run all non-interval-specific fragmentation diagnostics once."""
    flow = build_cluster_transition_long(original, chunks)
    save_table(flow, "cluster_transition_weighted_all_speeches.csv")

    plot_cluster_net_inflow_outflow(flow)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Original input: {ORIGINAL_INPUT_FILE}")
    print(f"Chunk input: {CHUNK_INPUT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")

    original, chunks, diagnostics, filter_summary = load_data()

    save_table(diagnostics, "chunk_weight_diagnostics.csv")
    save_table(filter_summary, "sample_filter_summary.csv")
    save_table(
        pd.DataFrame(
            [
                {
                    "interval_label": label,
                    "chunk_min_inclusive": lower,
                    "chunk_max_exclusive": (
                        upper if upper is not None else "inf"
                    ),
                }
                for lower, upper, label in TRANSITION_INTERVALS
            ]
        ),
        "transition_interval_definitions.csv",
    )

    metrics = build_speech_metrics(original, chunks)
    save_table(metrics, "speech_level_entropy_retention.csv")

    entropy_summary = summarize_exact(
        metrics,
        [
            "category_entropy_raw",
            "category_entropy_adjusted",
            "cluster_entropy_raw",
            "cluster_entropy_adjusted",
        ],
    )
    save_table(
        entropy_summary,
        "entropy_summary_by_exact_chunk_total.csv",
    )

    retention_summary = summarize_exact(
        metrics,
        [
            "category_retention",
            "cluster_retention",
        ],
    )
    save_table(
        retention_summary,
        "retention_summary_by_exact_chunk_total.csv",
    )

    plot_entropy(entropy_summary)
    plot_retention(retention_summary)
    create_transition_outputs(original, chunks)
    create_global_fragmentation_outputs(original, chunks, metrics)

    print("=" * 80)
    print("Analysis completed successfully.")
    print(f"Retained speeches: {len(original):,}")
    print(f"Retained chunk rows: {len(chunks):,}")
    print(f"Figure titles enabled: {SHOW_TITLES}")


if __name__ == "__main__":
    main()
