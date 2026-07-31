"""Plot score and category position effects separately by chunk_total.

Run from the project root with::

    python src/pic_position.py

The script compares each chunk judgment with the full-speech judgment for the
same ``basepk``.  It uses only chunk_total values 2 through 13 and produces
two 3-by-4 panel figures, one for procedural-score distance and one for
category mismatch rate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_FILE = PROJECT_ROOT / "output" / "Speech_sample.csv"
CHUNK_FILE = PROJECT_ROOT / "output" / "Speech_chunk.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pic" / "LLM_position"

CHUNK_TOTALS = list(range(2, 14))
BOOTSTRAP_REPS = 500
RANDOM_SEED = 20260731
FIGURE_DPI = 300

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def load_data() -> pd.DataFrame:
    """Load only non-overlapping fields needed for the position analysis."""
    full = pd.read_csv(
        FULL_FILE,
        usecols=["basepk", "category", "procedural_score"],
        low_memory=False,
    )
    if full["basepk"].duplicated().any():
        raise ValueError("Speech_sample.csv contains duplicate basepk values.")

    full_reference = full[
        ["basepk", "category", "procedural_score"]
    ].rename(
        columns={
            "category": "full_category",
            "procedural_score": "full_score",
        }
    )

    chunks = pd.read_csv(
        CHUNK_FILE,
        usecols=[
            "basepk",
            "chunk_index",
            "chunk_total",
            "category",
            "procedural_score",
        ],
        low_memory=False,
    )
    chunk_reference = chunks[
        [
            "basepk",
            "chunk_index",
            "chunk_total",
            "category",
            "procedural_score",
        ]
    ].rename(
        columns={
            "category": "chunk_category",
            "procedural_score": "chunk_score",
        }
    )

    data = chunk_reference.merge(
        full_reference,
        on="basepk",
        how="inner",
        validate="many_to_one",
    )

    expected_columns = {
        "basepk",
        "chunk_index",
        "chunk_total",
        "chunk_category",
        "chunk_score",
        "full_category",
        "full_score",
    }
    if not expected_columns.issubset(data.columns):
        missing = sorted(expected_columns - set(data.columns))
        raise ValueError(f"Merged data is missing required columns: {missing}")
    if any(column.endswith(("_x", "_y")) for column in data.columns):
        raise ValueError("Unexpected suffix columns found after the merge.")

    for column in ["chunk_index", "chunk_total", "chunk_score", "full_score"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(
        subset=["chunk_index", "chunk_total", "chunk_score", "full_score"]
    ).copy()
    data = data.loc[data["chunk_total"].isin(CHUNK_TOTALS)].copy()
    data["chunk_total"] = data["chunk_total"].astype(int)
    data["chunk_index"] = data["chunk_index"].astype(int)

    data["relative_position"] = (
        data["chunk_index"] + 0.5
    ) / data["chunk_total"]
    data["score_distance"] = (
        data["chunk_score"] - data["full_score"]
    ).abs()

    full_category = data["full_category"].fillna("").astype(str).str.strip()
    chunk_category = data["chunk_category"].fillna("").astype(str).str.strip()
    data["category_distance"] = (
        chunk_category != full_category
    ).astype(float)
    data["category_valid"] = full_category.ne("") & chunk_category.ne("")

    return data


def bootstrap_mean(
    data: pd.DataFrame,
    outcome: str,
) -> tuple[float, float, float, int]:
    """Return mean and speech-level bootstrap 95% CI for one position."""
    speech_values = data.groupby("basepk")[outcome].mean()
    values = speech_values.to_numpy(dtype=float)
    n_speeches = len(values)
    if n_speeches == 0:
        return np.nan, np.nan, np.nan, 0

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.integers(
        low=0,
        high=n_speeches,
        size=(BOOTSTRAP_REPS, n_speeches),
    )
    bootstrap_means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
        n_speeches,
    )


def summarize_positions(
    data: pd.DataFrame,
    outcome: str,
    category_only: bool = False,
) -> pd.DataFrame:
    """Summarize every actual chunk position within every chunk_total."""
    rows: list[dict] = []
    for chunk_total in CHUNK_TOTALS:
        total_data = data.loc[data["chunk_total"] == chunk_total]
        if category_only:
            total_data = total_data.loc[total_data["category_valid"]]

        for chunk_index in range(chunk_total):
            group = total_data.loc[total_data["chunk_index"] == chunk_index]
            mean, ci_low, ci_high, n_speeches = bootstrap_mean(group, outcome)
            rows.append({
                "chunk_total": chunk_total,
                "chunk_index": chunk_index,
                "relative_position": (chunk_index + 0.5) / chunk_total,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_speeches": n_speeches,
                "n_chunks": len(group),
            })
    return pd.DataFrame(rows)


def plot_score_figure(summary: pd.DataFrame) -> None:
    """Create the 12-panel procedural-score distance figure."""
    y_max = float(summary["ci_high"].max()) * 1.05
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(16, 11),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for axis, chunk_total in zip(axes.flat, CHUNK_TOTALS):
        subset = summary.loc[summary["chunk_total"] == chunk_total]
        x = subset["relative_position"].to_numpy()
        axis.plot(x, subset["mean"], marker="o", linewidth=1.8)
        axis.fill_between(
            x,
            subset["ci_low"],
            subset["ci_high"],
            alpha=0.2,
        )
        n_speeches = int(subset["n_speeches"].max())
        axis.set_title(f"The number of chunks per speech: {chunk_total}")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, y_max)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(alpha=0.25)

    fig.supxlabel("Relative position within speech")
    fig.supylabel("Mean absolute distance from full-speech score")
    fig.savefig(
        OUTPUT_DIR / "fig_01_score_position_by_chunk_total.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_category_figure(summary: pd.DataFrame) -> None:
    """Create the 12-panel category mismatch-rate figure."""
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(16, 11),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for axis, chunk_total in zip(axes.flat, CHUNK_TOTALS):
        subset = summary.loc[summary["chunk_total"] == chunk_total]
        x = subset["relative_position"].to_numpy()
        axis.plot(x, subset["mean"], marker="o", linewidth=1.8)
        axis.fill_between(
            x,
            subset["ci_low"],
            subset["ci_high"],
            alpha=0.2,
        )
        n_speeches = int(subset["n_speeches"].max())
        axis.set_title(f"The number of chunks per speech: {chunk_total}")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(alpha=0.25)

    fig.supxlabel("Relative position within speech")
    fig.supylabel("Category mismatch rate")
    fig.savefig(
        OUTPUT_DIR / "fig_02_category_position_by_chunk_total.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    score_summary = summarize_positions(data, "score_distance")
    category_summary = summarize_positions(
        data,
        "category_distance",
        category_only=True,
    )
    plot_score_figure(score_summary)
    plot_category_figure(category_summary)

    print(f"Analyzed {data['basepk'].nunique():,} speeches")
    print(f"Analyzed {len(data):,} chunks")
    print(f"chunk_total levels: {CHUNK_TOTALS}")
    print("Generated two 12-panel position figures.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
