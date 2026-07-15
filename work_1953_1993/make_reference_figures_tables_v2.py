# -*- coding: utf-8 -*-
r"""
Generate figures and tables for the 1963–1993 Canadian parliamentary speeches project.

Put this file in:
    work_1953_1993\make_reference_figures_tables.py

Default input:
    work_1953_1993\output\Speeches_final.csv

Default output folder:
    work_1953_1993\output\figures_tables_llm_kmeans\

Run in terminal from work_1953_1993:
    python make_reference_figures_tables.py

Or specify paths manually:
    python make_reference_figures_tables.py --input output\Speeches_final.csv --output-dir output\figures_tables_llm_kmeans
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# =========================
# 1. Configurable settings
# =========================

PROCEDURE_CATEGORY = "Parliamentary Procedure"
TOP_N_CATEGORIES = 10
HIGH_SCORE_THRESHOLD = 0.75

# Important years from the reference-paper logic.
# You can add or delete years here.
EVENT_YEARS = {
    1968: "Budget / committee reforms",
    1982: "Lefebvre reform",
    1986: "McGrath reform",
}

# Periods used for summary tables.
# The end year is inclusive.
PERIODS = [
    (1963, 1968, "1963-1968"),
    (1969, 1981, "1969-1981"),
    (1982, 1986, "1982-1986"),
    (1987, 1993, "1987-1993"),
]

FIG_EXTENSIONS = ["png"]  # default: save PNG only; use --save-pdf to also save PDF


# =========================
# 2. Helper functions
# =========================

def safe_filename(text: str, max_len: int = 90) -> str:
    """Convert category/party names to safe file names."""
    text = str(text).strip().lower()
    text = re.sub(r"[&/\\]+", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if text else "unknown"


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save CSV with utf-8-sig so Excel can open it cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    """Save figure in all requested formats."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in FIG_EXTENSIONS:
        fig.savefig(fig_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def add_event_lines(ax: plt.Axes, year_min: int, year_max: int) -> None:
    """Add vertical reference lines for institutional reform years."""
    ymin, ymax = ax.get_ylim()
    if np.isfinite(ymax):
        label_y = ymax - (ymax - ymin) * 0.03
    else:
        label_y = 0

    for year, label in EVENT_YEARS.items():
        if year_min <= year <= year_max:
            ax.axvline(year, linestyle="--", linewidth=1, alpha=0.7)
            ax.text(
                year + 0.1,
                label_y,
                str(year),
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
            )


def percent_axis(ax: plt.Axes, axis: str = "y") -> None:
    formatter = FuncFormatter(lambda value, _: f"{value:.0f}%")
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def auto_ylim_from_series(ax: plt.Axes, values: Iterable[float], lower_zero: bool = True) -> None:
    """
    Set a readable y-axis without forcing it to 0-1 or 0-100.
    For share figures, lower bound is usually 0, but upper bound follows the data.
    """
    values = pd.Series(list(values)).dropna()
    if values.empty:
        return
    ymin = values.min()
    ymax = values.max()
    if ymax == ymin:
        pad = max(abs(ymax) * 0.1, 1)
    else:
        pad = (ymax - ymin) * 0.15
    lower = 0 if lower_zero and ymin >= 0 else ymin - pad
    upper = ymax + pad
    ax.set_ylim(lower, upper)


def get_word_count_column(df: pd.DataFrame) -> str:
    for col in ["speechtext_word_count", "speech_length_words"]:
        if col in df.columns:
            return col
    raise ValueError("Cannot find word-count column. Expected 'speechtext_word_count' or 'speech_length_words'.")


def assign_period(year: int) -> str:
    for start, end, label in PERIODS:
        if start <= year <= end:
            return label
    return "outside_periods"


def classify_speaker_position(value: object) -> str:
    """
    Coarse speaker-position grouping.
    This is intentionally simple and transparent; adjust it if your speakerposition coding changes.
    """
    if pd.isna(value) or str(value).strip() == "":
        return "Unknown"

    s = str(value).lower()

    # Put Speaker/Chair first, because many entries explicitly contain chair titles.
    if "speaker" in s or "chair" in s or "chairman" in s:
        return "Speaker / Chair"
    if "prime minister" in s:
        return "Prime Minister"
    if "leader of the opposition" in s or "opposition leader" in s:
        return "Opposition Leader"
    if "house leader" in s or "whip" in s:
        return "House Leader / Whip"
    if "parliamentary secretary" in s:
        return "Parliamentary Secretary"
    if (
        "minister" in s
        or "secretary of state" in s
        or "president of the privy council" in s
        or "solicitor general" in s
        or "postmaster general" in s
    ):
        return "Minister / Cabinet"

    return "Other / Backbench"


def normalize_party_name(value: object) -> str:
    """Map detailed party strings to a few stable party families."""
    if pd.isna(value) or str(value).strip() == "":
        return "Unknown"
    s = str(value).lower()
    if "liberal" in s:
        return "Liberal"
    if "conservative" in s or "progressive conservative" in s or s.strip() in {"pc", "cpc"}:
        return "Conservative/Progressive Conservative"
    if "new democratic" in s or "n.d.p" in s or "ndp" in s:
        return "NDP"
    if "bloc" in s or "québécois" in s or "quebecois" in s:
        return "Bloc Québécois"
    if "reform" in s:
        return "Reform"
    if "independent" in s:
        return "Independent"
    if "no affiliation" in s or "recognised" in s or "recognized" in s:
        return "No affiliation"
    return str(value).strip()


def government_party_for_date(dt: pd.Timestamp, year: int) -> str:
    """
    Approximate Canadian federal governing party for 1963-1993 speeches.
    Date cut-points use changes in ministry after federal elections.
    If a precise date is unavailable, the year-level fallback is used.
    """
    if pd.isna(dt):
        # Year-level fallback. This is less precise for 1979, 1980, 1984, and 1993.
        if 1963 <= year <= 1978:
            return "Liberal"
        if year == 1979:
            return "Conservative/Progressive Conservative"
        if 1980 <= year <= 1983:
            return "Liberal"
        if 1984 <= year <= 1993:
            return "Conservative/Progressive Conservative"
        return "Unknown"

    if pd.Timestamp(1963, 4, 22) <= dt < pd.Timestamp(1979, 6, 4):
        return "Liberal"
    if pd.Timestamp(1979, 6, 4) <= dt < pd.Timestamp(1980, 3, 3):
        return "Conservative/Progressive Conservative"
    if pd.Timestamp(1980, 3, 3) <= dt < pd.Timestamp(1984, 9, 17):
        return "Liberal"
    if pd.Timestamp(1984, 9, 17) <= dt < pd.Timestamp(1993, 11, 4):
        return "Conservative/Progressive Conservative"
    if pd.Timestamp(1993, 11, 4) <= dt < pd.Timestamp(2006, 2, 6):
        return "Liberal"
    return "Unknown"


def official_opposition_party_for_date(dt: pd.Timestamp, year: int) -> str:
    """Approximate official opposition party for the same period."""
    if pd.isna(dt):
        if 1963 <= year <= 1978:
            return "Conservative/Progressive Conservative"
        if year == 1979:
            return "Liberal"
        if 1980 <= year <= 1983:
            return "Conservative/Progressive Conservative"
        if 1984 <= year <= 1993:
            return "Liberal"
        return "Unknown"

    if pd.Timestamp(1963, 4, 22) <= dt < pd.Timestamp(1979, 6, 4):
        return "Conservative/Progressive Conservative"
    if pd.Timestamp(1979, 6, 4) <= dt < pd.Timestamp(1980, 3, 3):
        return "Liberal"
    if pd.Timestamp(1980, 3, 3) <= dt < pd.Timestamp(1984, 9, 17):
        return "Conservative/Progressive Conservative"
    if pd.Timestamp(1984, 9, 17) <= dt < pd.Timestamp(1993, 11, 4):
        return "Liberal"
    if pd.Timestamp(1993, 11, 4) <= dt < pd.Timestamp(1997, 6, 2):
        return "Bloc Québécois"
    return "Unknown"


def classify_government_status(row: pd.Series) -> str:
    """Classify each speech by government/opposition status rather than party name."""
    party = row.get("party_family", "Unknown")
    gov_party = row.get("government_party", "Unknown")
    official_opp = row.get("official_opposition_party", "Unknown")

    if party in {"Unknown", "Independent", "No affiliation"}:
        return "Independent / no affiliation"
    if gov_party != "Unknown" and party == gov_party:
        return "Government party"
    if official_opp != "Unknown" and party == official_opp:
        return "Official opposition"
    return "Other opposition parties"


def weighted_category_share_by_year(df: pd.DataFrame, category_col: str, weight_col: str) -> pd.DataFrame:
    """Return year × category weighted shares in percent."""
    sums = df.groupby(["year", category_col], dropna=False)[weight_col].sum().unstack(fill_value=0)
    totals = df.groupby("year")[weight_col].sum()
    shares = sums.div(totals, axis=0) * 100
    return shares


def unweighted_category_share_by_year(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """Return year × category unweighted shares in percent."""
    counts = pd.crosstab(df["year"], df[category_col], normalize="index") * 100
    return counts


def plot_line_table(
    table: pd.DataFrame,
    columns: list[str],
    fig_dir: Path,
    stem: str,
    title: str,
    ylabel: str,
    year_min: int,
    year_max: int,
    percent: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for col in columns:
        if col in table.columns:
            ax.plot(table.index, table[col], marker="o", linewidth=1.8, markersize=3, label=col)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    ax.margins(x=0.01)
    values = []
    for col in columns:
        if col in table.columns:
            values.extend(table[col].dropna().tolist())
    auto_ylim_from_series(ax, values, lower_zero=percent)
    if percent:
        percent_axis(ax, "y")
    add_event_lines(ax, year_min, year_max)
    save_figure(fig, fig_dir, stem)


# =========================
# 3. Main analysis
# =========================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to Speeches_final.csv")
    parser.add_argument("--output-dir", type=str, default=None, help="Output folder for all figures and tables")
    parser.add_argument("--top-n", type=int, default=TOP_N_CATEGORIES, help="Number of top categories for small multiples")
    parser.add_argument("--threshold", type=float, default=HIGH_SCORE_THRESHOLD, help="High-score threshold")
    parser.add_argument("--save-pdf", action="store_true", help="Also save PDF copies of every figure")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    input_file = Path(args.input) if args.input else project_dir / "output" / "Speeches_final.csv"
    out_dir = Path(args.output_dir) if args.output_dir else project_dir / "output" / "figures_tables_llm_kmeans"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    category_fig_dir = fig_dir / "category_yearly_lines"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    if args.save_pdf:
        FIG_EXTENSIONS.append("pdf")

    plt.ioff()
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    print(f"Reading: {input_file}")

    # Read only the columns needed for these figures/tables.
    # This is much faster than loading the full speech text columns from a large CSV.
    header_cols = pd.read_csv(input_file, nrows=0).columns.tolist()
    useful_cols = {
        "year",
        "category",
        "procedural_score",
        "cluster_score",
        "disagreement",
        "speechtext_word_count",
        "speech_length_words",
        "cluster",
        "speakerposition",
        "party_simplified",
        "party_at_date",
        "speakerparty",
        "speech_date",
        "speechdate",
    }
    usecols = [col for col in header_cols if col in useful_cols]
    df = pd.read_csv(input_file, usecols=usecols, low_memory=False)

    required = ["year", "category", "procedural_score", "cluster_score"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    word_col = get_word_count_column(df)

    # Basic cleaning
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df["category"] = df["category"].fillna("Unknown")
    df[word_col] = pd.to_numeric(df[word_col], errors="coerce").fillna(0)
    df["procedural_score"] = pd.to_numeric(df["procedural_score"], errors="coerce")
    df["cluster_score"] = pd.to_numeric(df["cluster_score"], errors="coerce")

    if "disagreement" not in df.columns:
        df["disagreement"] = df["procedural_score"] - df["cluster_score"]
    else:
        df["disagreement"] = pd.to_numeric(df["disagreement"], errors="coerce")

    df["abs_disagreement"] = df["disagreement"].abs()
    df["is_procedure_category"] = df["category"].eq(PROCEDURE_CATEGORY)
    df["cluster_high"] = df["cluster_score"] >= args.threshold
    df["procedural_high"] = df["procedural_score"] >= args.threshold
    df["period"] = df["year"].apply(assign_period)

    # Date parsing for government/opposition-status analysis.
    # Prefer exact speech date where available; otherwise the government-status function falls back to year.
    date_source_col = None
    for candidate in ["speech_date", "speechdate"]:
        if candidate in df.columns:
            date_source_col = candidate
            break
    if date_source_col is not None:
        df["_speech_date_parsed"] = pd.to_datetime(df[date_source_col], errors="coerce")
    else:
        df["_speech_date_parsed"] = pd.NaT

    if "party_simplified" in df.columns:
        df["party_family"] = df["party_simplified"].apply(normalize_party_name)
    elif "party_at_date" in df.columns:
        df["party_family"] = df["party_at_date"].apply(normalize_party_name)
    elif "speakerparty" in df.columns:
        df["party_family"] = df["speakerparty"].apply(normalize_party_name)
    else:
        df["party_family"] = "Unknown"

    df["government_party"] = [
        government_party_for_date(dt, yr) for dt, yr in zip(df["_speech_date_parsed"], df["year"])
    ]
    df["official_opposition_party"] = [
        official_opposition_party_for_date(dt, yr) for dt, yr in zip(df["_speech_date_parsed"], df["year"])
    ]
    df["government_status"] = df.apply(classify_government_status, axis=1)

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    all_years = list(range(year_min, year_max + 1))

    # -------------------------
    # Table 00: basic summaries
    # -------------------------
    basic_summary = pd.DataFrame({
        "metric": [
            "n_speeches",
            "year_min",
            "year_max",
            "n_categories",
            "n_clusters" if "cluster" in df.columns else "n_clusters_column_missing",
            "word_count_column",
            "high_score_threshold",
        ],
        "value": [
            len(df),
            year_min,
            year_max,
            df["category"].nunique(dropna=False),
            df["cluster"].nunique(dropna=True) if "cluster" in df.columns else np.nan,
            word_col,
            args.threshold,
        ],
    })
    save_table(basic_summary, table_dir / "table_00_basic_summary.csv")

    # -----------------------------------
    # Table and Figure 01: procedure share
    # -----------------------------------
    yearly_total = df.groupby("year").size().reindex(all_years, fill_value=0)
    yearly_words = df.groupby("year")[word_col].sum().reindex(all_years, fill_value=0)

    proc_count = df.groupby("year")["is_procedure_category"].sum().reindex(all_years, fill_value=0)
    proc_words = (
        df.loc[df["is_procedure_category"]]
        .groupby("year")[word_col]
        .sum()
        .reindex(all_years, fill_value=0)
    )

    procedure_yearly = pd.DataFrame({
        "year": all_years,
        "procedure_count": proc_count.values,
        "total_count": yearly_total.values,
        "procedure_share_unweighted_pct": np.where(yearly_total.values > 0, proc_count.values / yearly_total.values * 100, np.nan),
        "procedure_words": proc_words.values,
        "total_words": yearly_words.values,
        "procedure_share_weighted_pct": np.where(yearly_words.values > 0, proc_words.values / yearly_words.values * 100, np.nan),
    })
    save_table(procedure_yearly, table_dir / "table_01_parliamentary_procedure_yearly_share.csv")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(procedure_yearly["year"], procedure_yearly["procedure_share_unweighted_pct"], marker="o", linewidth=1.8, markersize=3, label="Unweighted share")
    ax.plot(procedure_yearly["year"], procedure_yearly["procedure_share_weighted_pct"], marker="s", linewidth=1.8, markersize=3, label="Word-count weighted share")
    ax.set_title("Parliamentary Procedure Share over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of speeches / words")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    ax.margins(x=0.01)
    auto_ylim_from_series(
        ax,
        procedure_yearly[["procedure_share_unweighted_pct", "procedure_share_weighted_pct"]].to_numpy().ravel(),
        lower_zero=True,
    )
    percent_axis(ax, "y")
    add_event_lines(ax, year_min, year_max)
    save_figure(fig, fig_dir, "fig_01_parliamentary_procedure_share_weighted_unweighted")

    # ------------------------------------
    # Table 02: category yearly shares
    # ------------------------------------
    unweighted_shares = unweighted_category_share_by_year(df, "category").reindex(all_years)
    weighted_shares = weighted_category_share_by_year(df, "category", word_col).reindex(all_years)

    unweighted_out = unweighted_shares.reset_index().rename(columns={"index": "year"})
    weighted_out = weighted_shares.reset_index().rename(columns={"index": "year"})
    save_table(unweighted_out, table_dir / "table_02_category_yearly_share_unweighted_pct.csv")
    save_table(weighted_out, table_dir / "table_02_category_yearly_share_weighted_pct.csv")

    category_counts = df["category"].value_counts()
    top_categories = category_counts.head(args.top_n).index.tolist()

    def plot_category_small_multiples(share_table: pd.DataFrame, weighted_label: str, stem: str) -> None:
        n = len(top_categories)
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, max(3.2 * nrows, 5)), sharex=True)
        axes = np.array(axes).reshape(-1)
        for i, cat in enumerate(top_categories):
            ax = axes[i]
            if cat in share_table.columns:
                ax.plot(share_table.index, share_table[cat], marker="o", linewidth=1.5, markersize=2.5)
                ax.set_title(cat)
                ax.grid(True, axis="y", alpha=0.3)
                auto_ylim_from_series(ax, share_table[cat], lower_zero=True)
                percent_axis(ax, "y")
                add_event_lines(ax, year_min, year_max)
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"Top {args.top_n} Categories: Yearly Share ({weighted_label})", y=1.01, fontsize=14)
        fig.supxlabel("Year")
        fig.supylabel("Share")
        fig.tight_layout()
        save_figure(fig, fig_dir, stem)

    plot_category_small_multiples(
        unweighted_shares,
        "unweighted by speech count",
        "fig_02a_top_categories_yearly_share_unweighted_small_multiples",
    )
    plot_category_small_multiples(
        weighted_shares,
        "weighted by word count",
        "fig_02b_top_categories_yearly_share_weighted_small_multiples",
    )

    # Individual category trend plots, with category placed at the beginning of the filename.
    for cat in top_categories:
        safe_cat = safe_filename(cat)
        for table, label, suffix in [
            (unweighted_shares, "Unweighted share", "unweighted"),
            (weighted_shares, "Word-count weighted share", "weighted"),
        ]:
            if cat not in table.columns:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(table.index, table[cat], marker="o", linewidth=1.8, markersize=3)
            ax.set_title(f"{cat}: Yearly Share ({label})")
            ax.set_xlabel("Year")
            ax.set_ylabel("Share")
            ax.grid(True, axis="y", alpha=0.3)
            ax.margins(x=0.01)
            auto_ylim_from_series(ax, table[cat], lower_zero=True)
            percent_axis(ax, "y")
            add_event_lines(ax, year_min, year_max)
            save_figure(fig, category_fig_dir, f"category_{safe_cat}_yearly_share_{suffix}")

    # --------------------------------------
    # Table and Figures 03: score trajectories
    # --------------------------------------
    yearly_scores = (
        df.groupby("year", as_index=False)
        .agg(
            n_speeches=("category", "size"),
            mean_procedural_score=("procedural_score", "mean"),
            mean_cluster_score=("cluster_score", "mean"),
            mean_disagreement=("disagreement", "mean"),
            mean_abs_disagreement=("abs_disagreement", "mean"),
            procedural_high_share_pct=("procedural_high", lambda x: x.mean() * 100),
            cluster_high_share_pct=("cluster_high", lambda x: x.mean() * 100),
        )
        .sort_values("year")
    )
    save_table(yearly_scores, table_dir / "table_03_yearly_scores_and_disagreement.csv")

    scores_plot_table = yearly_scores.set_index("year")
    plot_line_table(
        scores_plot_table,
        ["mean_procedural_score", "mean_cluster_score"],
        fig_dir,
        "fig_03a_mean_procedural_and_cluster_score_by_year",
        "Mean Procedural Score and Cluster Score over Time",
        "Mean score",
        year_min,
        year_max,
        percent=False,
    )
    plot_line_table(
        scores_plot_table,
        ["mean_disagreement"],
        fig_dir,
        "fig_03b_mean_disagreement_by_year",
        "Mean Disagreement over Time: procedural_score - cluster_score",
        "Mean disagreement",
        year_min,
        year_max,
        percent=False,
    )
    plot_line_table(
        scores_plot_table,
        ["mean_abs_disagreement"],
        fig_dir,
        "fig_03c_mean_abs_disagreement_by_year",
        "Mean Absolute Disagreement over Time",
        "Mean absolute disagreement",
        year_min,
        year_max,
        percent=False,
    )

    # ----------------------------------------------------------
    # Table and Figure 04: cluster_score >= threshold by category
    # ----------------------------------------------------------
    cluster_high_by_category = (
        df.groupby("category", as_index=False)
        .agg(
            n_speeches=("category", "size"),
            cluster_high_count=("cluster_high", "sum"),
            cluster_high_share_pct=("cluster_high", lambda x: x.mean() * 100),
            mean_cluster_score=("cluster_score", "mean"),
            mean_procedural_score=("procedural_score", "mean"),
        )
        .sort_values("cluster_high_share_pct", ascending=False)
    )
    save_table(cluster_high_by_category, table_dir / "table_04_cluster_score_high_by_category.csv")

    plot_df = cluster_high_by_category.sort_values("cluster_high_share_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(plot_df))))
    ax.barh(plot_df["category"], plot_df["cluster_high_share_pct"])
    ax.set_title(f"Share of Speeches with cluster_score >= {args.threshold} by Category")
    ax.set_xlabel("Share")
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.3)
    percent_axis(ax, "x")
    xmax = plot_df["cluster_high_share_pct"].max()
    ax.set_xlim(0, xmax * 1.15 if xmax > 0 else 1)
    for y, value in enumerate(plot_df["cluster_high_share_pct"]):
        ax.text(value + xmax * 0.015, y, f"{value:.1f}%", va="center", fontsize=8)
    save_figure(fig, fig_dir, "fig_04_cluster_score_high_share_by_category")

    # -----------------------------------------
    # Table and Figures 05: four-quadrant logic
    # -----------------------------------------
    conditions = [
        df["procedural_high"] & df["cluster_high"],
        df["procedural_high"] & ~df["cluster_high"],
        ~df["procedural_high"] & df["cluster_high"],
    ]
    choices = ["Both high", "LLM/procedural high only", "K-means/cluster high only"]
    df["quadrant"] = np.select(conditions, choices, default="Neither high")
    quadrant_order = ["Both high", "LLM/procedural high only", "K-means/cluster high only", "Neither high"]

    quadrant_counts = (
        df["quadrant"]
        .value_counts()
        .reindex(quadrant_order, fill_value=0)
        .rename_axis("quadrant")
        .reset_index(name="n_speeches")
    )
    quadrant_counts["share_pct"] = quadrant_counts["n_speeches"] / quadrant_counts["n_speeches"].sum() * 100
    save_table(quadrant_counts, table_dir / "table_05a_quadrant_overall_counts.csv")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    plot_df = quadrant_counts.sort_values("share_pct", ascending=True)
    ax.barh(plot_df["quadrant"], plot_df["share_pct"])
    ax.set_title(f"LLM vs K-means High-Score Quadrants, threshold = {args.threshold}")
    ax.set_xlabel("Share of speeches")
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.3)
    percent_axis(ax, "x")
    xmax = plot_df["share_pct"].max()
    ax.set_xlim(0, xmax * 1.15 if xmax > 0 else 1)
    for y, value in enumerate(plot_df["share_pct"]):
        ax.text(value + xmax * 0.015, y, f"{value:.1f}%", va="center", fontsize=8)
    save_figure(fig, fig_dir, "fig_05a_quadrant_overall_share")

    quadrant_by_year_counts = pd.crosstab(df["year"], df["quadrant"]).reindex(all_years, fill_value=0)
    for q in quadrant_order:
        if q not in quadrant_by_year_counts.columns:
            quadrant_by_year_counts[q] = 0
    quadrant_by_year_counts = quadrant_by_year_counts[quadrant_order]
    quadrant_by_year_share = quadrant_by_year_counts.div(quadrant_by_year_counts.sum(axis=1), axis=0) * 100
    quadrant_by_year_out = quadrant_by_year_share.reset_index().rename(columns={"index": "year"})
    save_table(quadrant_by_year_out, table_dir / "table_05b_quadrant_by_year_share_pct.csv")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.stackplot(quadrant_by_year_share.index, [quadrant_by_year_share[q] for q in quadrant_order], labels=quadrant_order, alpha=0.85)
    ax.set_title(f"LLM vs K-means High-Score Quadrants over Time, threshold = {args.threshold}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of speeches")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.set_xlim(year_min, year_max)
    ax.set_ylim(0, 100)
    percent_axis(ax, "y")
    add_event_lines(ax, year_min, year_max)
    save_figure(fig, fig_dir, "fig_05b_quadrant_by_year_stacked_share")

    # A clearer version of the time trend: exclude "Neither high" and normalize only among high-score cases.
    high_quadrant_order = ["Both high", "LLM/procedural high only", "K-means/cluster high only"]
    high_counts = quadrant_by_year_counts[high_quadrant_order].copy()
    high_denominator = high_counts.sum(axis=1).replace(0, np.nan)
    high_share = high_counts.div(high_denominator, axis=0) * 100
    high_share_out = high_share.reset_index().rename(columns={"index": "year"})
    save_table(high_share_out, table_dir / "table_05c_high_quadrants_by_year_share_among_high_cases_pct.csv")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.stackplot(high_share.index, [high_share[q] for q in high_quadrant_order], labels=high_quadrant_order, alpha=0.85)
    ax.set_title(f"Composition of High-Score Cases over Time, threshold = {args.threshold}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share among high-score cases")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.set_xlim(year_min, year_max)
    ax.set_ylim(0, 100)
    percent_axis(ax, "y")
    add_event_lines(ax, year_min, year_max)
    save_figure(fig, fig_dir, "fig_05c_high_quadrants_by_year_stacked_share_among_high_cases")

    # --------------------------------------------------
    # Table and Figure 06: disagreement by category
    # --------------------------------------------------
    disagreement_by_category = (
        df.groupby("category", as_index=False)
        .agg(
            n_speeches=("category", "size"),
            mean_disagreement=("disagreement", "mean"),
            mean_abs_disagreement=("abs_disagreement", "mean"),
            median_abs_disagreement=("abs_disagreement", "median"),
            mean_procedural_score=("procedural_score", "mean"),
            mean_cluster_score=("cluster_score", "mean"),
        )
        .sort_values("mean_abs_disagreement", ascending=False)
    )
    save_table(disagreement_by_category, table_dir / "table_06_disagreement_by_category.csv")

    plot_df = disagreement_by_category.sort_values("mean_abs_disagreement", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(plot_df))))
    ax.barh(plot_df["category"], plot_df["mean_abs_disagreement"])
    ax.set_title("Mean Absolute Disagreement by Category")
    ax.set_xlabel("Mean absolute disagreement")
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.3)
    xmax = plot_df["mean_abs_disagreement"].max()
    ax.set_xlim(0, xmax * 1.15 if xmax > 0 else 1)
    for y, value in enumerate(plot_df["mean_abs_disagreement"]):
        ax.text(value + xmax * 0.015, y, f"{value:.3f}", va="center", fontsize=8)
    save_figure(fig, fig_dir, "fig_06_mean_abs_disagreement_by_category")

    # ------------------------------------------------------
    # Table and Figure 07: disagreement by speech length bin
    # ------------------------------------------------------
    # Use ranks to avoid qcut errors when many speeches have the same word count.
    q_labels = ["Q1 shortest", "Q2", "Q3", "Q4", "Q5 longest"]
    df["length_bin"] = pd.qcut(df[word_col].rank(method="first"), q=5, labels=q_labels)

    length_bin_table = (
        df.groupby("length_bin", observed=False)
        .agg(
            n_speeches=("category", "size"),
            min_words=(word_col, "min"),
            max_words=(word_col, "max"),
            mean_words=(word_col, "mean"),
            mean_abs_disagreement=("abs_disagreement", "mean"),
            mean_disagreement=("disagreement", "mean"),
            procedure_category_share_pct=("is_procedure_category", lambda x: x.mean() * 100),
            procedural_high_share_pct=("procedural_high", lambda x: x.mean() * 100),
            cluster_high_share_pct=("cluster_high", lambda x: x.mean() * 100),
        )
        .reset_index()
    )
    length_bin_table["length_bin_label"] = length_bin_table.apply(
        lambda r: f"{r['length_bin']} ({int(r['min_words'])}-{int(r['max_words'])} words)", axis=1
    )
    save_table(length_bin_table, table_dir / "table_07_disagreement_by_speech_length_bin.csv")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(length_bin_table["length_bin_label"], length_bin_table["mean_abs_disagreement"], marker="o", linewidth=1.8)
    ax.set_title("Mean Absolute Disagreement by Speech Length Quintile")
    ax.set_xlabel("Speech length quintile")
    ax.set_ylabel("Mean absolute disagreement")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    auto_ylim_from_series(ax, length_bin_table["mean_abs_disagreement"], lower_zero=True)
    save_figure(fig, fig_dir, "fig_07_mean_abs_disagreement_by_speech_length_bin")

    # -----------------------------------------------------------
    # Table and Figure 08: coarse speaker position and procedures
    # -----------------------------------------------------------
    if "speakerposition" in df.columns:
        df["speakerposition_group"] = df["speakerposition"].apply(classify_speaker_position)
        position_table = (
            df.groupby("speakerposition_group", as_index=False)
            .agg(
                n_speeches=("category", "size"),
                procedure_category_share_pct=("is_procedure_category", lambda x: x.mean() * 100),
                mean_procedural_score=("procedural_score", "mean"),
                mean_cluster_score=("cluster_score", "mean"),
                cluster_high_share_pct=("cluster_high", lambda x: x.mean() * 100),
                mean_abs_disagreement=("abs_disagreement", "mean"),
            )
            .sort_values("procedure_category_share_pct", ascending=False)
        )
        save_table(position_table, table_dir / "table_08_speakerposition_group_procedure.csv")

        plot_df = position_table.sort_values("procedure_category_share_pct", ascending=True)
        fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.45 * len(plot_df))))
        ax.barh(plot_df["speakerposition_group"], plot_df["procedure_category_share_pct"])
        ax.set_title("Parliamentary Procedure Share by Speaker-Position Group")
        ax.set_xlabel("Share of speeches classified as Parliamentary Procedure")
        ax.set_ylabel("")
        ax.grid(True, axis="x", alpha=0.3)
        percent_axis(ax, "x")
        xmax = plot_df["procedure_category_share_pct"].max()
        ax.set_xlim(0, xmax * 1.15 if xmax > 0 else 1)
        for y, (_, row) in enumerate(plot_df.iterrows()):
            value = row["procedure_category_share_pct"]
            n = int(row["n_speeches"])
            ax.text(value + xmax * 0.015, y, f"{value:.1f}% (n={n:,})", va="center", fontsize=8)
        save_figure(fig, fig_dir, "fig_08_procedure_share_by_speakerposition_group")

    # ------------------------------------------------------
    # Table and Figure 09: party-year procedural score trends
    # ------------------------------------------------------
    if "party_simplified" in df.columns:
        party_counts = df["party_simplified"].fillna("Unknown").value_counts()
        top_parties = party_counts.head(5).index.tolist()
        party_df = df[df["party_simplified"].isin(top_parties)].copy()

        party_year_table = (
            party_df.groupby(["year", "party_simplified"], as_index=False)
            .agg(
                n_speeches=("category", "size"),
                mean_procedural_score=("procedural_score", "mean"),
                mean_cluster_score=("cluster_score", "mean"),
                procedure_category_share_pct=("is_procedure_category", lambda x: x.mean() * 100),
                cluster_high_share_pct=("cluster_high", lambda x: x.mean() * 100),
            )
            .sort_values(["party_simplified", "year"])
        )
        save_table(party_year_table, table_dir / "table_09_party_year_procedural_trends.csv")

        fig, ax = plt.subplots(figsize=(10, 5.8))
        for party in top_parties:
            sub = party_year_table[party_year_table["party_simplified"] == party]
            ax.plot(sub["year"], sub["mean_procedural_score"], marker="o", linewidth=1.6, markersize=3, label=party)
        ax.set_title("Mean Procedural Score by Party over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean procedural_score")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.margins(x=0.01)
        auto_ylim_from_series(ax, party_year_table["mean_procedural_score"], lower_zero=True)
        add_event_lines(ax, year_min, year_max)
        save_figure(fig, fig_dir, "fig_09_mean_procedural_score_by_party_year")

    # -----------------------------------------------------------------
    # Table and Figure 10: government/opposition-status procedural trends
    # -----------------------------------------------------------------
    government_status_order = [
        "Government party",
        "Official opposition",
        "Other opposition parties",
        "Independent / no affiliation",
    ]
    government_status_table = (
        df.groupby(["year", "government_status"], as_index=False)
        .agg(
            n_speeches=("category", "size"),
            mean_procedural_score=("procedural_score", "mean"),
            mean_cluster_score=("cluster_score", "mean"),
            procedure_category_share_pct=("is_procedure_category", lambda x: x.mean() * 100),
            procedural_high_share_pct=("procedural_high", lambda x: x.mean() * 100),
            cluster_high_share_pct=("cluster_high", lambda x: x.mean() * 100),
            mean_abs_disagreement=("abs_disagreement", "mean"),
        )
        .sort_values(["government_status", "year"])
    )
    save_table(government_status_table, table_dir / "table_13_government_status_year_trends.csv")

    government_status_overall = (
        df.groupby("government_status", as_index=False)
        .agg(
            n_speeches=("category", "size"),
            mean_procedural_score=("procedural_score", "mean"),
            mean_cluster_score=("cluster_score", "mean"),
            procedure_category_share_pct=("is_procedure_category", lambda x: x.mean() * 100),
            procedural_high_share_pct=("procedural_high", lambda x: x.mean() * 100),
            cluster_high_share_pct=("cluster_high", lambda x: x.mean() * 100),
            mean_abs_disagreement=("abs_disagreement", "mean"),
        )
    )
    government_status_overall["government_status"] = pd.Categorical(
        government_status_overall["government_status"], categories=government_status_order, ordered=True
    )
    government_status_overall = government_status_overall.sort_values("government_status")
    save_table(government_status_overall, table_dir / "table_14_government_status_overall_summary.csv")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for status in government_status_order:
        sub = government_status_table[government_status_table["government_status"] == status]
        if sub.empty:
            continue
        ax.plot(sub["year"], sub["mean_procedural_score"], marker="o", linewidth=1.6, markersize=3, label=status)
    ax.set_title("Mean Procedural Score by Government/Opposition Status over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean procedural_score")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.margins(x=0.01)
    auto_ylim_from_series(ax, government_status_table["mean_procedural_score"], lower_zero=True)
    add_event_lines(ax, year_min, year_max)
    save_figure(fig, fig_dir, "fig_10_mean_procedural_score_by_government_status_year")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    plot_df = government_status_overall.sort_values("procedure_category_share_pct", ascending=True)
    ax.barh(plot_df["government_status"].astype(str), plot_df["procedure_category_share_pct"])
    ax.set_title("Parliamentary Procedure Share by Government/Opposition Status")
    ax.set_xlabel("Share of speeches classified as Parliamentary Procedure")
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.3)
    percent_axis(ax, "x")
    xmax = plot_df["procedure_category_share_pct"].max()
    ax.set_xlim(0, xmax * 1.20 if xmax > 0 else 1)
    for y, (_, row) in enumerate(plot_df.iterrows()):
        value = row["procedure_category_share_pct"]
        n = int(row["n_speeches"])
        ax.text(value + xmax * 0.015, y, f"{value:.1f}% (n={n:,})", va="center", fontsize=8)
    save_figure(fig, fig_dir, "fig_11_procedure_share_by_government_status")

    # ------------------------------------------------
    # Table 10: period summaries for compact reporting
    # ------------------------------------------------
    period_df = df[df["period"] != "outside_periods"].copy()
    period_summary = (
        period_df.groupby("period", as_index=False)
        .agg(
            n_speeches=("category", "size"),
            total_words=(word_col, "sum"),
            procedure_category_share_pct=("is_procedure_category", lambda x: x.mean() * 100),
            mean_procedural_score=("procedural_score", "mean"),
            mean_cluster_score=("cluster_score", "mean"),
            cluster_high_share_pct=("cluster_high", lambda x: x.mean() * 100),
            procedural_high_share_pct=("procedural_high", lambda x: x.mean() * 100),
            mean_abs_disagreement=("abs_disagreement", "mean"),
        )
    )
    # Preserve configured period order
    period_order = [label for _, _, label in PERIODS]
    period_summary["period"] = pd.Categorical(period_summary["period"], categories=period_order, ordered=True)
    period_summary = period_summary.sort_values("period")
    save_table(period_summary, table_dir / "table_10_period_summary.csv")

    period_category_share = pd.crosstab(period_df["period"], period_df["category"], normalize="index") * 100
    period_category_share = period_category_share.reindex(period_order)
    save_table(period_category_share.reset_index(), table_dir / "table_11_period_category_share_unweighted_pct.csv")

    # Optional table for category × cluster, without drawing heatmap.
    if "cluster" in df.columns:
        category_cluster_share = pd.crosstab(df["category"], df["cluster"], normalize="index") * 100
        save_table(category_cluster_share.reset_index(), table_dir / "table_12_category_cluster_share_row_normalized_pct.csv")

    # README
    readme = out_dir / "README_outputs.md"
    readme.write_text(
        "# Figures and tables generated by make_reference_figures_tables.py\n\n"
        f"Input file: `{input_file}`\n\n"
        f"High-score threshold: `{args.threshold}`\n\n"
        "Main figures:\n"
        "- `fig_01_parliamentary_procedure_share_weighted_unweighted`: Parliamentary Procedure share, weighted vs unweighted.\n"
        "- `fig_02a` / `fig_02b`: top-category yearly shares.\n"
        "- `fig_03a`-`fig_03c`: procedural_score, cluster_score, and disagreement over time.\n"
        "- `fig_04`: share of cluster_score >= threshold by category.\n"
        "- `fig_05a` / `fig_05b`: LLM vs K-means high-score quadrants.\n"
        "- `fig_05c`: high-score quadrants over time after excluding Neither high and renormalizing.\n"
        "- `fig_06`: mean absolute disagreement by category.\n"
        "- `fig_07`: disagreement by speech length quintile.\n"
        "- `fig_08`: procedure share by speaker-position group, with group sample sizes.\n"
        "- `fig_09`: mean procedural_score by party over time, if party_simplified exists.\n"
        "- `fig_10` / `fig_11`: procedural trends by government/opposition status.\n\n"
        "Tables are saved in the `tables` subfolder as CSV files.\n",
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Output folder: {out_dir.resolve()}")
    print(f"Figures: {fig_dir.resolve()}")
    print(f"Tables: {table_dir.resolve()}")


if __name__ == "__main__":
    main()
