from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# User settings
# =============================================================================

YEAR_COLUMN = "year"
CATEGORY_COLUMN = "category"
PARTY_COLUMN = "party_simplified"

PROCEDURAL_SCORE_COLUMN = "procedural_score"
CLUSTER_SCORE_COLUMN = "cluster_score"

WEIGHT_COLUMN = "speechtext_word_count"

CSV_SEPARATOR = ","
FIGSIZE = (12, 6)
DPI = 300

# score >= 这个阈值，计算占比
HIGH_SCORE_THRESHOLD = 0.75

# 是否固定 weighted mean 图的 y 轴为 0-1
# 你现在不想固定，所以设为 False
SET_Y_AXIS_FROM_0_TO_1 = False

# 每个 category-year 至少要有多少条记录才保留
MIN_OBSERVATIONS_PER_YEAR = 1

SHOW_FIGURE = False
SHOW_GRID = True


# =============================================================================
# Government party background settings
# =============================================================================

GOVERNMENT_PERIODS = [
    {
        "party": "Progressive Conservative",
        "start": "1963-01-01",
        "end": "1963-04-21",
        "color": "tab:red",
    },
    {
        "party": "Liberal",
        "start": "1963-04-22",
        "end": "1979-06-03",
        "color": "tab:blue",
    },
    {
        "party": "Progressive Conservative",
        "start": "1979-06-04",
        "end": "1980-03-02",
        "color": "tab:red",
    },
    {
        "party": "Liberal",
        "start": "1980-03-03",
        "end": "1984-09-16",
        "color": "tab:blue",
    },
    {
        "party": "Progressive Conservative",
        "start": "1984-09-17",
        "end": "1993-11-03",
        "color": "tab:red",
    },
    {
        "party": "Liberal",
        "start": "1993-11-04",
        "end": "1993-12-31",
        "color": "tab:blue",
    },
]

SHOW_GOVERNMENT_BACKGROUND = True
GOVERNMENT_BACKGROUND_ALPHA = 0.08


# =============================================================================
# Paths
# =============================================================================

WORK_DIR = Path(__file__).resolve().parent

INPUT_FILE = WORK_DIR / "output" / "Speeches_final.csv"
BASE_OUTPUT_DIR = WORK_DIR / "output" / "temp_pic"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MAIN_OUTPUT_DIR = (
    BASE_OUTPUT_DIR
    / f"category_score_weighted_mean_and_high_share_{timestamp}"
)

SUBSET_OUTPUT_DIRS = {
    "1_all_data": MAIN_OUTPUT_DIR / "1_all_data",
    "2_liberal_only": MAIN_OUTPUT_DIR / "2_liberal_only",
    "3_conservative_progressive_conservative_only": (
        MAIN_OUTPUT_DIR / "3_conservative_progressive_conservative_only"
    ),
}

for subset_dir in SUBSET_OUTPUT_DIRS.values():
    (subset_dir / "weighted_mean_scores").mkdir(parents=True, exist_ok=True)
    (subset_dir / "weighted_share_score_ge_075").mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helper functions
# =============================================================================

def make_safe_filename(text):
    """Convert text into a safe filename for Windows."""
    safe_text = str(text).strip()

    invalid_characters = '<>:"/\\|?*'
    for character in invalid_characters:
        safe_text = safe_text.replace(character, "_")

    safe_text = safe_text.replace("\n", "_")
    safe_text = safe_text.replace("\r", "_")

    if safe_text == "":
        safe_text = "empty_category"

    return safe_text


def check_columns_exist(dataframe, required_columns):
    """Check whether all required columns exist."""
    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def date_to_decimal_year(date_string):
    """Convert YYYY-MM-DD into decimal year."""
    date = pd.to_datetime(date_string)
    year_start = pd.Timestamp(year=date.year, month=1, day=1)
    next_year_start = pd.Timestamp(year=date.year + 1, month=1, day=1)

    year_fraction = (
        (date - year_start).days
        / (next_year_start - year_start).days
    )

    return date.year + year_fraction


def add_government_background(axis):
    """Add semi-transparent background spans for governing parties."""
    if not SHOW_GOVERNMENT_BACKGROUND:
        return

    used_labels = set()

    for period in GOVERNMENT_PERIODS:
        start_year = date_to_decimal_year(period["start"])
        end_year = date_to_decimal_year(period["end"])

        party_label = period["party"]
        legend_label = party_label if party_label not in used_labels else None
        used_labels.add(party_label)

        axis.axvspan(
            start_year,
            end_year,
            color=period["color"],
            alpha=GOVERNMENT_BACKGROUND_ALPHA,
            label=legend_label,
            zorder=0,
        )


def clean_data(data):
    """Clean year, category, party, score, and weight columns."""
    cleaned = data.copy()

    cleaned[YEAR_COLUMN] = pd.to_numeric(
        cleaned[YEAR_COLUMN],
        errors="coerce",
    )

    cleaned[PROCEDURAL_SCORE_COLUMN] = pd.to_numeric(
        cleaned[PROCEDURAL_SCORE_COLUMN],
        errors="coerce",
    )

    cleaned[CLUSTER_SCORE_COLUMN] = pd.to_numeric(
        cleaned[CLUSTER_SCORE_COLUMN],
        errors="coerce",
    )

    cleaned[WEIGHT_COLUMN] = pd.to_numeric(
        cleaned[WEIGHT_COLUMN],
        errors="coerce",
    )

    cleaned[CATEGORY_COLUMN] = (
        cleaned[CATEGORY_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cleaned[PARTY_COLUMN] = (
        cleaned[PARTY_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cleaned = cleaned.dropna(subset=[YEAR_COLUMN]).copy()
    cleaned = cleaned[cleaned[CATEGORY_COLUMN] != ""].copy()

    cleaned = cleaned.dropna(subset=[WEIGHT_COLUMN]).copy()
    cleaned = cleaned[cleaned[WEIGHT_COLUMN] > 0].copy()

    cleaned[YEAR_COLUMN] = cleaned[YEAR_COLUMN].astype(int)

    return cleaned


def weighted_mean(values, weights):
    """Compute weighted mean, ignoring missing values."""
    valid = values.notna() & weights.notna() & (weights > 0)

    if valid.sum() == 0:
        return pd.NA

    return (values[valid] * weights[valid]).sum() / weights[valid].sum()


def weighted_share_score_ge_threshold(values, weights, threshold):
    """
    Weighted share of speeches whose score >= threshold.

    Formula:
        sum(weight_i for score_i >= threshold)
        /
        sum(weight_i for valid score_i)
    """
    valid = values.notna() & weights.notna() & (weights > 0)

    if valid.sum() == 0:
        return pd.NA

    high_score = values[valid] >= threshold

    return weights[valid][high_score].sum() / weights[valid].sum()


# =============================================================================
# Summary functions
# =============================================================================

def make_weighted_mean_yearly_summary(data):
    """Create category-year weighted mean summary."""
    rows = []

    grouped = data.groupby([CATEGORY_COLUMN, YEAR_COLUMN], sort=True)

    for (category, year), group in grouped:
        weights = group[WEIGHT_COLUMN]

        rows.append(
            {
                CATEGORY_COLUMN: category,
                YEAR_COLUMN: year,
                "weighted_mean_procedural_score": weighted_mean(
                    group[PROCEDURAL_SCORE_COLUMN],
                    weights,
                ),
                "weighted_mean_cluster_score": weighted_mean(
                    group[CLUSTER_SCORE_COLUMN],
                    weights,
                ),
                "count": len(group),
                "total_word_count": weights.sum(),
            }
        )

    yearly_summary = pd.DataFrame(rows)

    yearly_summary = yearly_summary.sort_values(
        [CATEGORY_COLUMN, YEAR_COLUMN]
    )

    return yearly_summary


def make_high_score_share_yearly_summary(data):
    """Create category-year weighted share summary for score >= threshold."""
    rows = []

    grouped = data.groupby([CATEGORY_COLUMN, YEAR_COLUMN], sort=True)

    for (category, year), group in grouped:
        weights = group[WEIGHT_COLUMN]

        procedural_valid = group[PROCEDURAL_SCORE_COLUMN].notna()
        cluster_valid = group[CLUSTER_SCORE_COLUMN].notna()

        procedural_high = (
            group[PROCEDURAL_SCORE_COLUMN] >= HIGH_SCORE_THRESHOLD
        )
        cluster_high = (
            group[CLUSTER_SCORE_COLUMN] >= HIGH_SCORE_THRESHOLD
        )

        rows.append(
            {
                CATEGORY_COLUMN: category,
                YEAR_COLUMN: year,

                "weighted_share_procedural_score_ge_075":
                    weighted_share_score_ge_threshold(
                        group[PROCEDURAL_SCORE_COLUMN],
                        weights,
                        HIGH_SCORE_THRESHOLD,
                    ),

                "weighted_share_cluster_score_ge_075":
                    weighted_share_score_ge_threshold(
                        group[CLUSTER_SCORE_COLUMN],
                        weights,
                        HIGH_SCORE_THRESHOLD,
                    ),

                "unweighted_share_procedural_score_ge_075":
                    procedural_high[procedural_valid].mean()
                    if procedural_valid.sum() > 0 else pd.NA,

                "unweighted_share_cluster_score_ge_075":
                    cluster_high[cluster_valid].mean()
                    if cluster_valid.sum() > 0 else pd.NA,

                "count": len(group),
                "procedural_valid_count": int(procedural_valid.sum()),
                "cluster_valid_count": int(cluster_valid.sum()),
                "procedural_high_count": int(
                    (procedural_valid & procedural_high).sum()
                ),
                "cluster_high_count": int(
                    (cluster_valid & cluster_high).sum()
                ),
                "total_word_count": weights.sum(),
            }
        )

    yearly_summary = pd.DataFrame(rows)

    yearly_summary = yearly_summary.sort_values(
        [CATEGORY_COLUMN, YEAR_COLUMN]
    )

    return yearly_summary


# =============================================================================
# Plot functions
# =============================================================================

def plot_weighted_mean_scores(data, output_dir, subset_label):
    """Plot weighted mean scores by year for each category."""
    print("-" * 80)
    print(f"Weighted mean scores | Subset: {subset_label}")
    print(f"Rows in subset: {len(data):,}")

    if data.empty:
        print(f"No data found for subset: {subset_label}")
        return

    yearly_summary = make_weighted_mean_yearly_summary(data)

    yearly_summary = yearly_summary[
        yearly_summary["count"] >= MIN_OBSERVATIONS_PER_YEAR
    ].copy()

    summary_file = output_dir / "yearly_category_weighted_mean_scores.csv"

    yearly_summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved weighted mean summary: {summary_file}")

    categories = (
        yearly_summary[CATEGORY_COLUMN]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    saved_count = 0

    for category in categories:
        category_data = yearly_summary[
            yearly_summary[CATEGORY_COLUMN] == category
        ].copy()

        if category_data.empty:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE)

        add_government_background(ax)

        ax.plot(
            category_data[YEAR_COLUMN],
            category_data["weighted_mean_procedural_score"],
            marker="o",
            linewidth=2,
            label="Weighted mean procedural_score",
            zorder=2,
        )

        ax.plot(
            category_data[YEAR_COLUMN],
            category_data["weighted_mean_cluster_score"],
            marker="s",
            linewidth=2,
            linestyle="--",
            label="Weighted mean cluster_score",
            zorder=2,
        )

        ax.set_title(
            "Yearly Word-count Weighted Mean of Scores\n"
            f"Subset: {subset_label}\n"
            f"Category: {category}"
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Weighted mean score")

        if SET_Y_AXIS_FROM_0_TO_1:
            ax.set_ylim(0, 1)

        if SHOW_GRID:
            ax.grid(True, alpha=0.3, zorder=1)

        ax.legend(loc="best", fontsize=9)

        total_count = int(category_data["count"].sum())
        total_words = int(category_data["total_word_count"].sum())

        ax.text(
            0.01,
            -0.24,
            (
                f"Total observations used: {total_count} | "
                f"Total word count used as weight: {total_words:,}"
            ),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

        fig.tight_layout()

        safe_category = make_safe_filename(category)
        output_filename = f"{safe_category}__weighted_mean_scores.png"
        output_path = output_dir / output_filename

        fig.savefig(
            output_path,
            dpi=DPI,
            bbox_inches="tight",
        )

        saved_count += 1
        print(f"Saved: {output_path}")

        if SHOW_FIGURE:
            plt.show()
        else:
            plt.close(fig)

    print(f"Done. Saved {saved_count} weighted mean figures.")


def plot_high_score_shares(data, output_dir, subset_label):
    """Plot weighted share of score >= threshold by year for each category."""
    print("-" * 80)
    print(f"High-score shares | Subset: {subset_label}")
    print(f"Rows in subset: {len(data):,}")

    if data.empty:
        print(f"No data found for subset: {subset_label}")
        return

    yearly_summary = make_high_score_share_yearly_summary(data)

    yearly_summary = yearly_summary[
        yearly_summary["count"] >= MIN_OBSERVATIONS_PER_YEAR
    ].copy()

    summary_file = output_dir / "yearly_category_share_score_ge_075.csv"

    yearly_summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved high-score share summary: {summary_file}")

    categories = (
        yearly_summary[CATEGORY_COLUMN]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    saved_count = 0

    for category in categories:
        category_data = yearly_summary[
            yearly_summary[CATEGORY_COLUMN] == category
        ].copy()

        if category_data.empty:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE)

        add_government_background(ax)

        ax.plot(
            category_data[YEAR_COLUMN],
            category_data["weighted_share_procedural_score_ge_075"],
            marker="o",
            linewidth=2,
            label="Weighted share: procedural_score >= 0.75",
            zorder=2,
        )

        ax.plot(
            category_data[YEAR_COLUMN],
            category_data["weighted_share_cluster_score_ge_075"],
            marker="s",
            linewidth=2,
            linestyle="--",
            label="Weighted share: cluster_score >= 0.75",
            zorder=2,
        )

        ax.set_title(
            "Yearly Word-count Weighted Share of High-score Speeches\n"
            f"Threshold: score >= {HIGH_SCORE_THRESHOLD}\n"
            f"Subset: {subset_label}\n"
            f"Category: {category}"
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Weighted share")

        # 注意：这里不再使用 ax.set_ylim(0, 1)
        # y 轴会根据每张图的数据自动调整范围。

        if SHOW_GRID:
            ax.grid(True, alpha=0.3, zorder=1)

        ax.legend(loc="best", fontsize=9)

        total_count = int(category_data["count"].sum())
        total_words = int(category_data["total_word_count"].sum())

        ax.text(
            0.01,
            -0.24,
            (
                f"Total observations used: {total_count} | "
                f"Total word count used as weight: {total_words:,}"
            ),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

        fig.tight_layout()

        safe_category = make_safe_filename(category)
        output_filename = f"{safe_category}__weighted_share_score_ge_075.png"
        output_path = output_dir / output_filename

        fig.savefig(
            output_path,
            dpi=DPI,
            bbox_inches="tight",
        )

        saved_count += 1
        print(f"Saved: {output_path}")

        if SHOW_FIGURE:
            plt.show()
        else:
            plt.close(fig)

    print(f"Done. Saved {saved_count} high-score share figures.")


def run_outputs_for_subset(data, subset_base_dir, subset_label):
    """Run both output types for one data subset."""
    weighted_mean_dir = subset_base_dir / "weighted_mean_scores"
    high_share_dir = subset_base_dir / "weighted_share_score_ge_075"

    plot_weighted_mean_scores(
        data=data,
        output_dir=weighted_mean_dir,
        subset_label=subset_label,
    )

    plot_high_score_shares(
        data=data,
        output_dir=high_share_dir,
        subset_label=subset_label,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Input file: {INPUT_FILE}")
    print(f"Main output directory: {MAIN_OUTPUT_DIR}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    usecols = [
        YEAR_COLUMN,
        CATEGORY_COLUMN,
        PARTY_COLUMN,
        PROCEDURAL_SCORE_COLUMN,
        CLUSTER_SCORE_COLUMN,
        WEIGHT_COLUMN,
    ]

    data = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        usecols=usecols,
        low_memory=False,
        encoding="utf-8-sig",
    )

    check_columns_exist(data, usecols)

    print(f"Rows loaded: {len(data):,}")

    data = clean_data(data)

    print(f"Rows after cleaning: {len(data):,}")

    print("Party values found:")
    print(data[PARTY_COLUMN].value_counts(dropna=False).head(30))

    print("Weight summary:")
    print(data[WEIGHT_COLUMN].describe())

    # -------------------------------------------------------------------------
    # 1. All data
    # -------------------------------------------------------------------------
    all_data = data.copy()

    run_outputs_for_subset(
        data=all_data,
        subset_base_dir=SUBSET_OUTPUT_DIRS["1_all_data"],
        subset_label="All data",
    )

    # -------------------------------------------------------------------------
    # 2. Liberal only
    # -------------------------------------------------------------------------
    liberal_data = data[
        data[PARTY_COLUMN] == "Liberal"
    ].copy()

    run_outputs_for_subset(
        data=liberal_data,
        subset_base_dir=SUBSET_OUTPUT_DIRS["2_liberal_only"],
        subset_label="party_simplified == Liberal",
    )

    # -------------------------------------------------------------------------
    # 3. Conservative / Progressive Conservative only
    # -------------------------------------------------------------------------
    conservative_data = data[
        data[PARTY_COLUMN].isin(
            [
                "Conservative/Progressive Conservative",
                "Conservative",
                "Progressive Conservative",
            ]
        )
    ].copy()

    run_outputs_for_subset(
        data=conservative_data,
        subset_base_dir=SUBSET_OUTPUT_DIRS[
            "3_conservative_progressive_conservative_only"
        ],
        subset_label=(
            "party_simplified == Conservative/Progressive Conservative"
        ),
    )

    print("=" * 80)
    print("All tasks completed.")
    print(f"All outputs are saved in: {MAIN_OUTPUT_DIR}")


if __name__ == "__main__":
    main()