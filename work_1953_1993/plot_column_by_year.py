"""Plot a selected column over time."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


# =============================================================================
# User settings
# =============================================================================

# Change this value to the column that you want to plot.
VALUE_COLUMN = "party_at_date"


# =============================================================================
# Fixed settings
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "work_1953_1993"
    / "output"
    / "Kmeans"
    / "speeches_with_parlinfo_kmeans.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "work_1953_1993" / "output" / "temp_pic"

YEAR_COLUMN = "year"
CSV_SEPARATOR = ","


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    """Load the fixed year column and the selected value column."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file was not found: {INPUT_FILE}")

    header = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        nrows=0,
    )

    required_columns = [YEAR_COLUMN, VALUE_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in header.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columns were not found in the input file: {missing_columns}"
        )

    data = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        usecols=required_columns,
        low_memory=False,
    )

    data[YEAR_COLUMN] = pd.to_numeric(
        data[YEAR_COLUMN],
        errors="coerce",
    )
    data = data.dropna(subset=[YEAR_COLUMN, VALUE_COLUMN]).copy()
    data[YEAR_COLUMN] = data[YEAR_COLUMN].astype(int)

    if data.empty:
        raise ValueError(
            f"No valid data remained for column: {VALUE_COLUMN}"
        )

    return data


def is_numeric_column(data):
    """Return True when every non-missing value can be read as a number."""
    numeric_values = pd.to_numeric(
        data[VALUE_COLUMN],
        errors="coerce",
    )

    return numeric_values.notna().all()


# =============================================================================
# Plotting
# =============================================================================

def plot_numeric_column(data):
    """Plot the yearly mean of a numeric column."""
    data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN])

    yearly_values = (
        data.groupby(YEAR_COLUMN, as_index=False)[VALUE_COLUMN]
        .mean()
        .sort_values(YEAR_COLUMN)
    )

    figure, axis = plt.subplots(figsize=(12, 6))

    axis.plot(
        yearly_values[YEAR_COLUMN],
        yearly_values[VALUE_COLUMN],
        linewidth=2,
        marker="o",
        markersize=4,
    )

    axis.set_title(f"Yearly Mean of {VALUE_COLUMN}")
    axis.set_xlabel("Year")
    axis.set_ylabel(f"Mean of {VALUE_COLUMN}")

    return figure, axis


def plot_categorical_column(data):
    """Plot each category's yearly share for a non-numeric column."""
    data[VALUE_COLUMN] = data[VALUE_COLUMN].astype(str).str.strip()
    data = data[data[VALUE_COLUMN].ne("")].copy()

    yearly_counts = (
        data.groupby([YEAR_COLUMN, VALUE_COLUMN])
        .size()
        .rename("count")
        .reset_index()
    )

    yearly_totals = (
        data.groupby(YEAR_COLUMN)
        .size()
        .rename("year_total")
        .reset_index()
    )

    yearly_shares = yearly_counts.merge(
        yearly_totals,
        on=YEAR_COLUMN,
        how="left",
        validate="many_to_one",
    )
    yearly_shares["share"] = (
        yearly_shares["count"]
        / yearly_shares["year_total"]
    )

    share_table = yearly_shares.pivot(
        index=YEAR_COLUMN,
        columns=VALUE_COLUMN,
        values="share",
    ).fillna(0)

    share_table = share_table.sort_index()

    figure, axis = plt.subplots(figsize=(14, 7))

    for category in share_table.columns:
        axis.plot(
            share_table.index,
            share_table[category],
            linewidth=2,
            marker="o",
            markersize=3,
            label=str(category),
        )

    axis.set_title(f"Yearly Share of {VALUE_COLUMN}")
    axis.set_xlabel("Year")
    axis.set_ylabel("Share")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axis.legend(
        title=VALUE_COLUMN,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    return figure, axis


def make_safe_filename(column_name):
    """Replace characters that cannot be used in Windows filenames."""
    invalid_characters = '<>:"/\\|?*'
    safe_name = column_name

    for character in invalid_characters:
        safe_name = safe_name.replace(character, "_")

    return safe_name


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    if is_numeric_column(data):
        figure, axis = plot_numeric_column(data)
        plot_type = "numeric yearly mean"
    else:
        figure, axis = plot_categorical_column(data)
        plot_type = "categorical yearly share"

    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    output_file = (
        OUTPUT_DIR
        / f"{make_safe_filename(VALUE_COLUMN)}.png"
    )
    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Plot type: {plot_type}")
    print(f"Saved figure to: {output_file}")


if __name__ == "__main__":
    main()
