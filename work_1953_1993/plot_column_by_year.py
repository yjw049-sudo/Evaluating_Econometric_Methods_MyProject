"""Plot two selected columns over time with two y-axes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


# =============================================================================
# User settings
# =============================================================================

# Change these two values to the columns that you want to plot.
LEFT_VALUE_COLUMN = "party_at_date"
RIGHT_VALUE_COLUMN = "your_second_column"


# =============================================================================
# Fixed settings
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "work_1953_1993"
    / "output"
    / "Speeches_final.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "work_1953_1993" / "output" / "temp_pic"

YEAR_COLUMN = "year"
CSV_SEPARATOR = ","


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    """Load the fixed year column and the two selected value columns."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file was not found: {INPUT_FILE}")

    header = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        nrows=0,
    )

    required_columns = [
        YEAR_COLUMN,
        LEFT_VALUE_COLUMN,
        RIGHT_VALUE_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in header.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columns were not found in the input file: {missing_columns}"
        )

    # Use set() to avoid reading the same column twice if the two columns are identical.
    use_columns = list(dict.fromkeys(required_columns))

    data = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        usecols=use_columns,
        low_memory=False,
    )

    data[YEAR_COLUMN] = pd.to_numeric(
        data[YEAR_COLUMN],
        errors="coerce",
    )

    data = data.dropna(subset=[YEAR_COLUMN]).copy()
    data[YEAR_COLUMN] = data[YEAR_COLUMN].astype(int)

    if data.empty:
        raise ValueError("No valid data remained after reading the year column.")

    return data


def is_numeric_column(data, value_column):
    """Return True when every non-missing value can be read as a number."""
    column_data = data[value_column].dropna()

    if column_data.empty:
        raise ValueError(f"No valid data remained for column: {value_column}")

    numeric_values = pd.to_numeric(
        column_data,
        errors="coerce",
    )

    return numeric_values.notna().all()


# =============================================================================
# Data preparation for plotting
# =============================================================================

def prepare_numeric_column(data, value_column):
    """Prepare yearly mean for a numeric column."""
    temp = data[[YEAR_COLUMN, value_column]].dropna().copy()

    temp[value_column] = pd.to_numeric(
        temp[value_column],
        errors="coerce",
    )

    temp = temp.dropna(subset=[value_column])

    if temp.empty:
        raise ValueError(f"No numeric data remained for column: {value_column}")

    yearly_values = (
        temp.groupby(YEAR_COLUMN, as_index=False)[value_column]
        .mean()
        .sort_values(YEAR_COLUMN)
    )

    yearly_values = yearly_values.set_index(YEAR_COLUMN)

    return yearly_values, "numeric"


def prepare_categorical_column(data, value_column):
    """Prepare yearly share for each category in a categorical column."""
    temp = data[[YEAR_COLUMN, value_column]].dropna().copy()

    temp[value_column] = temp[value_column].astype(str).str.strip()
    temp = temp[temp[value_column].ne("")].copy()

    if temp.empty:
        raise ValueError(f"No categorical data remained for column: {value_column}")

    yearly_counts = (
        temp.groupby([YEAR_COLUMN, value_column])
        .size()
        .rename("count")
        .reset_index()
    )

    yearly_totals = (
        temp.groupby(YEAR_COLUMN)
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
        columns=value_column,
        values="share",
    ).fillna(0)

    share_table = share_table.sort_index()

    return share_table, "categorical"


def prepare_column_for_plotting(data, value_column):
    """Prepare one column for plotting."""
    if is_numeric_column(data, value_column):
        return prepare_numeric_column(data, value_column)

    return prepare_categorical_column(data, value_column)


# =============================================================================
# Plotting
# =============================================================================

def plot_one_column(axis, plot_data, value_column, column_type, line_style):
    """Plot one prepared column on the given axis."""
    if column_type == "numeric":
        axis.plot(
            plot_data.index,
            plot_data[value_column],
            linewidth=2,
            marker="o",
            markersize=4,
            linestyle=line_style,
            label=value_column,
        )

        axis.set_ylabel(f"Mean of {value_column}")

    else:
        for category in plot_data.columns:
            axis.plot(
                plot_data.index,
                plot_data[category],
                linewidth=2,
                marker="o",
                markersize=3,
                linestyle=line_style,
                label=f"{value_column}: {category}",
            )

        axis.set_ylabel(f"Share of {value_column}")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))


def plot_two_columns(data):
    """Plot two selected columns over time using two y-axes."""
    left_data, left_type = prepare_column_for_plotting(
        data,
        LEFT_VALUE_COLUMN,
    )

    right_data, right_type = prepare_column_for_plotting(
        data,
        RIGHT_VALUE_COLUMN,
    )

    figure, left_axis = plt.subplots(figsize=(14, 7))
    right_axis = left_axis.twinx()

    plot_one_column(
        axis=left_axis,
        plot_data=left_data,
        value_column=LEFT_VALUE_COLUMN,
        column_type=left_type,
        line_style="-",
    )

    plot_one_column(
        axis=right_axis,
        plot_data=right_data,
        value_column=RIGHT_VALUE_COLUMN,
        column_type=right_type,
        line_style="--",
    )

    left_axis.set_title(
        f"{LEFT_VALUE_COLUMN} and {RIGHT_VALUE_COLUMN} over Time"
    )
    left_axis.set_xlabel("Year")

    left_axis.grid(True, alpha=0.3)

    left_handles, left_labels = left_axis.get_legend_handles_labels()
    right_handles, right_labels = right_axis.get_legend_handles_labels()

    left_axis.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        title="Variables",
        bbox_to_anchor=(1.12, 1),
        loc="upper left",
    )

    return figure, left_axis, right_axis, left_type, right_type


# =============================================================================
# Utilities
# =============================================================================

def make_safe_filename(column_name):
    """Replace characters that cannot be used in Windows filenames."""
    invalid_characters = '<>:"/\\|?*'
    safe_name = column_name

    for character in invalid_characters:
        safe_name = safe_name.replace(character, "_")

    return safe_name


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    figure, left_axis, right_axis, left_type, right_type = plot_two_columns(data)

    figure.tight_layout()

    output_file = (
        OUTPUT_DIR
        / f"{make_safe_filename(LEFT_VALUE_COLUMN)}__and__{make_safe_filename(RIGHT_VALUE_COLUMN)}.png"
    )

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Left column: {LEFT_VALUE_COLUMN}")
    print(f"Left plot type: {left_type}")
    print(f"Right column: {RIGHT_VALUE_COLUMN}")
    print(f"Right plot type: {right_type}")
    print(f"Saved figure to: {output_file}")


if __name__ == "__main__":
    main()