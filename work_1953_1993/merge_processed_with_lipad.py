import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# User settings: change these values when needed
# =============================================================================

START_YEAR = 1963
END_YEAR = 1993

MIN_WORD_COUNT = 50
MAX_WORD_COUNT_PERCENTILE = 99


# =============================================================================
# Paths and column settings
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = Path(__file__).resolve().parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "output" / "processed_data"
LIPAD_DIR = PROJECT_ROOT / "input" / "lipad"
OUTPUT_DIR = WORK_DIR / "output"

OUTPUT_FILE = (
    OUTPUT_DIR
    / f"processed_lipad_{START_YEAR}_{END_YEAR}_filtered.csv"
)
TEMP_OUTPUT_FILE = (
    OUTPUT_DIR
    / f"processed_lipad_{START_YEAR}_{END_YEAR}_filtered.tmp.csv"
)

PROCESSED_SEPARATOR = ";"
LIPAD_SEPARATOR = ","
OUTPUT_SEPARATOR = ";"

MERGE_KEY = "basepk"
TEXT_COLUMN = "speechtext"
SPEAKER_COLUMN = "speakername"
ORIGINAL_TEXT_COLUMN = "speechtext_oringinal"
LIPAD_MERGE_COLUMNS = [
    "speechdate",
    "speakerparty",
    "speakerposition",
    "maintopic",
]
LIPAD_COLUMNS = [
    MERGE_KEY,
    TEXT_COLUMN,
    *LIPAD_MERGE_COLUMNS,
]

REQUIRED_NON_MISSING_COLUMNS = [
    SPEAKER_COLUMN,
    ORIGINAL_TEXT_COLUMN,
    "speechdate",
    "speakerparty",
    "speakerposition",
]

DATE_FILE_PATTERN = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\.csv$"
)


# =============================================================================
# File selection
# =============================================================================

def parse_date_filename(file_path):
    """Return (year, month, day) for a daily CSV filename."""
    match = DATE_FILE_PATTERN.match(file_path.name)

    if match is None:
        return None

    return (
        int(match.group("year")),
        int(match.group("month")),
        int(match.group("day")),
    )


def find_processed_files():
    """Find and chronologically sort processed files in the selected years."""
    selected_files = []

    for file_path in PROCESSED_DATA_DIR.glob("*.csv"):
        file_date = parse_date_filename(file_path)

        if file_date is None:
            continue

        year, _, _ = file_date

        if START_YEAR <= year <= END_YEAR:
            selected_files.append((file_date, file_path))

    selected_files.sort(key=lambda item: item[0])
    return [file_path for _, file_path in selected_files]


def get_lipad_file(processed_file):
    """Return the Lipad file corresponding to one processed daily file."""
    year, month, _ = parse_date_filename(processed_file)
    return LIPAD_DIR / str(year) / str(month) / processed_file.name


# =============================================================================
# Data validation and loading
# =============================================================================

def validate_year_settings():
    """Check that the selected year range and filter settings are valid."""
    if START_YEAR > END_YEAR:
        raise ValueError(
            f"START_YEAR ({START_YEAR}) cannot be greater than "
            f"END_YEAR ({END_YEAR})."
        )

    if MIN_WORD_COUNT < 0:
        raise ValueError("MIN_WORD_COUNT cannot be negative.")

    if not 0 < MAX_WORD_COUNT_PERCENTILE <= 100:
        raise ValueError(
            "MAX_WORD_COUNT_PERCENTILE must be greater than 0 "
            "and no greater than 100."
        )


def validate_column_settings():
    """Check that configured column names are individual strings."""
    invalid_columns = [
        column
        for column in REQUIRED_NON_MISSING_COLUMNS
        if not isinstance(column, str)
    ]

    if invalid_columns:
        raise TypeError(
            "REQUIRED_NON_MISSING_COLUMNS must contain individual column "
            f"names, not nested lists: {invalid_columns}"
        )


def validate_columns(data, required_columns, data_name):
    """Check that a data frame contains all required columns."""
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{data_name} is missing required columns: {missing_columns}"
        )


def normalize_basepk(data, data_name):
    """Convert basepk to a consistent nullable integer type."""
    validate_columns(data, [MERGE_KEY], data_name)

    data = data.copy()
    data[MERGE_KEY] = pd.to_numeric(
        data[MERGE_KEY],
        errors="coerce",
    ).astype("Int64")

    invalid_count = data[MERGE_KEY].isna().sum()

    if invalid_count > 0:
        print(
            f"Warning: {data_name} has {invalid_count} rows with "
            f"missing or invalid {MERGE_KEY}."
        )

    return data


def load_lipad_data(lipad_file):
    """Load the selected speech information from one Lipad file."""
    lipad_data = pd.read_csv(
        lipad_file,
        sep=LIPAD_SEPARATOR,
        usecols=LIPAD_COLUMNS,
        low_memory=False,
    )
    lipad_data = normalize_basepk(lipad_data, str(lipad_file))
    lipad_data = lipad_data.rename(
        columns={TEXT_COLUMN: ORIGINAL_TEXT_COLUMN}
    )

    duplicate_count = lipad_data.duplicated(subset=[MERGE_KEY]).sum()

    if duplicate_count > 0:
        print(
            f"Warning: {lipad_file} has {duplicate_count} duplicate "
            f"{MERGE_KEY} rows; keeping the first occurrence."
        )
        lipad_data = lipad_data.drop_duplicates(
            subset=[MERGE_KEY],
            keep="first",
        )

    return lipad_data


# =============================================================================
# Merging and filtering
# =============================================================================

def merge_daily_file(processed_file):
    """Merge one processed file with its corresponding Lipad file."""
    processed_data = pd.read_csv(
        processed_file,
        sep=PROCESSED_SEPARATOR,
        low_memory=False,
    )
    validate_columns(
        processed_data,
        [MERGE_KEY, TEXT_COLUMN, SPEAKER_COLUMN],
        str(processed_file),
    )
    processed_data = normalize_basepk(processed_data, str(processed_file))

    lipad_file = get_lipad_file(processed_file)

    if not lipad_file.exists():
        merged_data = processed_data.copy()
        merged_data[ORIGINAL_TEXT_COLUMN] = pd.NA

        for column in LIPAD_MERGE_COLUMNS:
            merged_data[column] = pd.NA

        return merged_data, 0, False

    lipad_data = load_lipad_data(lipad_file)

    merged_data = processed_data.merge(
        lipad_data,
        on=MERGE_KEY,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    matched_row_count = (merged_data["_merge"] == "both").sum()
    merged_data = merged_data.drop(columns=["_merge"])

    return merged_data, int(matched_row_count), True


def add_word_count(data):
    """Add the speechtext word-count column."""
    data = data.copy()
    speechtext = data[TEXT_COLUMN].fillna("").astype(str)
    data["speechtext_word_count"] = speechtext.str.split().str.len()
    return data


def get_basic_filter_mask(data):
    """
    Identify rows with enough words and complete required information.

    Empty strings and values containing only spaces are treated as missing.
    """
    keep_rows = data["speechtext_word_count"].ge(MIN_WORD_COUNT)

    for column in REQUIRED_NON_MISSING_COLUMNS:
        complete_values = data[column].fillna("").astype(str).str.strip().ne("")
        keep_rows = keep_rows & complete_values

    return keep_rows


def calculate_max_word_count(processed_files):
    """
    Calculate the global upper word-count threshold.

    The percentile is calculated after applying the minimum-word and
    required-information filters.
    """
    eligible_word_counts = []

    print(
        f"First pass: calculating the {MAX_WORD_COUNT_PERCENTILE}th "
        "percentile of speech length."
    )

    for file_number, processed_file in enumerate(processed_files, start=1):
        merged_data, _, _ = merge_daily_file(processed_file)
        merged_data = add_word_count(merged_data)
        basic_filter_mask = get_basic_filter_mask(merged_data)

        eligible_word_counts.extend(
            merged_data.loc[
                basic_filter_mask,
                "speechtext_word_count",
            ].tolist()
        )

        if file_number % 250 == 0 or file_number == len(processed_files):
            print(
                f"Percentile pass {file_number}/{len(processed_files)} files; "
                f"eligible rows counted: {len(eligible_word_counts)}."
            )

    if not eligible_word_counts:
        raise ValueError(
            "No rows remain after applying the minimum-word and "
            "required-information filters."
        )

    max_word_count = float(
        np.percentile(
            eligible_word_counts,
            MAX_WORD_COUNT_PERCENTILE,
        )
    )

    print(
        f"{MAX_WORD_COUNT_PERCENTILE}th percentile word-count threshold: "
        f"{max_word_count:.2f} words."
    )

    return max_word_count


def filter_speeches(data, max_word_count):
    """Apply the basic filters and remove texts above the percentile."""
    data = add_word_count(data)
    keep_rows = get_basic_filter_mask(data)
    keep_rows = (
        keep_rows
        & data["speechtext_word_count"].le(max_word_count)
    )

    return data.loc[keep_rows].copy()


# =============================================================================
# Main workflow
# =============================================================================

def main():
    validate_year_settings()
    validate_column_settings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if TEMP_OUTPUT_FILE.exists():
        TEMP_OUTPUT_FILE.unlink()

    processed_files = find_processed_files()

    if not processed_files:
        raise FileNotFoundError(
            f"No processed CSV files were found for "
            f"{START_YEAR}-{END_YEAR}."
        )

    print(
        f"Found {len(processed_files)} processed files "
        f"from {START_YEAR} through {END_YEAR}."
    )

    max_word_count = calculate_max_word_count(processed_files)

    print(
        "Second pass: merging data and applying all filters, including "
        f"speechtext_word_count <= {max_word_count:.2f}."
    )

    total_rows = 0
    matched_rows = 0
    retained_rows = 0
    removed_above_percentile = 0
    missing_lipad_files = []
    write_header = True

    for file_number, processed_file in enumerate(processed_files, start=1):
        merged_data, daily_matched_rows, lipad_file_exists = merge_daily_file(
            processed_file
        )
        merged_data = add_word_count(merged_data)
        basic_filter_mask = get_basic_filter_mask(merged_data)

        removed_above_percentile += int(
            (
                basic_filter_mask
                & merged_data["speechtext_word_count"].gt(max_word_count)
            ).sum()
        )

        filtered_data = merged_data.loc[
            basic_filter_mask
            & merged_data["speechtext_word_count"].le(max_word_count)
        ].copy()

        filtered_data.to_csv(
            TEMP_OUTPUT_FILE,
            sep=OUTPUT_SEPARATOR,
            index=False,
            mode="w" if write_header else "a",
            header=write_header,
            encoding="utf-8-sig" if write_header else "utf-8",
        )

        write_header = False
        total_rows += len(merged_data)
        matched_rows += daily_matched_rows
        retained_rows += len(filtered_data)

        if not lipad_file_exists:
            missing_lipad_files.append(str(get_lipad_file(processed_file)))

        if file_number % 250 == 0 or file_number == len(processed_files):
            print(
                f"Processed {file_number}/{len(processed_files)} files; "
                f"rows read: {total_rows}; "
                f"rows retained: {retained_rows}."
            )

    os.replace(TEMP_OUTPUT_FILE, OUTPUT_FILE)

    unmatched_rows = total_rows - matched_rows
    removed_rows = total_rows - retained_rows

    print(f"Saved merged and filtered data to: {OUTPUT_FILE}")
    print(f"Input rows: {total_rows}")
    print(f"Rows matched by {MERGE_KEY}: {matched_rows}")
    print(f"Rows without a Lipad match: {unmatched_rows}")
    print(
        f"Maximum retained word count ({MAX_WORD_COUNT_PERCENTILE}th "
        f"percentile): {max_word_count:.2f}"
    )
    print(
        f"Rows removed above the percentile threshold: "
        f"{removed_above_percentile}"
    )
    print(f"Retained rows: {retained_rows}")
    print(f"Removed rows: {removed_rows}")
    print(f"Missing corresponding Lipad files: {len(missing_lipad_files)}")

    if missing_lipad_files:
        print("First missing Lipad files:")

        for file_path in missing_lipad_files[:10]:
            print(f"  {file_path}")


if __name__ == "__main__":
    main()
