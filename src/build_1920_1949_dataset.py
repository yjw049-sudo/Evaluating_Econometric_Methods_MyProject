from pathlib import Path
import re
import time

import pandas as pd


# =============================================================================
# Parameters
# =============================================================================

DATA_DIR = Path(r"E:\study\S2\Evaluating-Econometric-Methods_data\Evaluating_Econometric_Methods_MyProject")

PROCESSED_DATA_DIR = DATA_DIR / "output" / "processed_data"
CLASSIFIED_FILE = (
    DATA_DIR
    / "lecture"
    / "speeches_classified_2605"
    / "speeches_categorized_2605.csv"
)
OUTPUT_DIR = DATA_DIR / "output" / "1920-1949-v1"

START_YEAR = 1920
END_YEAR = 1949

INPUT_SEPARATOR = ";"
OUTPUT_SEPARATOR = ";"
CLASSIFIED_SEPARATOR = ","

MERGE_KEY = "basepk"
MERGE_TYPE = "inner"

PROCESSED_OUTPUT_FILE = OUTPUT_DIR / "processed_data_1920_1949.csv"
MERGED_OUTPUT_FILE = OUTPUT_DIR / "speeches_1920_1949_merged.csv"


# =============================================================================
# File selection
# =============================================================================


def get_year_from_filename(file_path):
    """Extract the year from a filename such as 1920-2-26.csv."""
    match = re.match(r"^(\d{4})-\d{1,2}-\d{1,2}\.csv$", file_path.name)

    if match is None:
        return None

    return int(match.group(1))


def find_processed_files():
    """Find all processed_data CSV files between START_YEAR and END_YEAR."""
    selected_files = []

    for file_path in PROCESSED_DATA_DIR.glob("*.csv"):
        year = get_year_from_filename(file_path)

        if year is not None and START_YEAR <= year <= END_YEAR:
            selected_files.append(file_path)

    return sorted(selected_files)


# =============================================================================
# Data loading and cleaning
# =============================================================================


def load_processed_data(file_paths):
    """Read and combine the selected processed_data files."""
    data_frames = []

    for file_path in file_paths:
        daily_data = pd.read_csv(file_path, sep=INPUT_SEPARATOR)
        daily_data["source_file"] = file_path.name

        if len(daily_data) > 0:
            data_frames.append(daily_data)

    if not data_frames:
        raise ValueError("No processed_data files were found for 1920-1949.")

    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data


def normalize_basepk(data, data_name):
    """Convert basepk to a consistent nullable integer type for merging."""
    if MERGE_KEY not in data.columns:
        raise ValueError(f"{data_name} does not contain a {MERGE_KEY} column.")

    data = data.copy()
    data[MERGE_KEY] = pd.to_numeric(data[MERGE_KEY], errors="coerce").astype("Int64")
    missing_basepk_count = data[MERGE_KEY].isna().sum()

    if missing_basepk_count > 0:
        print(f"{data_name}: {missing_basepk_count} rows have missing or invalid {MERGE_KEY}.")

    return data


# =============================================================================
# Main workflow
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    processed_files = find_processed_files()
    print(f"Found {len(processed_files)} processed_data files from {START_YEAR} to {END_YEAR}.")

    processed_data = load_processed_data(processed_files)
    processed_data = normalize_basepk(processed_data, "processed_data")

    processed_data.to_csv(PROCESSED_OUTPUT_FILE, sep=OUTPUT_SEPARATOR, index=False)
    print(f"Saved combined processed_data to: {PROCESSED_OUTPUT_FILE}")

    categorized_data = pd.read_csv(CLASSIFIED_FILE, sep=CLASSIFIED_SEPARATOR)
    categorized_data = normalize_basepk(categorized_data, "categorized_data")

    merged_data = processed_data.merge(
        categorized_data,
        on=MERGE_KEY,
        how=MERGE_TYPE,
        validate="many_to_one",
    )

    merged_data.to_csv(MERGED_OUTPUT_FILE, sep=OUTPUT_SEPARATOR, index=False)
    print(f"Saved merged data to: {MERGED_OUTPUT_FILE}")
    print(f"Rows in combined processed_data: {len(processed_data)}")
    print(f"Rows in merged data: {len(merged_data)}")

    end_time = time.time()
    print(f"Duration of dataset building: {end_time - start_time:8.2f} sec.")


if __name__ == "__main__":
    main()
