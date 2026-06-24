from pathlib import Path

import pandas as pd
import numpy as np


# =============================================================================
# Settings
# =============================================================================

WORK_DIR = Path(__file__).resolve().parent
INPUT_FILE = WORK_DIR / "output" / "processed_lipad_1953_1993_filtered.csv"
OUTPUT_FILE = WORK_DIR / "output" / "temporary_sample_50.csv"

CSV_SEPARATOR = ";"
SAMPLE_SIZE = 50
RANDOM_SEED = 42
CHUNK_SIZE = 20_000


def create_random_sample():
    """
    Randomly sample rows from the full input file.

    This method reads the large file in chunks and does not load the complete
    data set into memory.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file was not found: {INPUT_FILE}")

    sampled_data = None
    rows_seen = 0
    random_generator = np.random.default_rng(RANDOM_SEED)

    data_chunks = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        chunksize=CHUNK_SIZE,
        low_memory=True,
    )

    for chunk_number, data_chunk in enumerate(data_chunks, start=1):
        data_chunk = data_chunk.copy()
        data_chunk["_temporary_random_key"] = random_generator.random(
            len(data_chunk)
        )

        if sampled_data is None:
            candidate_data = data_chunk
        else:
            candidate_data = pd.concat(
                [sampled_data, data_chunk],
                ignore_index=True,
            )

        sampled_data = candidate_data.nsmallest(
            SAMPLE_SIZE,
            "_temporary_random_key",
        ).copy()
        rows_seen += len(data_chunk)

        print(
            f"Processed chunk {chunk_number}; "
            f"rows examined: {rows_seen}."
        )

    if sampled_data is None or len(sampled_data) < SAMPLE_SIZE:
        raise ValueError(
            f"Input file contains fewer than {SAMPLE_SIZE} rows."
        )

    sampled_data = sampled_data.drop(columns=["_temporary_random_key"])
    sampled_data = sampled_data.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    sampled_data.to_csv(
        OUTPUT_FILE,
        sep=CSV_SEPARATOR,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved {len(sampled_data)} sampled rows to: {OUTPUT_FILE}")
    print(f"Columns: {list(sampled_data.columns)}")


if __name__ == "__main__":
    create_random_sample()
