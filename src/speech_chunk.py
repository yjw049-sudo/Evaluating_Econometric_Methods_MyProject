"""Split each cleaned speech into balanced word chunks."""

import csv
import math
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "output" / "Speech_sample.csv"
OUTPUT_FILE = PROJECT_DIR / "output" / "Speech_chunk.csv"
TEMP_OUTPUT_FILE = PROJECT_DIR / "output" / "Speech_chunk.tmp.csv"

TEXT_COLUMN = "speechtext"
CHUNK_INDEX_COLUMN = "chunk_index"
CHUNK_TOTAL_COLUMN = "chunk_total"
CHUNK_TEXT_COLUMN = "speechtext_chunk"

# Target number of words per chunk. Each chunk will normally contain between
# SIZE - TOLERANCE and SIZE + TOLERANCE words.
SIZE = 100
TOLERANCE = SIZE // 2


def split_text_into_chunks(text, size=SIZE, tolerance=TOLERANCE):
    """Split text into balanced chunks without cutting through a word."""
    if size <= 0:
        raise ValueError("SIZE must be greater than 0")
    if tolerance < 0:
        raise ValueError("TOLERANCE must not be negative")
    if size <= tolerance:
        raise ValueError("SIZE must be greater than TOLERANCE")

    words = str(text).split()
    if not words:
        return [""]

    maximum_words = size + tolerance

    # Use the fewest chunks that satisfy the maximum length, then distribute
    # all words evenly. For example, 500 words become two chunks of 250 words.
    chunk_total = max(1, math.ceil(len(words) / maximum_words))
    base_length, longer_chunk_count = divmod(len(words), chunk_total)

    chunks = []
    start = 0
    for chunk_index in range(chunk_total):
        chunk_length = base_length + (chunk_index < longer_chunk_count)
        end = start + chunk_length
        chunks.append(" ".join(words[start:end]))
        start = end

    return chunks


def validate_columns(fieldnames):
    """Check that the source text exists and new columns will not overwrite data."""
    if not fieldnames or TEXT_COLUMN not in fieldnames:
        raise ValueError(f"Input CSV is missing required column: {TEXT_COLUMN}")

    output_columns = [CHUNK_INDEX_COLUMN, CHUNK_TOTAL_COLUMN, CHUNK_TEXT_COLUMN]
    duplicate_columns = [column for column in output_columns if column in fieldnames]
    if duplicate_columns:
        raise ValueError(
            "Input CSV already contains output columns: "
            f"{duplicate_columns}"
        )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    input_rows = 0
    output_rows = 0

    try:
        with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as input_file:
            reader = csv.DictReader(input_file)
            validate_columns(reader.fieldnames)
            output_fieldnames = reader.fieldnames + [
                CHUNK_INDEX_COLUMN,
                CHUNK_TOTAL_COLUMN,
                CHUNK_TEXT_COLUMN,
            ]

            with TEMP_OUTPUT_FILE.open(
                "w", encoding="utf-8-sig", newline=""
            ) as output_file:
                writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
                writer.writeheader()

                for row in reader:
                    chunks = split_text_into_chunks(row[TEXT_COLUMN])
                    chunk_total = len(chunks)

                    for chunk_index, chunk_text in enumerate(chunks):
                        output_row = row.copy()
                        output_row[CHUNK_INDEX_COLUMN] = chunk_index
                        output_row[CHUNK_TOTAL_COLUMN] = chunk_total
                        output_row[CHUNK_TEXT_COLUMN] = chunk_text
                        writer.writerow(output_row)
                        output_rows += 1

                    input_rows += 1
                    if input_rows % 10_000 == 0:
                        print(
                            f"Processed {input_rows:,} speeches; "
                            f"created {output_rows:,} chunks"
                        )

        TEMP_OUTPUT_FILE.replace(OUTPUT_FILE)
    except Exception:
        if TEMP_OUTPUT_FILE.exists():
            TEMP_OUTPUT_FILE.unlink()
        raise

    print("\nChunking completed")
    print(f"Input speeches: {input_rows:,}")
    print(f"Output chunks: {output_rows:,}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
