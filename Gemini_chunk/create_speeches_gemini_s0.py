from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "output" / "Speech_chunk.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "speeches_Gemin_s0.csv"


KEEP_COLUMNS = ["basepk", "chunk_index", "speechtext_chunk"]


def main() -> None:
    speeches = pd.read_csv(INPUT_FILE)

    missing_columns = [column for column in KEEP_COLUMNS if column not in speeches.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in input file: {missing_columns}")

    selected_speeches = speeches[KEEP_COLUMNS].copy()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    selected_speeches.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    speech_text = selected_speeches["speechtext_chunk"]
    missing_value_count = speech_text.isna().sum()
    blank_text_count = speech_text.fillna("").astype(str).str.strip().eq("").sum()
    total_characters = speech_text.fillna("").astype(str).str.len().sum()

    print(f"Saved file: {OUTPUT_FILE}")
    print(f"Rows copied without filtering: {len(selected_speeches):,}")
    print(f"speechtext_chunk total characters: {total_characters}")
    print(f"speechtext_chunk has missing values: {missing_value_count > 0}")
    print(f"speechtext_chunk missing value count: {missing_value_count}")
    print(f"speechtext_chunk has blank texts: {blank_text_count > 0}")
    print(f"speechtext_chunk blank text count: {blank_text_count}")


if __name__ == "__main__":
    main()






