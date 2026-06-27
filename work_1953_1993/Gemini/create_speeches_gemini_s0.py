from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "output" / "speeches_with_parlinfo.csv"
OUTPUT_FILE = PROJECT_DIR / "Gemini" / "speeches_Gemin_s0.csv"

MATCH_COLUMN = "is_mp_matched"
KEEP_COLUMNS = ["basepk", "speechtext_oringinal"]


def main() -> None:
    speeches = pd.read_csv(INPUT_FILE)

    missing_columns = [
        column for column in [MATCH_COLUMN, *KEEP_COLUMNS]
        if column not in speeches.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns in input file: {missing_columns}")

    is_matched = (
        speeches[MATCH_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    matched_speeches = speeches.loc[is_matched, KEEP_COLUMNS].copy()
    matched_speeches.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    speech_text = matched_speeches["speechtext_oringinal"]
    missing_value_count = speech_text.isna().sum()
    blank_text_count = speech_text.fillna("").astype(str).str.strip().eq("").sum()
    total_characters = speech_text.fillna("").astype(str).str.len().sum()

    print(f"Saved file: {OUTPUT_FILE}")
    print(f"speechtext_oringinal total characters: {total_characters}")
    print(f"speechtext_oringinal has missing values: {missing_value_count > 0}")
    print(f"speechtext_oringinal missing value count: {missing_value_count}")
    print(f"speechtext_oringinal has blank texts: {blank_text_count > 0}")
    print(f"speechtext_oringinal blank text count: {blank_text_count}")


if __name__ == "__main__":
    main()
