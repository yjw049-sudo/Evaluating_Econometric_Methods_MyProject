"""Add stemmed speech text to output/Speech_cleaned.csv."""

from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_OUTPUT_FILE = PROJECT_DIR / "output" / "Speech_sample.csv"
TEMP_OUTPUT_FILE = PROJECT_DIR / "output" / "Speech_sample.stem.tmp.csv"

ORIGINAL_TEXT_COLUMN = "speechtext"
STEM_TEXT_COLUMN = "speechtext_stem"
STEM_WORD_COUNT_COLUMN = "speechtext_stem_word_count"
ORIGINAL_WORD_COUNT_COLUMN = "speechtext_word_count"
CHUNK_SIZE = 10_000

TOKENIZER = RegexpTokenizer(r"\w+")
STEMMER = PorterStemmer()
KEEP_POS = {"NOUN", "VERB", "ADJ", "ADP"}
EXTRA_STOP = {
    "hon",
    "member",
    "mr",
    "speaker",
    "chairman",
    "minister",
    "would",
    "read",
    "think",
    "know",
    "made",
    "say",
    "ye",
    "b",
    "aba",
    "surditi",
    "newsom",
    "oh",
    "zurich",
}


def load_stopwords():
    """Load the same English stopword list used by the original pipeline."""
    try:
        return set(stopwords.words("english"))
    except LookupError as error:
        raise LookupError(
            "NLTK stopwords are missing. Run: "
            "python -m nltk.downloader stopwords averaged_perceptron_tagger_eng "
            "universal_tagset"
        ) from error


def stem_text(text, english_stopwords):
    """Apply tokenization, POS filtering, stopword removal, and stemming."""
    if pd.isna(text):
        return ""

    tokens = TOKENIZER.tokenize(str(text).lower())
    tagged_tokens = nltk.pos_tag(tokens, tagset="universal")

    stemmed_words = []
    for word, part_of_speech in tagged_tokens:
        if (
            part_of_speech in KEEP_POS
            and word not in english_stopwords
            and word not in EXTRA_STOP
        ):
            stemmed_words.append(STEMMER.stem(word))

    return " ".join(stemmed_words)


def validate_input():
    """Check that Speech_merged.csv contains the raw text columns."""
    if not INPUT_OUTPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_OUTPUT_FILE}")

    columns = pd.read_csv(INPUT_OUTPUT_FILE, nrows=0).columns.tolist()
    required_columns = ["basepk", ORIGINAL_TEXT_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in columns
    ]
    if missing_columns:
        raise ValueError(
            f"Speech_merged.csv is missing required columns: {missing_columns}"
        )


def prepare_stem_chunk(chunk, english_stopwords):
    """Add or replace speechtext_stem and its word-count columns."""
    prepared = chunk.copy()
    prepared[STEM_TEXT_COLUMN] = prepared[ORIGINAL_TEXT_COLUMN].map(
        lambda text: stem_text(text, english_stopwords)
    )
    prepared[STEM_WORD_COUNT_COLUMN] = (
        prepared[STEM_TEXT_COLUMN].fillna("").astype(str).str.split().str.len()
    )

    # Keep the two length measures separate: speechtext_word_count describes
    # the original text; speechtext_stem_word_count describes the stemmed text.
    prepared[ORIGINAL_WORD_COUNT_COLUMN] = (
        prepared[ORIGINAL_TEXT_COLUMN].fillna("").astype(str).str.split().str.len()
    )

    preferred_columns = [
        "basepk",
        ORIGINAL_TEXT_COLUMN,
        STEM_TEXT_COLUMN,
        "speakername",
        "year",
        "speechdate",
        "speakerposition",
        "maintopic",
        "speakerparty",
        STEM_WORD_COUNT_COLUMN,
        ORIGINAL_WORD_COUNT_COLUMN,
    ]
    existing_preferred = [
        column for column in preferred_columns if column in prepared.columns
    ]
    remaining_columns = [
        column for column in prepared.columns if column not in existing_preferred
    ]
    return prepared[existing_preferred + remaining_columns]


def main():
    """Generate stemmed text and atomically update Speech_cleaned.csv."""
    validate_input()
    english_stopwords = load_stopwords()

    if TEMP_OUTPUT_FILE.exists():
        TEMP_OUTPUT_FILE.unlink()

    total_rows = 0
    first_write = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_OUTPUT_FILE,
            chunksize=CHUNK_SIZE,
            dtype={"basepk": "string"},
            low_memory=False,
        ),
        start=1,
    ):
        prepared = prepare_stem_chunk(chunk, english_stopwords)
        prepared.to_csv(
            TEMP_OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig" if first_write else "utf-8",
        )
        total_rows += len(prepared)
        first_write = False
        print(f"Processed chunk {chunk_number}; rows: {total_rows:,}")

    if first_write:
        raise ValueError("Speech_cleaned.csv contains no data rows")

    TEMP_OUTPUT_FILE.replace(INPUT_OUTPUT_FILE)

    print("\nStemming completed")
    print(f"Rows: {total_rows:,}")
    print(f"Updated: {INPUT_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
