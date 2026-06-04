from pathlib import Path
import time

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer


# =============================================================================
# Parameters
# =============================================================================

DATA_DIR = Path(r"E:\study\S2\Evaluating-Econometric-Methods_data\Evaluating_Econometric_Methods_MyProject")
INPUT_DIR = DATA_DIR / "input" / "lipad"
OUTPUT_DIR = DATA_DIR / "output" / "processed_data"

TOKENIZER = RegexpTokenizer(r"\w+")
STEMMER = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))
KEEP_POS = {"NOUN", "VERB", "ADJ", "ADP"}

EXTRA_STOP = {
    "hon", "member", "mr", "speaker", "chairman", "minister",
    "would", "read", "think", "know", "made", "say",
    "ye", "b", "aba", "surditi", "newsom", "oh", "zurich",
}


# =============================================================================
# Text processing
# =============================================================================

def load_speeches(year, month, day):
    input_path = INPUT_DIR / str(year) / str(month) / f"{year}-{month}-{day}.csv"

    data = pd.read_csv(input_path)
    data = data.dropna(subset=["speakername", "speechtext"])
    data = data.drop_duplicates(subset="speechtext")

    data["year"] = data["speechdate"].str.split("-").str[0]
    data = data[["basepk", "speechtext", "speakername", "year"]]

    print(f"Mean length: {data['speechtext'].str.len().mean():.1f}")

    return data.copy()


def clean_text(text, extra_stop=EXTRA_STOP):
    """Tokenize, keep selected parts of speech, remove stop words, and stem words."""
    tokens = TOKENIZER.tokenize(text.lower())
    tagged_tokens = nltk.pos_tag(tokens, tagset="universal")

    cleaned_words = []
    for word, pos in tagged_tokens:
        if pos in KEEP_POS and word not in STOPWORDS and word not in extra_stop:
            cleaned_words.append(STEMMER.stem(word))

    return " ".join(cleaned_words)


def preprocess_file(year, month, day):
    data = load_speeches(year, month, day)
    print(f"Loaded {len(data)} speeches.")

    data["speechtext"] = data["speechtext"].map(clean_text)
    data = data.dropna(subset=["speechtext"])

    return data


# =============================================================================
# Main workflow
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for year in range(1901, 2020):
        for month in range(1, 13):
            for day in range(1, 32):
                input_path = INPUT_DIR / str(year) / str(month) / f"{year}-{month}-{day}.csv"
                output_path = OUTPUT_DIR / f"{year}-{month}-{day}.csv"

                if not input_path.exists():
                    print(f"File not found: {year}-{month}-{day}.csv")
                    continue

                if output_path.exists():
                    print(f"Already processed: {year}-{month}-{day}.csv")
                    continue

                data = preprocess_file(year, month, day)
                data.to_csv(output_path, index=False, sep=";")

    end_time = time.time()
    print(f"Duration of Preparation: {end_time - start_time:8.2f} sec.")


if __name__ == "__main__":
    main()
