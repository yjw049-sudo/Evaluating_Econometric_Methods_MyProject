from pathlib import Path
import pandas as pd
import nltk
import time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

DATA_DIR = Path('E:\study\S2\Evaluating-Econometric-Methods_data\Evaluating_Econometric_Methods_MyProject')

# Instantiate once, not per row
TOKENIZER = RegexpTokenizer(r'\w+')
STEMMER   = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))
KEEP_POS  = {'NOUN', 'VERB', 'ADJ', 'ADP'}
EXTRA_STOP = {'canada', 'hon', 'member', 'question', 'ask'}


def load_speeches(year: int, month: int,step: int) -> pd.DataFrame:
    """Load one Speeches{year}_{step}.xlsx file, dedupe, filter by length."""
    path = DATA_DIR / 'input' / str(year) / str(month) / f'{year}-{month}-{step}.csv'
    df = (pd.read_csv(path)
            .dropna(subset=['speakername', 'speechtext'])
            .drop_duplicates(subset='speechtext'))
    df['year'] = df['speechdate'].str.split('-').str[0]
    df = df[['basepk', 'speechtext', 'speakername', 'year']]
    print(f"Mean length: {df['speechtext'].str.len().mean():.1f}")
    return df.copy()


def clean_text(text: str, extra_stop: set = EXTRA_STOP) -> str:
    """Tokenize → POS-tag → keep content words → stem."""
    tokens = TOKENIZER.tokenize(text.lower())
    tagged = nltk.pos_tag(tokens, tagset='universal')
    keep = [
        STEMMER.stem(w) for w, pos in tagged
        if pos in KEEP_POS and w not in STOPWORDS and w not in extra_stop
    ]
    return ' '.join(keep)


def preprocess(year: int, month: int, step: int, extra_stop: set = EXTRA_STOP) -> pd.DataFrame:
    df = load_speeches(year, month, step)
    print(f"Loaded {len(df)} speeches.")
    df['speechtext'] = df['speechtext'].map(lambda t: clean_text(t, extra_stop))
    df = df.dropna(subset=['speechtext'])
    return df


if __name__ == '__main__':
    beg = time.time()
    for year in range(1901, 2019):
        for month in range(1, 13):
            for step in range(0, 40):
                path = DATA_DIR / 'input' / str(year) / str(month) / f'{year}-{month}-{step}.csv'
                if not path.exists():
                    print(f"File not found: {path}")
                    continue
                df = preprocess(year, month, step)
                df.to_csv(DATA_DIR / 'output' / 'processed_data' / f'{year}-{month}-{step}.csv', index=False, sep=';')
    end = time.time()
    print("Duration of 