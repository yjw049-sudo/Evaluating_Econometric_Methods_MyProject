import joblib
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from project_config import (
    AI_LABEL_COLUMN,
    BASEPK_COLUMN,
    CLUSTER_COLUMN,
    CSV_SEPARATOR,
    FULL_CORPUS_OUTPUT_DIR,
    LIPAD_DIR,
    ORIGINAL_TEXT_COLUMN,
    OUTPUT_1920_1949_DIR,
    TEXT_COLUMN,
    YEAR_COLUMN,
)


# =============================================================================
# Project paths
# =============================================================================

INPUT_DIR = OUTPUT_1920_1949_DIR
INPUT_FILE = INPUT_DIR / "speeches_1920_1949_merged.csv"
ORIGINAL_INPUT_DIR = LIPAD_DIR

OUTPUT_DIR = FULL_CORPUS_OUTPUT_DIR
MODEL_DIR = OUTPUT_DIR / "models"

PROCESSED_OUTPUT_FILE = OUTPUT_DIR / "speeches_1920_1949_kmeans_ai_label_structure.csv"

CLUSTER_TERMS_FILE = OUTPUT_DIR / "kmeans_cluster_top_terms.csv"
CLUSTER_COUNTS_FILE = OUTPUT_DIR / "kmeans_cluster_counts.csv"
CLUSTER_COUNTS_BY_YEAR_FILE = OUTPUT_DIR / "kmeans_cluster_counts_by_year.csv"
CLUSTER_COUNTS_BY_PERIOD_FILE = OUTPUT_DIR / "kmeans_cluster_counts_by_period.csv"

CLUSTER_AI_LABEL_COUNTS_FILE = OUTPUT_DIR / "cluster_ai_label_crosstab_counts.csv"
CLUSTER_AI_LABEL_ROW_SHARES_FILE = OUTPUT_DIR / "cluster_ai_label_crosstab_row_shares.csv"

AI_LABEL_CLUSTER_COUNTS_FILE = OUTPUT_DIR / "ai_label_cluster_crosstab_counts.csv"
AI_LABEL_CLUSTER_ROW_SHARES_FILE = OUTPUT_DIR / "ai_label_cluster_crosstab_row_shares.csv"

CLUSTER_DOMINANT_LABEL_FILE = OUTPUT_DIR / "cluster_dominant_ai_label_mapping.csv"
AI_LABEL_DOMINANT_CLUSTER_FILE = OUTPUT_DIR / "ai_label_dominant_cluster_mapping.csv"

TFIDF_MODEL_FILE = MODEL_DIR / "tfidf_vectorizer_1920_1949.joblib"
KMEANS_MODEL_FILE = MODEL_DIR / "kmeans_1920_1949.joblib"
ALL_TFIDF_FILE = MODEL_DIR / "tfidf_all_1920_1949.npz"


# =============================================================================
# Parameters
# =============================================================================

MIN_SPEECH_WORD_COUNT = 20

N_CLUSTERS = 20
NGRAM_RANGE = (1, 2)
TOP_TERMS_PER_CLUSTER = 30
RANDOM_STATE = 42

START_YEAR = 1920
END_YEAR = 1949


# =============================================================================
# Data loading and preprocessing
# =============================================================================

def load_original_speechtext():
    """
    Load raw speechtext from input/lipad for 1920-1949.

    The raw text is kept as speechtext_oringinal so that the final output
    contains both the cleaned text used by the model and the original text.
    """
    original_frames = []

    for year in range(START_YEAR, END_YEAR + 1):
        year_dir = ORIGINAL_INPUT_DIR / str(year)

        if not year_dir.exists():
            print(f"Original input year folder not found: {year_dir}")
            continue

        for file_path in sorted(year_dir.glob("*/*.csv")):
            daily_data = pd.read_csv(
                file_path,
                usecols=[BASEPK_COLUMN, TEXT_COLUMN],
            )
            original_frames.append(daily_data)

    if not original_frames:
        raise FileNotFoundError("No original input files were found for 1920-1949.")

    original_data = pd.concat(original_frames, ignore_index=True)
    original_data = original_data.rename(columns={TEXT_COLUMN: ORIGINAL_TEXT_COLUMN})

    original_data[BASEPK_COLUMN] = pd.to_numeric(
        original_data[BASEPK_COLUMN],
        errors="coerce",
    ).astype("Int64")

    original_data = original_data.dropna(subset=[BASEPK_COLUMN])
    original_data = original_data.drop_duplicates(subset=[BASEPK_COLUMN], keep="first")

    return original_data


def load_and_filter_data():
    """
    Load the merged speech data and remove very short speeches.

    The filtering is based on word count rather than character length, because
    very short texts are usually not informative for TF-IDF and k-means.
    """
    data = pd.read_csv(INPUT_FILE, sep=CSV_SEPARATOR)

    required_columns = [BASEPK_COLUMN, TEXT_COLUMN, YEAR_COLUMN, AI_LABEL_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        raise ValueError(f"Input file is missing required columns: {missing_columns}")

    data[BASEPK_COLUMN] = pd.to_numeric(
        data[BASEPK_COLUMN],
        errors="coerce",
    ).astype("Int64")
    data = data.dropna(subset=[BASEPK_COLUMN]).copy()

    original_data = load_original_speechtext()
    data = data.merge(
        original_data,
        on=BASEPK_COLUMN,
        how="left",
        validate="many_to_one",
    )

    missing_original_count = data[ORIGINAL_TEXT_COLUMN].isna().sum()
    if missing_original_count > 0:
        print(f"Rows without original speechtext: {missing_original_count}")

    data[TEXT_COLUMN] = data[TEXT_COLUMN].fillna("").astype(str)
    data[ORIGINAL_TEXT_COLUMN] = data[ORIGINAL_TEXT_COLUMN].fillna("").astype(str)
    data[AI_LABEL_COLUMN] = data[AI_LABEL_COLUMN].fillna("Unknown").astype(str)

    data["speechtext_word_count"] = data[TEXT_COLUMN].str.split().str.len()
    data["speechtext_char_length"] = data[TEXT_COLUMN].str.len()

    filtered_data = data[data["speechtext_word_count"] >= MIN_SPEECH_WORD_COUNT].copy()
    filtered_data = filtered_data.reset_index(drop=True)

    return filtered_data


def add_descriptive_periods(data):
    """
    Add broad descriptive periods.

    These are not train/test labels. They are only for descriptive tables and
    optional historical interpretation.
    """
    data = data.copy()

    data["period"] = "outside_1920_1949"

    data.loc[data[YEAR_COLUMN].between(1920, 1929), "period"] = "1920_1929"
    data.loc[data[YEAR_COLUMN].between(1930, 1945), "period"] = "1930_1945"
    data.loc[data[YEAR_COLUMN].between(1946, 1949), "period"] = "1946_1949"

    return data


# =============================================================================
# TF-IDF and KMeans
# =============================================================================

def fit_tfidf_vectorizer(data):
    """
    Fit TF-IDF on all 1920-1949 texts.

    This script does not perform temporal extrapolation. Therefore, TF-IDF is
    fitted on the full selected corpus.
    """
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=NGRAM_RANGE,
        min_df=5,
        max_df=0.90,
        stop_words="english",
    )

    all_tfidf = vectorizer.fit_transform(data[TEXT_COLUMN])

    return vectorizer, all_tfidf


def fit_kmeans(data, all_tfidf):
    """
    Fit KMeans on the full 1920-1949 corpus and assign each speech to a cluster.
    """
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=RANDOM_STATE,
    )

    data = data.copy()
    data[CLUSTER_COLUMN] = kmeans.fit_predict(all_tfidf)

    return data, kmeans


# =============================================================================
# Cluster interpretation outputs
# =============================================================================

def save_cluster_top_terms(kmeans, vectorizer):
    """
    Save top TF-IDF terms for each KMeans cluster.
    """
    feature_names = vectorizer.get_feature_names_out()
    rows = []

    for cluster_id, center in enumerate(kmeans.cluster_centers_):
        top_indices = center.argsort()[::-1][:TOP_TERMS_PER_CLUSTER]

        for rank, term_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    CLUSTER_COLUMN: cluster_id,
                    "rank": rank,
                    "term": feature_names[term_index],
                    "tfidf_center_weight": center[term_index],
                }
            )

    cluster_terms = pd.DataFrame(rows)
    cluster_terms.to_csv(CLUSTER_TERMS_FILE, sep=CSV_SEPARATOR, index=False)


def save_cluster_counts(data):
    """
    Save cluster size tables.
    """
    cluster_counts = (
        data.groupby(CLUSTER_COLUMN)
        .size()
        .reset_index(name="count")
        .sort_values(CLUSTER_COLUMN)
    )

    cluster_counts["share"] = cluster_counts["count"] / cluster_counts["count"].sum()
    cluster_counts.to_csv(CLUSTER_COUNTS_FILE, sep=CSV_SEPARATOR, index=False)

    cluster_counts_by_year = (
        data.groupby([YEAR_COLUMN, CLUSTER_COLUMN])
        .size()
        .reset_index(name="count")
        .sort_values([YEAR_COLUMN, CLUSTER_COLUMN])
    )

    cluster_counts_by_year.to_csv(
        CLUSTER_COUNTS_BY_YEAR_FILE,
        sep=CSV_SEPARATOR,
        index=False,
    )

    cluster_counts_by_period = (
        data.groupby(["period", CLUSTER_COLUMN])
        .size()
        .reset_index(name="count")
        .sort_values(["period", CLUSTER_COLUMN])
    )

    cluster_counts_by_period.to_csv(
        CLUSTER_COUNTS_BY_PERIOD_FILE,
        sep=CSV_SEPARATOR,
        index=False,
    )


# =============================================================================
# AI label and KMeans comparison
# =============================================================================

def calculate_entropy_from_shares(shares):
    """
    Calculate entropy from a vector of shares.

    This avoids importing scipy.stats only for entropy.
    Zero shares are ignored because 0 * log(0) is treated as 0.
    """
    import numpy as np

    positive_shares = shares[shares > 0]

    if len(positive_shares) == 0:
        return 0.0

    return float(-(positive_shares * np.log(positive_shares)).sum())


def save_cluster_ai_label_crosstabs(data):
    """
    Save both directions of the AI label x KMeans comparison.

    1. cluster_ai_label_*:
       Rows are k-means clusters. Columns are AI labels.
       This answers: what AI labels appear inside each cluster?

    2. ai_label_cluster_*:
       Rows are AI labels. Columns are k-means clusters.
       This answers: how is each AI label distributed across clusters?
    """
    cluster_ai_label_counts = pd.crosstab(
        data[CLUSTER_COLUMN],
        data[AI_LABEL_COLUMN],
    )

    cluster_ai_label_row_shares = cluster_ai_label_counts.div(
        cluster_ai_label_counts.sum(axis=1),
        axis=0,
    )

    ai_label_cluster_counts = pd.crosstab(
        data[AI_LABEL_COLUMN],
        data[CLUSTER_COLUMN],
    )

    ai_label_cluster_row_shares = ai_label_cluster_counts.div(
        ai_label_cluster_counts.sum(axis=1),
        axis=0,
    )

    cluster_ai_label_counts.to_csv(CLUSTER_AI_LABEL_COUNTS_FILE, sep=CSV_SEPARATOR)
    cluster_ai_label_row_shares.to_csv(
        CLUSTER_AI_LABEL_ROW_SHARES_FILE,
        sep=CSV_SEPARATOR,
    )

    ai_label_cluster_counts.to_csv(AI_LABEL_CLUSTER_COUNTS_FILE, sep=CSV_SEPARATOR)
    ai_label_cluster_row_shares.to_csv(
        AI_LABEL_CLUSTER_ROW_SHARES_FILE,
        sep=CSV_SEPARATOR,
    )

    return (
        cluster_ai_label_counts,
        cluster_ai_label_row_shares,
        ai_label_cluster_counts,
        ai_label_cluster_row_shares,
    )


def save_cluster_dominant_label_mapping(cluster_ai_label_counts):
    """
    For each k-means cluster, identify the dominant AI label and how pure the
    cluster is with respect to AI labels.
    """
    rows = []

    for cluster_id, row in cluster_ai_label_counts.iterrows():
        cluster_total = int(row.sum())
        dominant_label = row.idxmax()
        dominant_count = int(row.max())
        dominant_share = dominant_count / cluster_total if cluster_total > 0 else 0.0

        shares = row / cluster_total if cluster_total > 0 else row
        entropy = calculate_entropy_from_shares(shares)

        rows.append(
            {
                CLUSTER_COLUMN: cluster_id,
                "dominant_ai_label": dominant_label,
                "dominant_label_count": dominant_count,
                "cluster_total": cluster_total,
                "dominant_label_share": dominant_share,
                "cluster_label_entropy": entropy,
            }
        )

    mapping = pd.DataFrame(rows)
    mapping.to_csv(CLUSTER_DOMINANT_LABEL_FILE, sep=CSV_SEPARATOR, index=False)

    return mapping


def save_ai_label_dominant_cluster_mapping(ai_label_cluster_counts):
    """
    For each AI label, identify the largest k-means cluster inside that label.

    This is useful for studying whether an AI label is concentrated in one
    cluster or dispersed across many clusters.
    """
    rows = []

    for ai_label, row in ai_label_cluster_counts.iterrows():
        label_total = int(row.sum())
        dominant_cluster = int(row.idxmax())
        dominant_cluster_count = int(row.max())
        dominant_cluster_share = (
            dominant_cluster_count / label_total if label_total > 0 else 0.0
        )

        shares = row / label_total if label_total > 0 else row
        entropy = calculate_entropy_from_shares(shares)

        nonzero_clusters = int((row > 0).sum())

        rows.append(
            {
                "ai_label": ai_label,
                "dominant_kmeans_cluster": dominant_cluster,
                "dominant_cluster_count": dominant_cluster_count,
                "ai_label_total": label_total,
                "dominant_cluster_share_within_ai_label": dominant_cluster_share,
                "nonzero_kmeans_clusters": nonzero_clusters,
                "ai_label_cluster_entropy": entropy,
            }
        )

    mapping = pd.DataFrame(rows)
    mapping.to_csv(AI_LABEL_DOMINANT_CLUSTER_FILE, sep=CSV_SEPARATOR, index=False)

    return mapping


# =============================================================================
# Save fitted objects
# =============================================================================

def save_models_and_matrices(vectorizer, kmeans, all_tfidf):
    """
    Save fitted vectorizer, k-means model, and the full TF-IDF matrix.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, TFIDF_MODEL_FILE)
    joblib.dump(kmeans, KMEANS_MODEL_FILE)

    sparse.save_npz(ALL_TFIDF_FILE, all_tfidf)


# =============================================================================
# Main workflow
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_and_filter_data()
    data = add_descriptive_periods(data)

    print(f"Rows after word-count filter: {len(data)}")
    print(f"Years covered: {data[YEAR_COLUMN].min()}-{data[YEAR_COLUMN].max()}")
    print(f"AI labels: {data[AI_LABEL_COLUMN].nunique()}")

    vectorizer, all_tfidf = fit_tfidf_vectorizer(data)

    print(f"TF-IDF features: {len(vectorizer.get_feature_names_out())}")

    data, kmeans = fit_kmeans(data, all_tfidf)

    print(f"KMeans clusters: {N_CLUSTERS}")

    save_cluster_top_terms(kmeans, vectorizer)
    save_cluster_counts(data)

    (
        cluster_ai_label_counts,
        _cluster_ai_label_row_shares,
        ai_label_cluster_counts,
        _ai_label_cluster_row_shares,
    ) = save_cluster_ai_label_crosstabs(data)

    save_cluster_dominant_label_mapping(cluster_ai_label_counts)
    save_ai_label_dominant_cluster_mapping(ai_label_cluster_counts)

    save_models_and_matrices(vectorizer, kmeans, all_tfidf)

    data.to_csv(PROCESSED_OUTPUT_FILE, sep=CSV_SEPARATOR, index=False)

    print(f"Saved processed data to: {PROCESSED_OUTPUT_FILE}")
    print(f"Saved cluster top terms to: {CLUSTER_TERMS_FILE}")
    print(f"Saved cluster counts to: {CLUSTER_COUNTS_FILE}")
    print(f"Saved cluster x AI label crosstab to: {CLUSTER_AI_LABEL_COUNTS_FILE}")
    print(f"Saved AI label x cluster crosstab to: {AI_LABEL_CLUSTER_COUNTS_FILE}")
    print(f"Saved cluster dominant label mapping to: {CLUSTER_DOMINANT_LABEL_FILE}")
    print(f"Saved AI label dominant cluster mapping to: {AI_LABEL_DOMINANT_CLUSTER_FILE}")
    print(f"Saved models to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
