import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# =============================================================================
# User settings
# =============================================================================

N_CLUSTERS = 30
MAX_FEATURES = 20_000
NGRAM_RANGE = (1, 2)
MIN_DOCUMENT_FREQUENCY = 50
MAX_DOCUMENT_FREQUENCY = 0.70
TOP_WORDS_PER_CLUSTER = 15
RANDOM_STATE = 42


# =============================================================================
# Paths and column settings
# =============================================================================

WORK_DIR = Path(__file__).resolve().parent
INPUT_FILE = WORK_DIR / "output" / "speeches_with_parlinfo.csv"
OUTPUT_DIR = WORK_DIR / "output" / "Kmeans_30"

OUTPUT_DATA_FILE = OUTPUT_DIR / "speeches_with_parlinfo_kmeans.csv"
TEMP_OUTPUT_DATA_FILE = OUTPUT_DIR / "speeches_with_parlinfo_kmeans.tmp.csv"
CLUSTER_SUMMARY_FILE = OUTPUT_DIR / "cluster_summary.csv"
TFIDF_MODEL_FILE = OUTPUT_DIR / "tfidf_vectorizer.joblib"
KMEANS_MODEL_FILE = OUTPUT_DIR / "kmeans_model.joblib"
MODEL_SETTINGS_FILE = OUTPUT_DIR / "model_settings.json"

CSV_SEPARATOR = ","
TEXT_COLUMN = "speechtext"
MATCH_COLUMN = "is_mp_matched"
CLUSTER_COLUMN = "cluster"
DISTANCE_COLUMN = "cluster_distance"
SECOND_CLUSTER_COLUMN = "second_cluster"
SECOND_DISTANCE_COLUMN = "second_cluster_distance"

OUTPUT_CHUNK_SIZE = 20_000
DISTANCE_BATCH_SIZE = 20_000


# =============================================================================
# Data loading and validation
# =============================================================================

def get_mp_matched_mask(match_values):
    """
    Identify rows where is_mp_matched is True.

    This supports both boolean values and text values such as "True".
    Missing values are treated as not matched.
    """
    normalized_values = (
        match_values
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized_values.isin(["true", "1", "1.0"])


def load_training_text():
    """Load text only from rows where is_mp_matched is True."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file was not found: {INPUT_FILE}")

    text_data = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        usecols=[TEXT_COLUMN, MATCH_COLUMN],
        low_memory=True,
    )

    matched_mask = get_mp_matched_mask(text_data[MATCH_COLUMN])
    unmatched_count = int((~matched_mask).sum())
    text_data = text_data.loc[matched_mask, [TEXT_COLUMN]].copy()

    print(f"Rows excluded because {MATCH_COLUMN} is not True: {unmatched_count}")

    text_data[TEXT_COLUMN] = (
        text_data[TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_text_count = text_data[TEXT_COLUMN].eq("").sum()

    if empty_text_count > 0:
        raise ValueError(
            f"{empty_text_count} rows have missing or empty {TEXT_COLUMN}. "
            "Please clean the input data before running KMeans."
        )

    return text_data[TEXT_COLUMN]


# =============================================================================
# TF-IDF and KMeans
# =============================================================================

def fit_tfidf(text_data):
    """Fit the TF-IDF vectorizer using the reference script's settings."""
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DOCUMENT_FREQUENCY,
        max_df=MAX_DOCUMENT_FREQUENCY,
        stop_words="english",
    )

    tfidf_matrix = vectorizer.fit_transform(text_data)
    return vectorizer, tfidf_matrix


def fit_kmeans(tfidf_matrix):
    """Fit KMeans and return the assigned cluster for every speech."""
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        init="k-means++",
        max_iter=300,
        n_init=50,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    return kmeans, cluster_labels


def calculate_cluster_distances(
    kmeans,
    tfidf_matrix,
    cluster_labels,
):
    """
    Calculate distances to the nearest and second-nearest centroids.

    Distances are calculated in batches so that the complete
    rows-by-clusters distance matrix is not kept in memory.
    """
    if kmeans.n_clusters < 2:
        raise ValueError(
            "At least two KMeans clusters are required to calculate "
            "the second-nearest cluster."
        )

    row_count = tfidf_matrix.shape[0]
    cluster_distances = np.empty(row_count, dtype=np.float64)
    second_cluster_labels = np.empty(row_count, dtype=np.int64)
    second_cluster_distances = np.empty(row_count, dtype=np.float64)

    for start_row in range(0, row_count, DISTANCE_BATCH_SIZE):
        end_row = min(start_row + DISTANCE_BATCH_SIZE, row_count)

        batch_distances = kmeans.transform(
            tfidf_matrix[start_row:end_row]
        )
        batch_labels = cluster_labels[start_row:end_row]
        batch_row_positions = np.arange(end_row - start_row)

        cluster_distances[start_row:end_row] = batch_distances[
            batch_row_positions,
            batch_labels,
        ]

        # Exclude each row's assigned centroid. The nearest centroid left in
        # the matrix is that speech's second-nearest cluster.
        batch_distances[batch_row_positions, batch_labels] = np.inf
        batch_second_labels = np.argmin(batch_distances, axis=1)

        second_cluster_labels[start_row:end_row] = batch_second_labels
        second_cluster_distances[start_row:end_row] = batch_distances[
            batch_row_positions,
            batch_second_labels,
        ]

        print(
            f"Calculated distances for rows "
            f"{start_row + 1}-{end_row}/{row_count}."
        )

    return (
        cluster_distances,
        second_cluster_labels,
        second_cluster_distances,
    )


# =============================================================================
# Outputs
# =============================================================================

def save_models(vectorizer, kmeans):
    """Save the fitted TF-IDF vectorizer, KMeans model, and settings."""
    joblib.dump(vectorizer, TFIDF_MODEL_FILE)
    joblib.dump(kmeans, KMEANS_MODEL_FILE)

    model_settings = {
        "input_file": str(INPUT_FILE),
        "text_column": TEXT_COLUMN,
        "training_filter": f"{MATCH_COLUMN} == True",
        "n_clusters": N_CLUSTERS,
        "max_features": MAX_FEATURES,
        "ngram_range": list(NGRAM_RANGE),
        "min_document_frequency": MIN_DOCUMENT_FREQUENCY,
        "max_document_frequency": MAX_DOCUMENT_FREQUENCY,
        "stop_words": "english",
        "top_words_per_cluster": TOP_WORDS_PER_CLUSTER,
        "random_state": RANDOM_STATE,
        "distance_definition": (
            "Euclidean distance from each TF-IDF vector "
            "to its assigned KMeans centroid"
        ),
        "second_distance_definition": (
            "Euclidean distance from each TF-IDF vector "
            "to its second-nearest KMeans centroid"
        ),
    }

    with MODEL_SETTINGS_FILE.open("w", encoding="utf-8") as output_file:
        json.dump(model_settings, output_file, indent=2)


def save_cluster_summary(cluster_labels, kmeans, vectorizer):
    """
    Save cluster share and the top 10 centroid terms for every cluster.
    """
    feature_names = vectorizer.get_feature_names_out()
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    total_count = len(cluster_labels)
    rows = []

    for cluster_id in range(N_CLUSTERS):
        center = kmeans.cluster_centers_[cluster_id]
        top_indices = center.argsort()[::-1][:TOP_WORDS_PER_CLUSTER]
        top_words = [feature_names[index] for index in top_indices]

        rows.append(
            {
                CLUSTER_COLUMN: cluster_id,
                "share": cluster_counts.get(cluster_id, 0) / total_count,
                "top_10_words": ",".join(top_words),
            }
        )

    cluster_summary = pd.DataFrame(rows)
    cluster_summary.to_csv(
        CLUSTER_SUMMARY_FILE,
        sep=CSV_SEPARATOR,
        index=False,
        encoding="utf-8-sig",
    )


def save_data_with_clusters(
    cluster_labels,
    cluster_distances,
    second_cluster_labels,
    second_cluster_distances,
):
    """
    Save matched-MP rows with nearest and second-nearest cluster results.

    The same is_mp_matched filter used for training is applied here so that
    cluster assignments remain aligned with the output rows.
    """
    if TEMP_OUTPUT_DATA_FILE.exists():
        TEMP_OUTPUT_DATA_FILE.unlink()

    write_header = True
    start_row = 0

    data_chunks = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR,
        chunksize=OUTPUT_CHUNK_SIZE,
        low_memory=True,
    )

    for chunk_number, data_chunk in enumerate(data_chunks, start=1):
        if MATCH_COLUMN not in data_chunk.columns:
            raise ValueError(
                f"Input file is missing required column: {MATCH_COLUMN}"
            )

        matched_mask = get_mp_matched_mask(data_chunk[MATCH_COLUMN])
        matched_chunk = data_chunk.loc[matched_mask].copy()

        if matched_chunk.empty:
            print(
                f"Skipped output chunk {chunk_number}; "
                f"no rows have {MATCH_COLUMN} == True."
            )
            continue

        end_row = start_row + len(matched_chunk)

        if end_row > len(cluster_labels):
            raise ValueError(
                "Matched output rows exceed the number of cluster assignments. "
                "Check that the training and output filters are identical."
            )

        matched_chunk[CLUSTER_COLUMN] = cluster_labels[start_row:end_row]
        matched_chunk[DISTANCE_COLUMN] = cluster_distances[start_row:end_row]
        matched_chunk[SECOND_CLUSTER_COLUMN] = second_cluster_labels[
            start_row:end_row
        ]
        matched_chunk[SECOND_DISTANCE_COLUMN] = second_cluster_distances[
            start_row:end_row
        ]

        matched_chunk.to_csv(
            TEMP_OUTPUT_DATA_FILE,
            sep=CSV_SEPARATOR,
            index=False,
            mode="w" if write_header else "a",
            header=write_header,
            encoding="utf-8-sig" if write_header else "utf-8",
        )

        write_header = False
        start_row = end_row

        print(
            f"Saved output chunk {chunk_number}; "
            f"rows written: {end_row}/{len(cluster_labels)}."
        )

    if start_row != len(cluster_labels):
        raise ValueError(
            "The number of rows read while saving does not match "
            "the number of cluster assignments."
        )

    os.replace(TEMP_OUTPUT_DATA_FILE, OUTPUT_DATA_FILE)


# =============================================================================
# Main workflow
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading training text from: {INPUT_FILE}")
    text_data = load_training_text()
    print(f"Rows used for KMeans: {len(text_data)}")

    print("Fitting TF-IDF vectorizer...")
    vectorizer, tfidf_matrix = fit_tfidf(text_data)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    print("Fitting KMeans model...")
    kmeans, cluster_labels = fit_kmeans(tfidf_matrix)

    print("Calculating nearest and second-nearest cluster distances...")
    (
        cluster_distances,
        second_cluster_labels,
        second_cluster_distances,
    ) = calculate_cluster_distances(
        kmeans,
        tfidf_matrix,
        cluster_labels,
    )

    print("Saving models and summary...")
    save_models(vectorizer, kmeans)
    save_cluster_summary(cluster_labels, kmeans, vectorizer)

    print(
        f"Saving rows where {MATCH_COLUMN} is True "
        "with cluster results..."
    )
    save_data_with_clusters(
        cluster_labels,
        cluster_distances,
        second_cluster_labels,
        second_cluster_distances,
    )

    print(f"Saved clustered data to: {OUTPUT_DATA_FILE}")
    print(f"Saved cluster summary to: {CLUSTER_SUMMARY_FILE}")
    print(f"Saved TF-IDF vectorizer to: {TFIDF_MODEL_FILE}")
    print(f"Saved KMeans model to: {KMEANS_MODEL_FILE}")


if __name__ == "__main__":
    main()
