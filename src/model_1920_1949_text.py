from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


PROJECT_DIR = Path(
    r"E:\study\S2\Evaluating-Econometric-Methods_data\Evaluating_Econometric_Methods_MyProject"
)

INPUT_FILE = PROJECT_DIR / "output" / "1920-1949-v1" / "speeches_1920_1949_merged.csv"
OUTPUT_DIR = PROJECT_DIR / "output" / "1920-1949-v1"
MODEL_DIR = OUTPUT_DIR / "models"

PROCESSED_OUTPUT_FILE = OUTPUT_DIR / "speeches_1920_1949_second_stage.csv"
CLUSTER_TERMS_FILE = OUTPUT_DIR / "kmeans_cluster_top_terms.csv"
CLUSTER_COUNTS_FILE = OUTPUT_DIR / "kmeans_cluster_counts_by_year.csv"

TFIDF_MODEL_FILE = MODEL_DIR / "tfidf_vectorizer_1930_1945.joblib"
KMEANS_MODEL_FILE = MODEL_DIR / "kmeans_1930_1945.joblib"
LOGISTIC_MODEL_FILE = MODEL_DIR / "logistic_regression_1930_1945.joblib"
TRAIN_TFIDF_FILE = MODEL_DIR / "tfidf_train_1930_1945.npz"
ALL_TFIDF_FILE = MODEL_DIR / "tfidf_all_1920_1949.npz"

TRAIN_START_YEAR = 1930
TRAIN_END_YEAR = 1945
EARLY_TEST_START_YEAR = 1920
EARLY_TEST_END_YEAR = 1929
LATE_TEST_START_YEAR = 1945
LATE_TEST_END_YEAR = 1949

MIN_SPEECHTEXT_LENGTH = 10
N_CLUSTERS = 20
TOP_TERMS_PER_CLUSTER = 30
RANDOM_STATE = 42


def load_and_filter_data():
    """Load data, keep longer speeches, and add speechtext_length."""
    data = pd.read_csv(INPUT_FILE, sep=";")

    data["speechtext"] = data["speechtext"].fillna("").astype(str)
    data["speechtext_length"] = data["speechtext"].str.len()

    filtered_data = data[data["speechtext_length"] > MIN_SPEECHTEXT_LENGTH].copy()
    filtered_data = filtered_data.reset_index(drop=True)

    return filtered_data


def add_period_labels(data):
    """Add readable period labels for checking the train/test split."""
    data = data.copy()
    data["model_period"] = "outside_selected_periods"

    early_test_mask = data["year"].between(EARLY_TEST_START_YEAR, EARLY_TEST_END_YEAR)
    train_mask = data["year"].between(TRAIN_START_YEAR, TRAIN_END_YEAR)
    late_test_mask = data["year"].between(LATE_TEST_START_YEAR, LATE_TEST_END_YEAR)
    overlap_mask = train_mask & late_test_mask

    data["is_train_period"] = train_mask
    data["is_early_test_period"] = early_test_mask
    data["is_late_test_period"] = late_test_mask

    data.loc[early_test_mask, "model_period"] = "test_1920_1929"
    data.loc[train_mask, "model_period"] = "train_1930_1945"
    data.loc[late_test_mask, "model_period"] = "test_1945_1949"
    data.loc[overlap_mask, "model_period"] = "train_1930_1945_and_test_1945_1949"

    return data


def fit_tfidf_vectorizer(data):
    """Fit TF-IDF on 1930-1945 texts, then transform all texts."""
    train_mask = data["year"].between(TRAIN_START_YEAR, TRAIN_END_YEAR)
    train_texts = data.loc[train_mask, "speechtext"]
    all_texts = data["speechtext"]

    vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=5,
        max_df=0.90,
        stop_words="english",
    )

    train_tfidf = vectorizer.fit_transform(train_texts)
    all_tfidf = vectorizer.transform(all_texts)

    return vectorizer, train_tfidf, all_tfidf, train_mask


def fit_and_predict_kmeans(data, train_tfidf, all_tfidf, train_mask):
    """Fit KMeans on 1930-1945 only, then assign clusters to all rows."""
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    kmeans.fit(train_tfidf)

    data = data.copy()
    data["kmeans_cluster"] = kmeans.predict(all_tfidf)

    train_clusters = data.loc[train_mask, "kmeans_cluster"]
    print("KMeans clusters predicted for 1930-1945:", len(train_clusters))

    early_test_mask = data["year"].between(EARLY_TEST_START_YEAR, EARLY_TEST_END_YEAR)
    late_test_mask = data["year"].between(LATE_TEST_START_YEAR, LATE_TEST_END_YEAR)
    print("KMeans clusters predicted for 1920-1929:", early_test_mask.sum())
    print("KMeans clusters predicted for 1945-1949:", late_test_mask.sum())

    return data, kmeans


def save_cluster_top_terms(kmeans, vectorizer):
    """Save top TF-IDF terms for each KMeans cluster."""
    feature_names = vectorizer.get_feature_names_out()
    rows = []

    for cluster_id, center in enumerate(kmeans.cluster_centers_):
        top_indices = center.argsort()[::-1][:TOP_TERMS_PER_CLUSTER]

        for rank, term_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "cluster": cluster_id,
                    "rank": rank,
                    "term": feature_names[term_index],
                    "tfidf_center_weight": center[term_index],
                }
            )

    cluster_terms = pd.DataFrame(rows)
    cluster_terms.to_csv(CLUSTER_TERMS_FILE, sep=";", index=False)


def fit_and_predict_logistic_regression(data, train_tfidf, all_tfidf, train_mask):
    """Train Logistic Regression on 1930-1945 and predict category for all rows."""
    train_labels = data.loc[train_mask, "category"]

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    logistic_model.fit(train_tfidf, train_labels)

    data = data.copy()
    data["logistic_regression_category"] = logistic_model.predict(all_tfidf)

    predicted_probabilities = logistic_model.predict_proba(all_tfidf)
    data["logistic_regression_confidence"] = predicted_probabilities.max(axis=1)

    return data, logistic_model


def save_cluster_counts(data):
    """Save a simple year-by-cluster count table for later plotting."""
    cluster_counts = (
        data.groupby(["year", "kmeans_cluster"])
        .size()
        .reset_index(name="count")
        .sort_values(["year", "kmeans_cluster"])
    )

    cluster_counts.to_csv(CLUSTER_COUNTS_FILE, sep=";", index=False)


def save_models_and_matrices(vectorizer, kmeans, logistic_model, train_tfidf, all_tfidf):
    """Save fitted models and TF-IDF matrices for reproducibility."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, TFIDF_MODEL_FILE)
    joblib.dump(kmeans, KMEANS_MODEL_FILE)
    joblib.dump(logistic_model, LOGISTIC_MODEL_FILE)

    sparse.save_npz(TRAIN_TFIDF_FILE, train_tfidf)
    sparse.save_npz(ALL_TFIDF_FILE, all_tfidf)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_and_filter_data()
    data = add_period_labels(data)

    print(f"Rows after speechtext length filter: {len(data)}")

    vectorizer, train_tfidf, all_tfidf, train_mask = fit_tfidf_vectorizer(data)
    print(f"Rows in training period 1930-1945: {train_mask.sum()}")
    print(f"TF-IDF features: {len(vectorizer.get_feature_names_out())}")

    data, kmeans = fit_and_predict_kmeans(data, train_tfidf, all_tfidf, train_mask)
    save_cluster_top_terms(kmeans, vectorizer)
    save_cluster_counts(data)

    data, logistic_model = fit_and_predict_logistic_regression(
        data,
        train_tfidf,
        all_tfidf,
        train_mask,
    )

    save_models_and_matrices(vectorizer, kmeans, logistic_model, train_tfidf, all_tfidf)

    data.to_csv(PROCESSED_OUTPUT_FILE, sep=";", index=False)

    print(f"Saved second-stage data to: {PROCESSED_OUTPUT_FILE}")
    print(f"Saved cluster top terms to: {CLUSTER_TERMS_FILE}")
    print(f"Saved models to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
