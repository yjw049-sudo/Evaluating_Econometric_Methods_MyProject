from pathlib import Path
import importlib.util
import time

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =============================================================================
# Import the main model pipeline
# =============================================================================

MODEL_SCRIPT_PATH = Path(__file__).resolve().parent / "model_speech.py"

spec = importlib.util.spec_from_file_location("model_speech_pipeline", MODEL_SCRIPT_PATH)
model_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_pipeline)


# =============================================================================
# Temporal validation parameters
# =============================================================================

cluster_group_candidates = [15, 20, 25, 30]

# Each tuple is: (training years, validation years).
validation_splits = [
    ([1948], [1949]),
    ([1948, 1949], [1950]),
]

silhouette_sample_size = 5000
top_words_per_cluster = 10

cv_results_path = model_pipeline.FINAL_DIR / "Cluster_Group_Temporal_CV_Results.csv"


# =============================================================================
# Model evaluation
# =============================================================================

def calculate_average_distance_to_center(text_counts, model):
    distances = model.transform(text_counts)
    closest_distances = distances.min(axis=1)

    return closest_distances.mean()


def calculate_silhouette(text_counts, labels):
    if len(set(labels)) < 2:
        return None

    sample_size = min(silhouette_sample_size, text_counts.shape[0])

    return silhouette_score(
        text_counts,
        labels,
        sample_size=sample_size,
        random_state=model_pipeline.random_seed,
    )


def get_cluster_summary(labels, cluster_groups):
    label_counts = pd.Series(labels).value_counts().sort_index()
    label_shares = label_counts / label_counts.sum()

    largest_cluster_share = label_shares.max()
    smallest_cluster_share = label_shares.min()
    empty_clusters = cluster_groups - label_counts.shape[0]

    return largest_cluster_share, smallest_cluster_share, empty_clusters


def get_top_words(vectorizer, model):
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    cluster_summaries = []
    for cluster_id in range(model.n_clusters):
        words = []

        for word_index in order_centroids[cluster_id, :top_words_per_cluster]:
            words.append(terms[word_index])

        cluster_summaries.append(f"Cluster {cluster_id}: " + ", ".join(words))

    return " | ".join(cluster_summaries)


def evaluate_cluster_group(cluster_groups, train_data, validation_data):
    vectorizer = model_pipeline.build_vectorizer()
    train_text_counts = vectorizer.fit_transform(train_data["speechtext"])
    validation_text_counts = vectorizer.transform(validation_data["speechtext"].values.astype("U"))

    model = KMeans(
        n_clusters=cluster_groups,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=model_pipeline.random_seed,
    )

    start_time = time.time()
    train_labels = model.fit_predict(train_text_counts)
    validation_labels = model.predict(validation_text_counts)
    end_time = time.time()

    train_largest_share, train_smallest_share, train_empty_clusters = get_cluster_summary(
        train_labels,
        cluster_groups,
    )
    validation_largest_share, validation_smallest_share, validation_empty_clusters = get_cluster_summary(
        validation_labels,
        cluster_groups,
    )

    result = {
        "cluster_groups": cluster_groups,
        "train_rows": train_text_counts.shape[0],
        "validation_rows": validation_text_counts.shape[0],
        "vocabulary_size": train_text_counts.shape[1],
        "train_inertia": model.inertia_,
        "train_average_distance": calculate_average_distance_to_center(train_text_counts, model),
        "validation_average_distance": calculate_average_distance_to_center(validation_text_counts, model),
        "train_silhouette": calculate_silhouette(train_text_counts, train_labels),
        "validation_silhouette": calculate_silhouette(validation_text_counts, validation_labels),
        "train_largest_cluster_share": train_largest_share,
        "validation_largest_cluster_share": validation_largest_share,
        "train_smallest_cluster_share": train_smallest_share,
        "validation_smallest_cluster_share": validation_smallest_share,
        "train_empty_clusters": train_empty_clusters,
        "validation_empty_clusters": validation_empty_clusters,
        "runtime_seconds": end_time - start_time,
        "top_words": get_top_words(vectorizer, model),
    }

    return result


def evaluate_temporal_split(cluster_groups, train_years, validation_years):
    train_data = model_pipeline.read_processed_data(train_years)
    validation_data = model_pipeline.read_processed_data(validation_years)

    if train_data.empty:
        raise FileNotFoundError(f"No training files were found for years: {train_years}")

    if validation_data.empty:
        raise FileNotFoundError(f"No validation files were found for years: {validation_years}")

    result = evaluate_cluster_group(cluster_groups, train_data, validation_data)
    result["train_years"] = ", ".join(str(year) for year in train_years)
    result["validation_years"] = ", ".join(str(year) for year in validation_years)

    return result


# =============================================================================
# Main workflow
# =============================================================================

def main():
    model_pipeline.FINAL_DIR.mkdir(exist_ok=True)

    results = []

    for train_years, validation_years in validation_splits:
        for cluster_groups in cluster_group_candidates:
            print(
                "Evaluating "
                + f"cluster_groups={cluster_groups}, "
                + f"train_years={train_years}, "
                + f"validation_years={validation_years}"
            )

            result = evaluate_temporal_split(
                cluster_groups,
                train_years,
                validation_years,
            )
            results.append(result)

    results_data = pd.DataFrame(results)
    results_data.to_csv(cv_results_path, index=False)

    print(f"Saved temporal validation results to: {cv_results_path}")


if __name__ == "__main__":
    main()
