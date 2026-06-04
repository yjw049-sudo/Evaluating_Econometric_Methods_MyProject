from pathlib import Path
import math
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


# =============================================================================
# Parameters
# =============================================================================

PROJECT_DIR = Path(r"E:\study\S2\Evaluating-Econometric-Methods_data\Evaluating_Econometric_Methods_MyProject")
PROCESSED_DATA_DIR = PROJECT_DIR / "output" / "processed_data"
FINAL_DIR = PROJECT_DIR / "Final"

cluster_groups = 10
ngrams = 2
min_words_per_speech = 20
min_df = 5
max_df = 0.8

# Change these variables when you decide the sample design.
model_years = [1948, 1949, 1950]
train_fraction = 0.8
random_seed = 1
run_out_of_sample = False
out_of_sample_years = []

custom_stop_words = {
    # Parliamentary titles and address terms that usually do not describe speech topics.
    "hon", "member", "mr", "speaker", "chairman", "minist",

    # Very common debate words that make clusters less topic-specific.
    "would", "read", "think", "know", "made", "say",

    # Clear transcription/OCR noise found in the first clustering result.
    "ye", "b", "aba", "surditi", "newsom", "oh", "zurich",
}

vectorizer_path = FINAL_DIR / "vectorizer.sav"
model_path = FINAL_DIR / "NLTK_Model.sav"
cluster_path = FINAL_DIR / f"NLTK_Cluster_{cluster_groups}.csv"
prediction_path = FINAL_DIR / "Prediction.csv"
train_prediction_path = FINAL_DIR / "Prediction_train.csv"
test_prediction_path = FINAL_DIR / "Prediction_test.csv"
out_of_sample_prediction_path = FINAL_DIR / "Prediction_out_of_sample.csv"
cluster_frequency_path = FINAL_DIR / "Cluster_Frequency.png"
word_frequency_path = FINAL_DIR / "Word_Frequency.png"


# =============================================================================
# Data preparation
# =============================================================================

def get_processed_files(years):
    processed_files = []

    for year in years:
        year_files = sorted(PROCESSED_DATA_DIR.glob(f"{year}*.csv"))
        processed_files.extend(year_files)

    return processed_files


def remove_custom_stop_words(text):
    words = text.split()
    filtered_words = []

    for word in words:
        if word not in custom_stop_words:
            filtered_words.append(word)

    return " ".join(filtered_words)


def clean_processed_data(data):
    data = data.dropna(subset=["speechtext"]).copy()
    data["speechtext"] = data["speechtext"].astype(str).str.strip()
    data = data[data["speechtext"] != ""].copy()

    data["speechtext"] = data["speechtext"].map(remove_custom_stop_words)
    data["speechtext"] = data["speechtext"].str.strip()
    data = data[data["speechtext"] != ""].copy()

    data["word_count"] = data["speechtext"].str.split().str.len()
    data = data[data["word_count"] >= min_words_per_speech].copy()

    return data


def read_processed_data(years):
    processed_files = get_processed_files(years)

    if not processed_files:
        return pd.DataFrame()

    data_frames = []
    for file_path in processed_files:
        data_frame = pd.read_csv(file_path, delimiter=";")
        data_frames.append(data_frame)

    data = pd.concat(data_frames, ignore_index=True)
    data = clean_processed_data(data)

    return data


def split_train_test_by_year(data):
    if train_fraction <= 0 or train_fraction >= 1:
        raise ValueError("train_fraction must be between 0 and 1, for example 0.8.")

    train_frames = []
    test_frames = []

    for year, year_data in data.groupby("year"):
        train_data = year_data.sample(
            frac=train_fraction,
            replace=False,
            random_state=random_seed,
        )
        test_data = year_data.drop(train_data.index)

        train_frames.append(train_data)
        test_frames.append(test_data)

    train_data = pd.concat(train_frames, ignore_index=True)
    test_data = pd.concat(test_frames, ignore_index=True)

    train_data["sample"] = "train"
    test_data["sample"] = "test"

    return train_data, test_data


def load_data_samples():
    model_data = read_processed_data(model_years)

    if model_data.empty:
        raise FileNotFoundError("No model files were found. Please check model_years.")

    train_data, test_data = split_train_test_by_year(model_data)

    if run_out_of_sample:
        out_of_sample_data = read_processed_data(out_of_sample_years)
    else:
        out_of_sample_data = pd.DataFrame()

    if not out_of_sample_data.empty:
        out_of_sample_data["sample"] = "out_of_sample"

    if train_data.empty:
        raise ValueError("The training sample is empty. Please check train_fraction.")

    return train_data, test_data, out_of_sample_data


# =============================================================================
# Model training and prediction
# =============================================================================

def build_vectorizer():
    token = RegexpTokenizer("[a-zA-Z]+")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        tokenizer=token.tokenize,
        token_pattern=None,
        ngram_range=(1, ngrams),
        min_df=min_df,
        max_df=max_df,
    )

    return vectorizer


def train_model(train_data):
    vectorizer = build_vectorizer()

    start_time = time.time()
    text_counts = vectorizer.fit_transform(train_data["speechtext"])
    end_time = time.time()

    print(f"Duration of transformation: {end_time - start_time:8.2f} sec.")
    print(
        "We have in total "
        + str(text_counts.shape[1])
        + " words in "
        + str(text_counts.shape[0])
        + " speeches"
    )

    model = KMeans(
        n_clusters=cluster_groups,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=random_seed,
    )

    start_time = time.time()
    model.fit(text_counts)
    end_time = time.time()

    print(f"Duration of training: {end_time - start_time:8.2f} sec.")

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)

    return vectorizer, model


def predict_clusters(vectorizer, model, train_data, test_data, out_of_sample_data):
    prediction_frames = [train_data, test_data, out_of_sample_data]
    prediction_frames = [data_frame for data_frame in prediction_frames if not data_frame.empty]

    total = pd.concat(prediction_frames, ignore_index=True)
    text_counts = vectorizer.transform(total["speechtext"].values.astype("U"))
    total["prediction"] = model.predict(text_counts)

    return total


# =============================================================================
# Output tables
# =============================================================================

def save_cluster_words(vectorizer, model):
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    cluster_words = {}
    for cluster_id in range(cluster_groups):
        words = []

        for word_index in order_centroids[cluster_id, :10]:
            words.append(terms[word_index])

        cluster_words[cluster_id] = words
        print(f"Cluster {cluster_id}: " + ", ".join(words))

    cluster_words_data = pd.DataFrame(cluster_words)
    cluster_words_data.to_csv(cluster_path, index=False)


def save_predictions(total):
    total.to_csv(prediction_path, index=False)
    total[total["sample"] == "train"].to_csv(train_prediction_path, index=False)
    total[total["sample"] == "test"].to_csv(test_prediction_path, index=False)

    if run_out_of_sample and "out_of_sample" in total["sample"].unique():
        total[total["sample"] == "out_of_sample"].to_csv(out_of_sample_prediction_path, index=False)


# =============================================================================
# Plots
# =============================================================================

def create_cluster_axes():
    if cluster_groups == 40:
        return plt.subplots(5, 8, figsize=(50, 25), sharex=True)

    if cluster_groups == 20:
        return plt.subplots(4, 5, figsize=(30, 15), sharex=True)

    if cluster_groups == 10:
        return plt.subplots(2, 5, figsize=(30, 15), sharex=True)

    n_cols = math.ceil(math.sqrt(cluster_groups))
    n_rows = math.ceil(cluster_groups / n_cols)

    return plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True)


def plot_cluster_frequencies(total):
    print("Produce plot of cluster history across time")

    plot_data = total.groupby(["year", "prediction"], as_index=False)["basepk"].count()
    year_counts = plot_data.groupby("year")["basepk"].sum()
    plot_data["year_sum"] = plot_data["year"].map(year_counts)
    plot_data["fraction"] = plot_data["basepk"] / plot_data["year_sum"]

    fig, axes = create_cluster_axes()
    axes = axes.flatten()

    max_share = plot_data["fraction"].max()
    max_share = math.ceil(max_share * 10) / 10
    min_year = total["year"].min()
    max_year = total["year"].max()

    plt.setp(axes, xlim=(min_year, max_year), ylim=(0, max_share))

    for cluster_id in range(cluster_groups):
        cluster_data = plot_data[plot_data["prediction"] == cluster_id]

        ax = axes[cluster_id]
        ax.plot(cluster_data["year"].tolist(), cluster_data["fraction"].tolist())
        ax.set_title(f"Cluster {cluster_id}", fontsize=11)
        ax.tick_params(axis="both", which="major", labelsize=10)

        for spine in "top right left".split():
            ax.spines[spine].set_visible(False)

    for empty_axis_id in range(cluster_groups, len(axes)):
        axes[empty_axis_id].set_visible(False)

    fig.suptitle("Frequency of Clusters", fontsize=20)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.savefig(cluster_frequency_path, dpi=300)
    plt.close(fig)


def count_word_frequencies(total, clusters):
    frequency_data = {}

    for cluster_id in tqdm(range(cluster_groups)):
        speeches = total[total["prediction"] == cluster_id]["speechtext"]
        combined_text = " " + " ".join(speeches.astype(str).tolist()) + " "

        wordlist = clusters[str(cluster_id)].dropna().tolist()
        frequencies = []

        for word in wordlist:
            frequency = combined_text.count(" " + word + " ")
            frequencies.append(frequency)

        total_frequency = sum(frequencies)
        if total_frequency > 0:
            frequencies = [frequency / total_frequency for frequency in frequencies]

        frequency_data[cluster_id] = {
            "words": wordlist,
            "frequencies": frequencies,
        }

    return frequency_data


def plot_word_frequencies(total):
    print("Produce word frequency plot")

    clusters = pd.read_csv(cluster_path)
    frequency_data = count_word_frequencies(total, clusters)

    fig, axes = create_cluster_axes()
    axes = axes.flatten()

    for cluster_id in range(cluster_groups):
        words = frequency_data[cluster_id]["words"]
        frequencies = frequency_data[cluster_id]["frequencies"]

        ax = axes[cluster_id]
        ax.barh(words, frequencies, height=0.7)
        ax.invert_yaxis()
        ax.set_title(f"Cluster {cluster_id}", fontsize=11)
        ax.tick_params(axis="both", which="major", labelsize=10)

        for spine in "top right left".split():
            ax.spines[spine].set_visible(False)

    for empty_axis_id in range(cluster_groups, len(axes)):
        axes[empty_axis_id].set_visible(False)

    fig.suptitle("Frequency of the Most Representative Words in each Cluster", fontsize=20)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.savefig(word_frequency_path, dpi=300)
    plt.close(fig)


# =============================================================================
# Main workflow
# =============================================================================

def main():
    FINAL_DIR.mkdir(exist_ok=True)

    train_data, test_data, out_of_sample_data = load_data_samples()
    vectorizer, model = train_model(train_data)

    save_cluster_words(vectorizer, model)

    total = predict_clusters(vectorizer, model, train_data, test_data, out_of_sample_data)
    save_predictions(total)

    plot_cluster_frequencies(total)
    plot_word_frequencies(total)


if __name__ == "__main__":
    main()
