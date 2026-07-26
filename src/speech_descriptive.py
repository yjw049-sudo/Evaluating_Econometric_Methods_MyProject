"""Create descriptive statistics for the speech-processing pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"

MERGED_FILE = OUTPUT_DIR / "Speech_merged.csv"
CLEANED_FILE = OUTPUT_DIR / "Speech_cleaned.csv"
SAMPLE_FILE = OUTPUT_DIR / "Speech_sample.csv"
CHUNK_FILE = OUTPUT_DIR / "Speech_chunk.csv"
OUTPUT_FILE = OUTPUT_DIR / "descriptive.csv"

WORD_COUNT_COLUMN = "speechtext_word_count"
PERCENTILE_GROUP_COLUMN = "percentile_group"
CATEGORY_COLUMN = "category"
SPEECH_ID_COLUMN = "basepk"
CHUNK_TOTAL_COLUMN = "chunk_total"
CHUNK_INDEX_COLUMN = "chunk_index"
PROCEDURAL_SCORE_COLUMN = "procedural_score"
CLUSTER_SCORE_COLUMN = "cluster_score"
CLUSTER_COLUMN = "cluster"
TOPIC_ERROR_COLUMN = "topic_error"
CATEGORY_ERROR_COLUMN = "category_error"

PERCENTILES = [1, 25, 50, 75, 99]
PERCENTILE_GROUPS = ["0-1", "1-25", "25-50", "50-75", "75-99", "99-100"]
CHUNK_SIZE = 50_000
MIN_CHUNK_TOTAL = 2
TARGET_INFLOW_CLUSTER = 4
ZERO_SD_TOLERANCE = 1e-9


def validate_file(file_path, required_columns=None):
    """Check that a CSV exists and contains the required columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if required_columns:
        available_columns = pd.read_csv(file_path, nrows=0).columns.tolist()
        missing_columns = [
            column
            for column in required_columns
            if column not in available_columns
        ]
        if missing_columns:
            raise ValueError(
                f"{file_path.name} is missing required columns: "
                f"{missing_columns}"
            )


def count_rows(file_path):
    """Count CSV records in chunks, correctly handling multiline text."""
    validate_file(file_path)
    first_column = pd.read_csv(file_path, nrows=0).columns[0]
    row_count = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=[first_column],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        row_count += len(chunk)

    return row_count


def read_sample_statistics():
    """Calculate word-count, percentile-group, and category statistics."""
    required_columns = [
        WORD_COUNT_COLUMN,
        PERCENTILE_GROUP_COLUMN,
        CATEGORY_COLUMN,
    ]
    validate_file(SAMPLE_FILE, required_columns)

    word_count_parts = []
    percentile_group_counts = {}
    category_counts = {}
    sample_row_count = 0

    for chunk in pd.read_csv(
        SAMPLE_FILE,
        usecols=required_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        sample_row_count += len(chunk)

        word_counts = pd.to_numeric(
            chunk[WORD_COUNT_COLUMN], errors="coerce"
        ).dropna()
        word_count_parts.append(word_counts)

        group_labels = chunk[PERCENTILE_GROUP_COLUMN].fillna("Missing")
        for label, count in group_labels.value_counts().items():
            percentile_group_counts[str(label)] = (
                percentile_group_counts.get(str(label), 0) + int(count)
            )

        # Missing values are not categories. Category shares still use the
        # complete stratified sample as their denominator.
        category_labels = chunk[CATEGORY_COLUMN].dropna()
        for label, count in category_labels.value_counts().items():
            category_counts[str(label)] = (
                category_counts.get(str(label), 0) + int(count)
            )

    if sample_row_count == 0:
        raise ValueError(f"{SAMPLE_FILE.name} contains no records")

    all_word_counts = pd.concat(word_count_parts, ignore_index=True)
    if all_word_counts.empty:
        raise ValueError(
            f"{SAMPLE_FILE.name} contains no valid {WORD_COUNT_COLUMN} values"
        )

    quantile_levels = [percentile / 100 for percentile in PERCENTILES]
    word_count_percentiles = all_word_counts.quantile(quantile_levels)

    return (
        sample_row_count,
        word_count_percentiles,
        percentile_group_counts,
        category_counts,
    )


def read_chunk_statistics():
    """Count unique multi-chunk speeches and find the maximum chunk total."""
    required_columns = [SPEECH_ID_COLUMN, CHUNK_TOTAL_COLUMN]
    validate_file(CHUNK_FILE, required_columns)

    multi_chunk_speech_ids = set()
    maximum_chunk_total = None

    for chunk in pd.read_csv(
        CHUNK_FILE,
        usecols=required_columns,
        dtype={SPEECH_ID_COLUMN: "string"},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk_totals = pd.to_numeric(
            chunk[CHUNK_TOTAL_COLUMN], errors="coerce"
        )

        valid_chunk_totals = chunk_totals.dropna()
        if not valid_chunk_totals.empty:
            chunk_maximum = int(valid_chunk_totals.max())
            if (
                maximum_chunk_total is None
                or chunk_maximum > maximum_chunk_total
            ):
                maximum_chunk_total = chunk_maximum

        multi_chunk_ids = chunk.loc[
            chunk_totals.ge(2), SPEECH_ID_COLUMN
        ].dropna()
        multi_chunk_speech_ids.update(multi_chunk_ids.tolist())

    if maximum_chunk_total is None:
        raise ValueError(
            f"{CHUNK_FILE.name} contains no valid {CHUNK_TOTAL_COLUMN} values"
        )

    return len(multi_chunk_speech_ids), maximum_chunk_total


def mean_ci_95(values):
    """Return a mean and its normal-approximation 95% confidence interval."""
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    observation_count = len(clean_values)

    if observation_count == 0:
        return np.nan, np.nan, np.nan, 0

    mean_value = float(clean_values.mean())
    if observation_count == 1:
        return mean_value, np.nan, np.nan, 1

    standard_error = float(
        clean_values.std(ddof=1) / np.sqrt(observation_count)
    )
    margin = 1.96 * standard_error
    return (
        mean_value,
        mean_value - margin,
        mean_value + margin,
        observation_count,
    )


def remove_error_rows(data):
    """Remove rows carrying a Gemini topic or category error message."""
    topic_error = (
        data[TOPIC_ERROR_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    category_error = (
        data[CATEGORY_ERROR_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    return data.loc[~(topic_error | category_error)].copy()


def read_headline_data():
    """Read only the columns required for the headline-results table."""
    full_columns = [
        SPEECH_ID_COLUMN,
        PROCEDURAL_SCORE_COLUMN,
        CLUSTER_SCORE_COLUMN,
        CATEGORY_COLUMN,
        CLUSTER_COLUMN,
        TOPIC_ERROR_COLUMN,
        CATEGORY_ERROR_COLUMN,
    ]
    chunk_columns = [
        *full_columns,
        CHUNK_INDEX_COLUMN,
        CHUNK_TOTAL_COLUMN,
    ]

    validate_file(SAMPLE_FILE, full_columns)
    validate_file(CHUNK_FILE, chunk_columns)

    full_data = pd.read_csv(
        SAMPLE_FILE,
        usecols=full_columns,
        dtype={SPEECH_ID_COLUMN: "string"},
        low_memory=False,
    )
    chunk_data = pd.read_csv(
        CHUNK_FILE,
        usecols=chunk_columns,
        dtype={SPEECH_ID_COLUMN: "string"},
        low_memory=False,
    )

    full_data = full_data.dropna(subset=[SPEECH_ID_COLUMN]).copy()
    chunk_data = chunk_data.dropna(subset=[SPEECH_ID_COLUMN]).copy()
    return full_data, chunk_data


def build_score_panel(full_data, chunk_data, score_column):
    """
    Build one complete multi-chunk speech panel for a score variable.

    Complete speeches contain every chunk index from zero through
    ``chunk_total - 1``. Each chunk has equal weight within its speech.
    """
    full_scores = full_data[
        [SPEECH_ID_COLUMN, score_column]
    ].copy()
    full_scores[score_column] = pd.to_numeric(
        full_scores[score_column],
        errors="coerce",
    )
    full_scores = full_scores.dropna(subset=[score_column])
    full_scores = full_scores.loc[
        full_scores[score_column].between(0, 1, inclusive="both")
    ].copy()

    duplicate_full_ids = full_scores.loc[
        full_scores.duplicated(SPEECH_ID_COLUMN, keep=False),
        SPEECH_ID_COLUMN,
    ].unique()
    full_scores = full_scores.loc[
        ~full_scores[SPEECH_ID_COLUMN].isin(duplicate_full_ids)
    ].copy()

    chunks = chunk_data[
        [
            SPEECH_ID_COLUMN,
            CHUNK_INDEX_COLUMN,
            CHUNK_TOTAL_COLUMN,
            score_column,
        ]
    ].copy()
    for column in [
        CHUNK_INDEX_COLUMN,
        CHUNK_TOTAL_COLUMN,
        score_column,
    ]:
        chunks[column] = pd.to_numeric(chunks[column], errors="coerce")

    chunks = chunks.dropna(
        subset=[
            CHUNK_INDEX_COLUMN,
            CHUNK_TOTAL_COLUMN,
            score_column,
        ]
    ).copy()
    chunks = chunks.loc[
        chunks[score_column].between(0, 1, inclusive="both")
        & chunks[CHUNK_TOTAL_COLUMN].ge(MIN_CHUNK_TOTAL)
    ].copy()
    chunks[CHUNK_INDEX_COLUMN] = chunks[CHUNK_INDEX_COLUMN].astype(int)
    chunks[CHUNK_TOTAL_COLUMN] = chunks[CHUNK_TOTAL_COLUMN].astype(int)

    invalid_ids = set(
        chunks.loc[
            chunks.duplicated(
                [SPEECH_ID_COLUMN, CHUNK_INDEX_COLUMN],
                keep=False,
            ),
            SPEECH_ID_COLUMN,
        ]
    )
    inconsistent_totals = (
        chunks.groupby(SPEECH_ID_COLUMN)[CHUNK_TOTAL_COLUMN]
        .nunique()
        .loc[lambda values: values.ne(1)]
    )
    invalid_ids.update(inconsistent_totals.index)

    invalid_index = (
        chunks[CHUNK_INDEX_COLUMN].lt(0)
        | chunks[CHUNK_INDEX_COLUMN].ge(chunks[CHUNK_TOTAL_COLUMN])
    )
    invalid_ids.update(
        chunks.loc[invalid_index, SPEECH_ID_COLUMN].tolist()
    )
    chunks = chunks.loc[
        ~chunks[SPEECH_ID_COLUMN].isin(invalid_ids)
    ].copy()

    structure = chunks.groupby(SPEECH_ID_COLUMN).agg(
        observed_rows=(CHUNK_INDEX_COLUMN, "size"),
        observed_indices=(CHUNK_INDEX_COLUMN, "nunique"),
        minimum_index=(CHUNK_INDEX_COLUMN, "min"),
        maximum_index=(CHUNK_INDEX_COLUMN, "max"),
        declared_total=(CHUNK_TOTAL_COLUMN, "first"),
    )
    complete = (
        structure["observed_rows"].eq(structure["declared_total"])
        & structure["observed_indices"].eq(structure["declared_total"])
        & structure["minimum_index"].eq(0)
        & structure["maximum_index"].eq(structure["declared_total"] - 1)
    )
    complete_ids = structure.index[complete]
    chunks = chunks.loc[
        chunks[SPEECH_ID_COLUMN].isin(complete_ids)
    ].copy()

    speech_scores = (
        chunks.groupby(SPEECH_ID_COLUMN)[score_column]
        .agg(
            chunk_mean="mean",
            chunk_sd=lambda values: float(values.std(ddof=0)),
        )
        .reset_index()
    )
    panel = full_scores.merge(
        speech_scores,
        on=SPEECH_ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    panel["score_change"] = panel["chunk_mean"] - panel[score_column]
    return panel


def build_paired_score_changes(full_data, chunk_data):
    """
    Reproduce the paired score-change sample used by ``pic_chunk_check.py``.

    Both LLM and K-means scores must be observed before and after chunking.
    Chunk means exclude missing values, matching ``merge_final.py``.
    """
    chunk_scores = chunk_data[
        [
            SPEECH_ID_COLUMN,
            PROCEDURAL_SCORE_COLUMN,
            CLUSTER_SCORE_COLUMN,
        ]
    ].copy()
    for column in [PROCEDURAL_SCORE_COLUMN, CLUSTER_SCORE_COLUMN]:
        chunk_scores[column] = pd.to_numeric(
            chunk_scores[column],
            errors="coerce",
        )

    aggregated = (
        chunk_scores.groupby(SPEECH_ID_COLUMN)
        .agg(
            chunk_procedural_score=(PROCEDURAL_SCORE_COLUMN, "mean"),
            chunk_cluster_score=(CLUSTER_SCORE_COLUMN, "mean"),
            chunk_row_count=(SPEECH_ID_COLUMN, "size"),
        )
        .reset_index()
    )

    full_scores = full_data[
        [
            SPEECH_ID_COLUMN,
            PROCEDURAL_SCORE_COLUMN,
            CLUSTER_SCORE_COLUMN,
        ]
    ].copy()
    full_scores = full_scores.drop_duplicates(
        SPEECH_ID_COLUMN,
        keep="first",
    )
    paired = full_scores.merge(
        aggregated,
        on=SPEECH_ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    numeric_columns = [
        PROCEDURAL_SCORE_COLUMN,
        CLUSTER_SCORE_COLUMN,
        "chunk_procedural_score",
        "chunk_cluster_score",
        "chunk_row_count",
    ]
    for column in numeric_columns:
        paired[column] = pd.to_numeric(paired[column], errors="coerce")
    paired = paired.dropna(subset=numeric_columns).copy()
    paired = paired.loc[
        paired[PROCEDURAL_SCORE_COLUMN].between(0, 1, inclusive="both")
        & paired[CLUSTER_SCORE_COLUMN].between(0, 1, inclusive="both")
        & paired["chunk_procedural_score"].between(
            0, 1, inclusive="both"
        )
        & paired["chunk_cluster_score"].between(
            0, 1, inclusive="both"
        )
        & paired["chunk_row_count"].ge(MIN_CHUNK_TOTAL)
    ].copy()

    paired["llm_score_change"] = (
        paired["chunk_procedural_score"]
        - paired[PROCEDURAL_SCORE_COLUMN]
    )
    paired["kmeans_score_change"] = (
        paired["chunk_cluster_score"]
        - paired[CLUSTER_SCORE_COLUMN]
    )
    return paired


def build_classification_panel(full_data, chunk_data):
    """Build speech-level category and cluster retention measures."""
    full = remove_error_rows(
        full_data[
            [
                SPEECH_ID_COLUMN,
                CATEGORY_COLUMN,
                CLUSTER_COLUMN,
                TOPIC_ERROR_COLUMN,
                CATEGORY_ERROR_COLUMN,
            ]
        ].copy()
    )
    chunks = remove_error_rows(
        chunk_data[
            [
                SPEECH_ID_COLUMN,
                CATEGORY_COLUMN,
                CLUSTER_COLUMN,
                CHUNK_TOTAL_COLUMN,
                TOPIC_ERROR_COLUMN,
                CATEGORY_ERROR_COLUMN,
            ]
        ].copy()
    )

    for data in [full, chunks]:
        data[CATEGORY_COLUMN] = (
            data[CATEGORY_COLUMN].fillna("").astype(str).str.strip()
        )
        data[CLUSTER_COLUMN] = pd.to_numeric(
            data[CLUSTER_COLUMN],
            errors="coerce",
        )
    chunks[CHUNK_TOTAL_COLUMN] = pd.to_numeric(
        chunks[CHUNK_TOTAL_COLUMN],
        errors="coerce",
    )

    full = full.dropna(subset=[CLUSTER_COLUMN])
    full = full.loc[
        full[CATEGORY_COLUMN].ne("") & full[CLUSTER_COLUMN].ge(0)
    ].copy()
    chunks = chunks.dropna(
        subset=[CLUSTER_COLUMN, CHUNK_TOTAL_COLUMN]
    )
    chunks = chunks.loc[
        chunks[CATEGORY_COLUMN].ne("")
        & chunks[CLUSTER_COLUMN].ge(0)
        & chunks[CHUNK_TOTAL_COLUMN].ge(MIN_CHUNK_TOTAL)
    ].copy()
    full[CLUSTER_COLUMN] = full[CLUSTER_COLUMN].astype(int)
    chunks[CLUSTER_COLUMN] = chunks[CLUSTER_COLUMN].astype(int)
    chunks[CHUNK_TOTAL_COLUMN] = chunks[CHUNK_TOTAL_COLUMN].astype(int)

    full = full.drop_duplicates(SPEECH_ID_COLUMN, keep="first")
    common_ids = set(full[SPEECH_ID_COLUMN]).intersection(
        chunks[SPEECH_ID_COLUMN]
    )
    full = full.loc[full[SPEECH_ID_COLUMN].isin(common_ids)].copy()
    chunks = chunks.loc[
        chunks[SPEECH_ID_COLUMN].isin(common_ids)
    ].copy()

    diagnostics = chunks.groupby(SPEECH_ID_COLUMN).agg(
        rows_found=(SPEECH_ID_COLUMN, "size"),
        chunk_total_min=(CHUNK_TOTAL_COLUMN, "min"),
        chunk_total_max=(CHUNK_TOTAL_COLUMN, "max"),
    )
    consistent = diagnostics["chunk_total_min"].eq(
        diagnostics["chunk_total_max"]
    )
    complete = diagnostics["rows_found"].eq(
        diagnostics["chunk_total_max"]
    )
    complete_ids = diagnostics.index[consistent & complete]
    full = full.loc[
        full[SPEECH_ID_COLUMN].isin(complete_ids)
    ].copy()
    chunks = chunks.loc[
        chunks[SPEECH_ID_COLUMN].isin(complete_ids)
    ].copy()

    lookup = full[
        [SPEECH_ID_COLUMN, CATEGORY_COLUMN, CLUSTER_COLUMN]
    ].rename(
        columns={
            CATEGORY_COLUMN: "original_category",
            CLUSTER_COLUMN: "original_cluster",
        }
    )
    transitions = chunks.merge(
        lookup,
        on=SPEECH_ID_COLUMN,
        how="inner",
        validate="many_to_one",
    )
    transitions["category_retained"] = transitions[CATEGORY_COLUMN].eq(
        transitions["original_category"]
    )
    transitions["cluster_retained"] = transitions[CLUSTER_COLUMN].eq(
        transitions["original_cluster"]
    )
    transitions["speech_weight"] = (
        1.0 / transitions[CHUNK_TOTAL_COLUMN]
    )

    speech_metrics = (
        transitions.groupby(SPEECH_ID_COLUMN)
        .agg(
            category_retention=("category_retained", "mean"),
            cluster_retention=("cluster_retained", "mean"),
        )
        .reset_index()
    )
    return speech_metrics, transitions


def build_headline_results():
    """Calculate the six estimates requested by the headline-results table."""
    full_data, chunk_data = read_headline_data()
    paired_scores = build_paired_score_changes(full_data, chunk_data)
    llm_panel = build_score_panel(
        full_data,
        chunk_data,
        PROCEDURAL_SCORE_COLUMN,
    )
    classification, transitions = build_classification_panel(
        full_data,
        chunk_data,
    )

    results = []

    for statistic, values in [
        (
            "Mean LLM score change after chunking",
            paired_scores["llm_score_change"],
        ),
        (
            "Mean K-means score change after chunking",
            paired_scores["kmeans_score_change"],
        ),
        (
            "Mean LLM category retention",
            classification["category_retention"],
        ),
        (
            "Mean K-means cluster retention",
            classification["cluster_retention"],
        ),
    ]:
        estimate, ci_low, ci_high, n = mean_ci_95(values)
        add_result(
            results,
            "Headline result",
            statistic,
            value=estimate,
            ci_95_low=ci_low,
            ci_95_high=ci_high,
            n=n,
        )

    off_diagonal = transitions.loc[
        transitions[CLUSTER_COLUMN].ne(transitions["original_cluster"])
    ].copy()
    total_off_diagonal_weight = off_diagonal["speech_weight"].sum()
    cluster_4_weight = off_diagonal.loc[
        off_diagonal[CLUSTER_COLUMN].eq(TARGET_INFLOW_CLUSTER),
        "speech_weight",
    ].sum()
    inflow_share = (
        cluster_4_weight / total_off_diagonal_weight
        if total_off_diagonal_weight > 0
        else np.nan
    )
    add_result(
        results,
        "Headline result",
        "K-means off-diagonal inflow to Cluster 4",
        value=inflow_share,
        n=int(off_diagonal[SPEECH_ID_COLUMN].nunique()),
    )

    zero_sd_changes = llm_panel.loc[
        llm_panel["chunk_sd"].abs().le(ZERO_SD_TOLERANCE),
        "score_change",
    ]
    estimate, ci_low, ci_high, n = mean_ci_95(zero_sd_changes)
    add_result(
        results,
        "Headline result",
        "LLM score change when chunk-score SD=0",
        value=estimate,
        ci_95_low=ci_low,
        ci_95_high=ci_high,
        n=n,
    )
    return results


def add_result(
    results,
    section,
    statistic,
    group="",
    value=None,
    share=None,
    ci_95_low=None,
    ci_95_high=None,
    n=None,
):
    """Append one consistently structured output row."""
    results.append(
        {
            "section": section,
            "statistic": statistic,
            "group": group,
            "value": value,
            "share": share,
            "ci_95_low": ci_95_low,
            "ci_95_high": ci_95_high,
            "n": n,
        }
    )


def build_descriptive_table():
    """Build all requested descriptive statistics."""
    original_row_count = count_rows(MERGED_FILE)
    cleaned_row_count = count_rows(CLEANED_FILE)
    (
        sample_row_count,
        word_count_percentiles,
        percentile_group_counts,
        category_counts,
    ) = read_sample_statistics()
    multi_chunk_speech_count, maximum_chunk_total = read_chunk_statistics()

    results = []
    add_result(
        results,
        "Dataset",
        "Original record count",
        value=original_row_count,
    )
    add_result(
        results,
        "Dataset",
        "Filtered sample count",
        value=cleaned_row_count,
    )
    add_result(
        results,
        "Dataset",
        "Stratified sample count",
        value=sample_row_count,
    )

    for percentile in PERCENTILES:
        add_result(
            results,
            "Text length percentile",
            "Speech text length (words)",
            group=f"p{percentile}",
            value=word_count_percentiles.loc[percentile / 100],
            share=percentile / 100,
        )

    ordered_groups = PERCENTILE_GROUPS + sorted(
        set(percentile_group_counts) - set(PERCENTILE_GROUPS)
    )
    for group in ordered_groups:
        if group not in percentile_group_counts:
            continue
        group_count = percentile_group_counts[group]
        add_result(
            results,
            "Percentile stratum",
            "Sample count",
            group=group,
            value=group_count,
            share=group_count / sample_row_count,
        )

    for category in sorted(category_counts):
        category_count = category_counts[category]
        add_result(
            results,
            "Category",
            "Sample count",
            group=category,
            value=category_count,
            share=category_count / sample_row_count,
        )

    add_result(
        results,
        "Chunk",
        "Speeches with at least two chunks",
        value=multi_chunk_speech_count,
    )
    add_result(
        results,
        "Chunk",
        "Maximum chunks per speech",
        value=maximum_chunk_total,
    )
    results.extend(build_headline_results())

    descriptive_table = pd.DataFrame(results)
    headline_rows = descriptive_table["section"].eq("Headline result")
    descriptive_table.loc[~headline_rows, "value"] = (
        descriptive_table.loc[~headline_rows, "value"].round(2)
    )
    descriptive_table.loc[headline_rows, "value"] = (
        descriptive_table.loc[headline_rows, "value"].round(6)
    )
    descriptive_table["share"] = descriptive_table["share"].round(6)
    descriptive_table["ci_95_low"] = (
        descriptive_table["ci_95_low"].round(6)
    )
    descriptive_table["ci_95_high"] = (
        descriptive_table["ci_95_high"].round(6)
    )
    return descriptive_table


def main():
    """Generate and save the descriptive-statistics CSV."""
    descriptive_table = build_descriptive_table()
    descriptive_table.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        float_format="%.15g",
    )

    print("Descriptive statistics completed")
    print(f"Rows written: {len(descriptive_table):,}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
