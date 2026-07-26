# Same Speeches, Different Measurements: LLM and K-Means Responses to Text Length and Chunking

This project examines whether text length affects how LLM and K-means measure the procedurality of parliamentary speeches. Using Canadian House of Commons speeches from 1963–1993, it compares results from complete speeches with results from shorter text chunks.

## Research Questions

The project asks:

1. Do LLM and K-means scores change with full-speech length?
2. What happens when the same speeches are divided into shorter chunks?
3. Why do the two methods respond differently?

## Main Findings

- LLM scores remain relatively stable across complete speeches of different lengths.
- K-means scores vary more clearly with full-speech length.
- After chunking, the mean LLM score decreases, while the K-means score increases.
- K-means shows more cluster reassignment after chunking, especially toward a broad destination cluster.
- The full-speech LLM judgment cannot be reproduced by simply averaging the scores of individual chunks.

These results suggest that LLM and K-means are sensitive to text length and input construction in different ways.

## Repository Structure

```text
.
├── input/
│   └── Parliamentarians.xlsx   # Parliamentarian metadata used during data merging
├── src/                        # Data processing, sampling, stemming, merging, and analysis scripts
├── Gemini_merged/              # Gemini workflow and saved results for full speeches
├── Gemini_chunk/               # Gemini workflow and saved results for chunked speeches
├── output/                     # Processed datasets, descriptive statistics, figures, and tables
├── requirements.txt            # Python dependencies
├── runall_order.md             # Detailed pipeline execution order
├── README.md                   # Project overview and workflow documentation
└── .gitignore                  # Excluded local source data and generated outputs
```

The numbered scripts in `Gemini_merged/` and `Gemini_chunk/` cover topic
classification, taxonomy induction, category assignment, output merging, and
K-means cluster scoring. Their `batch_parts/` and related subdirectories contain
batch request, status, result, and manifest files retained for reproducibility.

Large LIPAD source files, K-means working files, and other local materials are
excluded through `.gitignore`; the retained files in `output/` provide the main
processed datasets and analysis results.

## Data

The project uses Canadian House of Commons speeches for 1963–1993. 
`input\lipad`
`input\Parliamentarians.xlsx`

## Analysis Workflow

### Data Processing

1. Merge and clean the speech data.
2. Construct the analysis sample.
3. Divide the sampled speech texts into chunks.

### K-means Pipeline

1. Stem the full-speech texts in the analysis sample.
2. Stem the chunked speech texts.
3. Cluster the stemmed full-speech texts and chunk-level texts separately.

### LLM Pipeline

1. Use Gemini to classify and score both the full-speech texts and the chunked texts.
2. Use Gemini to assign procedurality scores to the K-means clusters.

### Descriptive Statistics

1. Generate descriptive statistics.

### Figures

1. Produce full-speech length comparisons.
2. Merge and aggregate chunk-level results.
3. Analyze score changes after chunking.
4. Analyze category and cluster stability.
5. Analyze LLM positional sensitivity.

## Execution Order
Run `runall_order.md`

