# Execution Order
## Data Processing
1. `src\speech_merge.py`
   - input：`input\lipad\**\*.csv`，`input\Parliamentarians.xlsx`
   - output：`output\Speech_merged.csv`
  
2. `src\speech_clean.py`
   - input：`output\Speech_merged.csv`
   - output：`output\Speech_cleaned.csv`

3. `src\sample.py`
   - input: `output\Speech_cleaned.csv`
   - output: `output\Speech_sample.csv`

4. `src\speech_stem.py`
   - input：`output\Speech_sample.csv`
   - output：`output\Speech_sample.csv`

5. `src\speech_chunk.py`
   - input：`output\Speech_sample.csv`
   - output：`output\Speech_chunk.csv`

6. `src\speech_stem_chunk.py`
   - input：`output\Speech_chunk.csv`
   - output：`output\Speech_chunk.csv`

## Kmeans
### Kmeans_merged
1. `Kmeans\Kmeans_merged\run_kmeans.py`
   - input: `output\Speech_sample.csv`
   - output: `output\Speech_sample.csv`
   - output: `Kmeans\Kmeans_merged\cluster_summary.csv`

### Kmeans_chunk
1. `Kmeans\Kmeans_chunk\run_kmeans.py`
   - input: `output\Speech_chunk.csv`
   - 
   - output: `output\Speech_chunk.csv`
   - output: `Kmeans\Kmeans_chunk\cluster_summary.csv`

## Gemini_merged
   - input: `output\Speech_sample.csv`
   - input: `Kmeans\Kmeans_merged\cluster_summary.csv`
   - output: `Gemini_merged\cluster_procedural_scores_gemini.csv`
   - output: `Gemini_merged\speeches_topics_procedural_categories.csv`
1. `Gemini_merged\06_merge_gemini_results_into_speech_sample.py`
   - input: `output\Speech_sample.csv`
   - input: `Gemini_merged\cluster_procedural_scores_gemini.csv`
   - input: `Gemini_merged\speeches_topics_procedural_categories.csv`
   - output: `output\Speech_sample.csv`

## Gemini_chunk
   - input: `output\Speech_chunk.csv`
   - input: `Kmeans\Kmeans_chunk\cluster_summary.csv`
   - output: `Gemini_merged\cluster_procedural_scores_gemini.csv`
   - output: `Gemini_merged\speeches_topics_procedural_categories.csv`

1. `Gemini_chunk\06_merge_gemini_results_into_speech_chunk.py`
   - input: `output\Speech_chunk.csv`
   - input: `Gemini_chunk\cluster_procedural_scores_gemini.csv`
   - input: `Gemini_chunk\speeches_topics_procedural_categories.csv`
   - output: `output\Speech_chunk.csv`



## Pic
1. `src\merge_final.py`
1. `src\pic_merged.py`
   - input: `output/Speech_sample.csv`
   - output: `output/pic/merged/`

2. `src\pic_chunk_check.py`
3. `src\merge_final.py`
4. `src\pic_main.py`
5. `src\pic_classification.py`
6. `src\pic_LLM.py`
7. `src\pic_position.py`

## Descriptive
`src\speech_descriptive.py`
