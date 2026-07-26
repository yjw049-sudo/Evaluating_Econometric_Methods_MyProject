Gemini_merged run order
=======================

1. Create speech request chunks locally:

    python 01_batch_classify_topics_gemini.py make-chunks --chunk-size 1000

2. Submit and process speech chunks sequentially:

    python 01_batch_classify_topics_gemini.py run-chunks

3. Induce the taxonomy from all non-empty topics:

    python 02_induce_taxonomy_gemini.py

4. Assign every unique topic to a taxonomy category:

    python 03_batch_assign_categories_gemini.py make-jsonl
    python 03_batch_assign_categories_gemini.py submit
    python 03_batch_assign_categories_gemini.py wait
    python 03_batch_assign_categories_gemini.py download
    python 03_batch_assign_categories_gemini.py parse

   Step 3 is resumable: existing successful categories are preserved and only
   missing or failed topics are included in a newly generated request file.

5. Merge topic categories back to speeches:

    python 04_merge_gemini_outputs.py

6. Merge the final Gemini fields into output/Speech_sample.csv:

    python 06_merge_gemini_results_into_speech_sample.py

   The script creates output/Speech_sample_before_gemini_merge.csv once, then
   adds topic, category, procedural_score, topic_error, and category_error.

7. Optionally score K-means clusters:

    python 05_score_kmeans_clusters_gemini.py

Inputs:
    ../output/Speech_sample.csv
    ../output/Kmeans/cluster_summary.csv  (Step 7 only)

Main outputs:
    speeches_topics_procedural.csv
    topic_taxonomy.json
    topic_to_category.csv
    speeches_topics_procedural_categories.csv
    ../output/Speech_sample.csv

