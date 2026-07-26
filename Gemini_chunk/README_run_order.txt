Gemini_chunk run order
======================

0. Prepare chunk-level input without row filtering:

    python create_speeches_gemini_s0.py

External input:
    ../output/Speech_chunk.csv

Required columns:
    basepk
    chunk_index
    speechtext_chunk

Prepared input:
    speeches_Gemin_s0.csv

1. Create speech-chunk request files locally:

    python 01_batch_classify_topics_gemini.py make-chunks --chunk-size 1000 --overwrite

   Use --overwrite because request files created with the previous chunk size may
   already exist. This command does not call Gemini.

2. Submit and process chunks sequentially:

    python 01_batch_classify_topics_gemini.py run-chunks

3. Induce the taxonomy from all non-empty chunk topics:

    python 02_induce_taxonomy_gemini.py

4. Assign every unique topic to a taxonomy category:

    python 03_batch_assign_categories_gemini.py make-jsonl
    python 03_batch_assign_categories_gemini.py submit
    python 03_batch_assign_categories_gemini.py wait
    python 03_batch_assign_categories_gemini.py download
    python 03_batch_assign_categories_gemini.py parse

   Step 3 uses stable topic_id values. Existing successful categories are
   preserved, and a new make-jsonl run includes only missing or failed topics.

5. Merge topic categories back to chunk-level results:

    python 04_merge_gemini_outputs.py

6. Merge final fields into output/Speech_chunk.csv:

    python 06_merge_gemini_results_into_speech_chunk.py

   The script joins on (basepk, chunk_index), creates
   output/Speech_chunk_before_gemini_merge.csv once, then adds topic, category,
   procedural_score, topic_error, and category_error.

7. Optionally score K-means clusters:

    python 05_score_kmeans_clusters_gemini.py

Step 7 default input:
    ../Kmeans/Kmeans_chunk/cluster_summary.csv

Main outputs:
    speeches_topics_procedural.csv
    topic_taxonomy.json
    topic_to_category.csv
    speeches_topics_procedural_categories.csv
    ../output/Speech_chunk.csv
