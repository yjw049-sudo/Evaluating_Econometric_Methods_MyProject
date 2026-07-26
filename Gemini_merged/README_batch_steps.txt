Gemini Batch workflow for Gemini_merged
=======================================

External speech input
---------------------

    ../output/Speech_sample.csv

Required columns:

    basepk
    speechtext

Step 1: speech topics and procedural scores
-------------------------------------------

Create local request chunks (no API call):

    python 01_batch_classify_topics_gemini.py make-chunks --chunk-size 1000

Run chunks sequentially:

    python 01_batch_classify_topics_gemini.py run-chunks

The command resumes from existing job/result/part files, waits for each job before
submitting the next one, retries temporary quota failures, and merges parsed parts
into speeches_topics_procedural.csv.

Step 2: induce the taxonomy
---------------------------

    python 02_induce_taxonomy_gemini.py

This version uses all non-empty topic phrases; it does not filter by
procedural_score. It creates topic_taxonomy.json and saves the raw response.

Step 3: assign topics to categories
-----------------------------------

Create requests locally:

    python 03_batch_assign_categories_gemini.py make-jsonl

Then submit, wait, download, and parse:

    python 03_batch_assign_categories_gemini.py submit
    python 03_batch_assign_categories_gemini.py wait
    python 03_batch_assign_categories_gemini.py download
    python 03_batch_assign_categories_gemini.py parse

Each request contains stable topic_id values. If Gemini omits an item, the parser
keeps every returned item and marks only the missing topic as an error. Parsing
merges new results into topic_to_category.csv and never replaces an existing
successful category with a failed retry.

make-jsonl treats only rows with a non-empty category and an empty error as done.
Therefore, after archiving completed Batch job/request/result files, the same
workflow can be repeated to retry only remaining failures.

Step 4: merge speech and category outputs
-----------------------------------------

    python 04_merge_gemini_outputs.py

Final output:

    speeches_topics_procedural_categories.csv

Step 5: optional K-means cluster scoring
----------------------------------------

Default external input:

    ../output/Kmeans/cluster_summary.csv

Run:

    python 05_score_kmeans_clusters_gemini.py

Quota and safety notes
----------------------

The 3M queued-token quota is shared by unfinished Batch jobs for the same model.
Submit jobs sequentially and wait when the console is close to the limit. submit
is not idempotent; do not use --force unless a duplicate paid job is intentional.
GEMINI_API_KEY is loaded from .env in Gemini_merged or a parent directory.
