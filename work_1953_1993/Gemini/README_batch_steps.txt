Gemini Batch workflow for work_1953_1993/Gemini
================================================

Put these files in work_1953_1993/Gemini:

1. 01_batch_classify_topics_gemini.py
2. 02_induce_taxonomy_gemini.py
3. 03_batch_assign_categories_gemini.py

Make sure the input file is also in the same folder:

    speeches_Gemin_s0.csv

The input CSV must contain:

    basepk
    speechtext_oringinal

The scripts look for GEMINI_API_KEY in .env or system environment.

Suggested test workflow
-----------------------

Step 1: create topic/procedural_score requests for 20 speeches

    python 01_batch_classify_topics_gemini.py make-jsonl --limit 20

Step 2: submit the first batch job

    python 01_batch_classify_topics_gemini.py submit

Step 3: check or wait

    python 01_batch_classify_topics_gemini.py status
    python 01_batch_classify_topics_gemini.py wait

Step 4: download and parse

    python 01_batch_classify_topics_gemini.py download
    python 01_batch_classify_topics_gemini.py parse

This creates:

    speeches_topics_procedural.csv

with columns:

    basepk, topic, procedural_score, error

If the 20-row test looks good, delete or move these files before making a full batch:

    batch_requests_topics_procedural.jsonl
    batch_job_topics_procedural.json
    batch_results_topics_procedural.jsonl
    speeches_topics_procedural.csv

Then run the full Step 1 batch:

    python 01_batch_classify_topics_gemini.py make-jsonl
    python 01_batch_classify_topics_gemini.py submit
    python 01_batch_classify_topics_gemini.py wait
    python 01_batch_classify_topics_gemini.py download
    python 01_batch_classify_topics_gemini.py parse

Important: submit is not idempotent. Do not run submit twice unless you intentionally want to create another paid batch job. The scripts protect you by refusing to submit if the job json already exists. Only use --force when you know you want another job.

After Step 1 is complete
------------------------

Induce taxonomy. This remains a normal one-shot API call because it is only one request:

    python 02_induce_taxonomy_gemini.py

By default it excludes speeches with procedural_score >= 0.75 before inducing the taxonomy. To include them:

    python 02_induce_taxonomy_gemini.py --include-high-procedural

To change the threshold:

    python 02_induce_taxonomy_gemini.py --procedural-threshold 0.8

This creates:

    topic_taxonomy.json

Step 3: assign unique topic phrases to taxonomy categories with Batch API
------------------------------------------------------------------------

Test on the first 200 unique topics:

    python 03_batch_assign_categories_gemini.py make-jsonl --limit 200
    python 03_batch_assign_categories_gemini.py submit
    python 03_batch_assign_categories_gemini.py wait
    python 03_batch_assign_categories_gemini.py download
    python 03_batch_assign_categories_gemini.py parse

This creates:

    topic_to_category.csv

For the full run, delete or move the category batch job/result files and topic_to_category.csv, then run:

    python 03_batch_assign_categories_gemini.py make-jsonl
    python 03_batch_assign_categories_gemini.py submit
    python 03_batch_assign_categories_gemini.py wait
    python 03_batch_assign_categories_gemini.py download
    python 03_batch_assign_categories_gemini.py parse

Main generated files
--------------------

Step 1 batch files:

    batch_requests_topics_procedural.jsonl
    batch_job_topics_procedural.json
    batch_status_topics_procedural.json
    batch_results_topics_procedural.jsonl
    speeches_topics_procedural.csv

Step 2 taxonomy files:

    topic_taxonomy.json

Step 3 category batch files:

    batch_requests_topic_categories.jsonl
    batch_job_topic_categories.json
    batch_status_topic_categories.json
    batch_results_topic_categories.jsonl
    topic_to_category.csv
