Gemini Batch workflow for Gemini_chunk
======================================

Data structure
--------------

A speech can have multiple chunk rows. The stable row identifier is the composite
key (basepk, chunk_index). Do not deduplicate or merge on basepk alone.

External input:
    ../output/Speech_chunk.csv

Required columns:
    basepk
    chunk_index
    speechtext_chunk

Prepare the local two-key input without filtering:

    python create_speeches_gemini_s0.py

Step 1: chunk-level topics and procedural scores
------------------------------------------------

Create local request files with the conservative default size:

    python 01_batch_classify_topics_gemini.py make-chunks --chunk-size 1000 --overwrite

Run sequentially:

    python 01_batch_classify_topics_gemini.py run-chunks

run-chunks waits for each job before submitting the next one, retries temporary
quota errors, resumes from saved files, and merges part CSVs at the end.
Only rows with a non-empty topic, a valid procedural_score, and an empty error are
treated as complete; failed chunk rows remain eligible for retry.

Step 2: taxonomy
----------------

    python 02_induce_taxonomy_gemini.py

The script uses all non-empty topic phrases and does not filter by
procedural_score. Literal JSON braces in the prompt are escaped so template
formatting does not fail before the API call.

Step 3: topic category assignment
---------------------------------

    python 03_batch_assign_categories_gemini.py make-jsonl
    python 03_batch_assign_categories_gemini.py submit
    python 03_batch_assign_categories_gemini.py wait
    python 03_batch_assign_categories_gemini.py download
    python 03_batch_assign_categories_gemini.py parse

Requests contain stable topic_id values. If Gemini omits an item, returned items
are retained and only the missing topic is marked as an error. Parsing merges
new results into topic_to_category.csv and preserves prior successful rows.
make-jsonl retries only topics whose category is empty or whose error is nonempty.

Step 4 and final write-back
---------------------------

    python 04_merge_gemini_outputs.py
    python 06_merge_gemini_results_into_speech_chunk.py

Step 6 verifies that the final result is newer than its dependencies, joins on
(basepk, chunk_index), preserves the row count, creates a one-time backup, and
atomically replaces output/Speech_chunk.csv.

Quota and safety
----------------

The 3M queued-token quota is shared by unfinished Batch jobs for the same model.
Use 1000 rows per Step 1 job and submit jobs sequentially. Step 3 keeps 50 topics
per request because topic_id matching can salvage partial responses while the
larger group avoids repeating the taxonomy prompt too many times. Do not use
--force unless a duplicate paid job is intentional.

