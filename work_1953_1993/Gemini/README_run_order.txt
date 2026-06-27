Put these scripts in:
    work_1953_1993\Gemini

Make sure the input file is in the same folder:
    work_1953_1993\Gemini\speeches_Gemin_s0.csv

Required input columns:
    basepk
    speechtext_oringinal

Recommended run order:

1. Smoke test topic + procedural labelling:
    python 01_classify_topics_gemini.py --limit 20

2. Full topic + procedural labelling:
    python 01_classify_topics_gemini.py

   Output:
    speeches_topics_procedural.csv

3. Induce a 10-category taxonomy from the generated topic phrases:
    python 02_induce_taxonomy_gemini.py

   Output:
    topic_taxonomy.json

   By default, this excludes high-procedural speeches when inducing the substantive taxonomy.
   If you want procedural topics to participate in taxonomy induction, run:
    python 02_induce_taxonomy_gemini.py --include-high-procedural

4. Assign each unique topic phrase to one taxonomy category:
    python 03_assign_categories_gemini.py

   Output:
    topic_to_category.csv

5. Merge speech-level topic/procedural labels with topic-level category labels:
    python 04_merge_gemini_outputs.py

   Output:
    speeches_topics_procedural_categories.csv

Notes:
- The scripts look for GEMINI_API_KEY in .env in the Gemini folder or parent folders.
- All default inputs and outputs are inside work_1953_1993\Gemini because ROOT is set to the script's own folder.
- Step 1 is resumable: if speeches_topics_procedural.csv already exists, already-written basepks are skipped.
