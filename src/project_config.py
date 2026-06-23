from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_DIR / "input"
LIPAD_DIR = INPUT_DIR / "lipad"
LECTURE_DIR = PROJECT_DIR / "lecture"
OUTPUT_DIR = PROJECT_DIR / "output"
FINAL_DIR = PROJECT_DIR / "Final"

PROCESSED_DATA_DIR = OUTPUT_DIR / "processed_data"
OUTPUT_1920_1949_DIR = OUTPUT_DIR / "1920-1949-v1"
FULL_CORPUS_OUTPUT_DIR = OUTPUT_DIR / "1920-1949-full-corpus-v1"

CSV_SEPARATOR = ";"

BASEPK_COLUMN = "basepk"
YEAR_COLUMN = "year"
TEXT_COLUMN = "speechtext"
ORIGINAL_TEXT_COLUMN = "speechtext_oringinal"
AI_LABEL_COLUMN = "category"
CLUSTER_COLUMN = "kmeans_cluster"
