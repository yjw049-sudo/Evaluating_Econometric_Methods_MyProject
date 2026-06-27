"""
Step 2: Induce a topic taxonomy from the topic phrases produced in Step 1.

This remains a normal one-shot Gemini call because it is only one request, not a large batch.

Place and run this script inside:
    work_1953_1993/Gemini

Input:
    speeches_topics_procedural.csv

Output:
    topic_taxonomy.json

Usage:
    python 02_induce_taxonomy_gemini.py
    python 02_induce_taxonomy_gemini.py --include-high-procedural
    python 02_induce_taxonomy_gemini.py --procedural-threshold 0.75
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "speeches_topics_procedural.csv"
OUTPUT_PATH = ROOT / "topic_taxonomy.json"

MODEL = "gemini-2.5-pro"
TEMPERATURE = 0.0
TOP_N = 2000
N_CATEGORIES = 10
PROCEDURAL_SCORE_COLUMNS = ["procedural_score", "procedural_rhetorical_score"]

SYSTEM_INSTRUCTION = f"""\
You are a research assistant building a topical taxonomy for speeches in the Canadian House of Commons, 1953-1993.
You will be shown a list of {TOP_N} short topic descriptions, each with a frequency count.

Your task is to propose EXACTLY {N_CATEGORIES} categories that cover the substantive variation in these topics.

The categories must be:
- mutually exclusive: every topic should fit clearly into one category;
- collectively exhaustive: include a residual category if necessary;
- empirically grounded: derive the categories from the observed topic phrases, not from a generic political-science taxonomy;
- stable across the 1953-1993 period.

Important:
- Procedural or rhetorical speeches may appear as topics, but do not let them dominate the substantive taxonomy unless they are empirically frequent enough to require a category.
- If you include a procedural category, define clearly what belongs there and what should remain in substantive policy categories.

For each category, write a one-sentence definition that another coder could apply consistently.
Include inclusion/exclusion guidance where adjacent categories might otherwise be confused.

Return JSON in this exact shape:
{{
  "categories": [
    {{"name": "<short category name>", "definition": "<one sentence>"}}
  ]
}}
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "categories": {
            "type": "array",
            "minItems": N_CATEGORIES,
            "maxItems": N_CATEGORIES,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "definition": {"type": "string"},
                },
                "required": ["name", "definition"],
            },
        },
    },
    "required": ["categories"],
}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load_env_from_likely_locations() -> None:
    for folder in [ROOT, *ROOT.parents]:
        load_env(folder / ".env")


def detect_score_column(df: pd.DataFrame) -> str | None:
    for col in PROCEDURAL_SCORE_COLUMNS:
        if col in df.columns:
            return col
    return None


def top_topics(path: Path, n: int, exclude_high_procedural: bool, procedural_threshold: float) -> pd.Series:
    if not path.exists():
        sys.exit(f"Input file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["topic"].notna() & (df["topic"].astype(str).str.strip() != "")].copy()

    if exclude_high_procedural:
        score_col = detect_score_column(df)
        if score_col:
            score = pd.to_numeric(df[score_col], errors="coerce")
            before = len(df)
            df = df[~(score >= procedural_threshold)].copy()
            print(f"Excluded {before - len(df):,} speeches with {score_col} >= {procedural_threshold}")
        elif "is_high_procedural" in df.columns:
            flag = df["is_high_procedural"].astype(str).str.lower().isin(["true", "1", "yes"])
            before = len(df)
            df = df[~flag].copy()
            print(f"Excluded {before - len(df):,} high-procedural speeches before taxonomy induction")
        else:
            print("No procedural score/flag column found; taxonomy uses all topics.")

    counts = df["topic"].astype(str).value_counts()
    if counts.empty:
        sys.exit("No usable topic phrases found.")

    k = min(n, len(counts))
    print(f"{len(df):,} speeches; {len(counts):,} unique topics")
    print(f"Top {k}: covers {counts.head(k).sum():,} speeches "
          f"({100 * counts.head(k).sum() / len(df):.1f}%)")
    return counts.head(k)


def build_prompt(counts: pd.Series) -> str:
    lines = [f"{count}\t{topic}" for topic, count in counts.items()]
    return "Topic strings with frequency counts (count\\ttopic):\n\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--n-categories", type=int, default=N_CATEGORIES)
    parser.add_argument("--procedural-threshold", type=float, default=0.75)
    parser.add_argument(
        "--include-high-procedural",
        action="store_true",
        help="include high-procedural/high-rhetorical speeches when inducing the taxonomy",
    )
    args = parser.parse_args()

    if args.n_categories != N_CATEGORIES:
        sys.exit("This script currently has a fixed response schema for 10 categories. Change N_CATEGORIES in the file if needed.")

    load_env_from_likely_locations()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set.")

    counts = top_topics(
        args.input,
        args.top_n,
        exclude_high_procedural=not args.include_high_procedural,
        procedural_threshold=args.procedural_threshold,
    )
    prompt = build_prompt(counts)

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    print(f"Calling {MODEL} with {len(counts):,} topics in prompt...")
    resp = client.models.generate_content(model=MODEL, contents=prompt, config=config)
    taxonomy = json.loads(resp.text)

    categories = taxonomy.get("categories", [])
    if len(categories) != N_CATEGORIES:
        sys.exit(f"Expected {N_CATEGORIES} categories, got {len(categories)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output}")
    for i, category in enumerate(categories, 1):
        print(f"{i:2d}. {category['name']}")
        print(f"    {category['definition']}")


if __name__ == "__main__":
    main()
