"""
Step 2: Induce topic taxonomies from the topic phrases produced in Step 1.

This version returns two taxonomies in one JSON file:

1. `categories`: a fixed taxonomy with exactly N categories, kept for
   compatibility with later scripts that expect topic_taxonomy.json to contain
   a top-level `categories` field.
2. `natural_n_categories` and `natural_categories`: the model's own estimate of
   how many categories the observed topic phrases naturally form when the number
   is not fixed.

Current behavior:
- The script does NOT use PROCEDURAL_SCORE_COLUMNS, procedural_score,
  procedural_rhetorical_score, or is_high_procedural to filter speeches.
- All valid topic phrases are used when inducing the taxonomy.
- The script does NOT run a separate JSON format validation step. The model's
  raw response is saved so that you can inspect it manually.

Place and run this script inside:
    Gemini_merged

Input:
    speeches_topics_procedural.csv

Output:
    topic_taxonomy.json
    topic_taxonomy.raw_response.txt

Usage:
    python 02_induce_taxonomy_gemini.py
    python 02_induce_taxonomy_gemini.py --top-n 2000
    python 02_induce_taxonomy_gemini.py --n-categories 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "speeches_topics_procedural.csv"
OUTPUT_PATH = ROOT / "topic_taxonomy.json"
RAW_RESPONSE_PATH = ROOT / "topic_taxonomy.raw_response.txt"

MODEL = "gemini-2.5-pro"
TEMPERATURE = 0.0
TOP_N = 2000
N_CATEGORIES = 10
MIN_NATURAL_CATEGORIES = 3
MAX_NATURAL_CATEGORIES = 25

SYSTEM_INSTRUCTION_TEMPLATE = """\
You are a research assistant building a semantic taxonomy for speeches in the Canadian House of Commons, 1953-1993. You will be shown a list of up to {top_n} short topic descriptions, each with a frequency count.

Your task has two parts.

Part A: propose EXACTLY {n_categories} broad semantic categories that cover the main substantive variation in the observed topic phrases. Return these under `categories`.

Part B: propose a second taxonomy in which the number of categories is not fixed in advance. Choose the empirically most natural number of categories based on the observed topic phrases. Return this number as `natural_n_categories` and the categories under `natural_categories`.

The categories in both taxonomies should be mutually exclusive, collectively exhaustive, empirically grounded, stable across the 1963-1993 period, and useful for later coding.

Avoid categories that are too narrow, too event-specific, or based on a single proper noun. Prefer broader categories that capture recurring policy areas, institutional issues, and substantive forms of parliamentary discussion.

For each category, provide a short name and a one-sentence definition. Where categories may overlap, include brief inclusion/exclusion guidance in the definition.

Return JSON in this exact shape:
{{
"categories": [
{{"name": "<category name>", "definition": "<one-sentence definition>"}}
],
"natural_n_categories": 0,
"natural_categories": [
{{"name": "<category name>", "definition": "<one-sentence definition>"}}
],
"natural_rationale": "<brief explanation of why this number of natural categories was chosen>"
}}

"""

def make_response_schema(n_categories: int) -> dict[str, Any]:
    """Build the response schema dynamically so --n-categories can change."""
    category_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "definition": {"type": "string"},
        },
        "required": ["name", "definition"],
    }
    return {
        "type": "object",
        "properties": {
            "categories": {
                "type": "array",
                "minItems": n_categories,
                "maxItems": n_categories,
                "items": category_schema,
            },
            "natural_n_categories": {
                "type": "integer",
                "minimum": MIN_NATURAL_CATEGORIES,
                "maximum": MAX_NATURAL_CATEGORIES,
            },
            "natural_categories": {
                "type": "array",
                "minItems": MIN_NATURAL_CATEGORIES,
                "maxItems": MAX_NATURAL_CATEGORIES,
                "items": category_schema,
            },
            "natural_rationale": {"type": "string"},
        },
        "required": [
            "categories",
            "natural_n_categories",
            "natural_categories",
            "natural_rationale",
        ],
    }


# ---------- environment ----------
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


# ---------- data ----------
def top_topics(path: Path, n: int) -> tuple[pd.Series, dict[str, Any]]:
    """Return top topic phrases without procedural-score filtering."""
    if not path.exists():
        sys.exit(f"Input file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    if "topic" not in df.columns:
        sys.exit(f"Required column 'topic' not found. Available columns: {list(df.columns)}")

    before = len(df)
    df = df[df["topic"].notna() & (df["topic"].astype(str).str.strip() != "")].copy()
    after = len(df)
    if before != after:
        print(f"Dropped {before - after:,} rows with missing/empty topic")

    counts = df["topic"].astype(str).str.strip().value_counts()
    if counts.empty:
        sys.exit("No usable topic phrases found.")

    k = min(n, len(counts))
    coverage = counts.head(k).sum() / len(df)
    print("Procedural-score filtering is disabled; using all valid topic phrases.")
    print(f"{len(df):,} speeches; {len(counts):,} unique topics")
    print(f"Top {k}: covers {counts.head(k).sum():,} speeches ({100 * coverage:.1f}%)")

    metadata = {
        "input_path": str(path),
        "n_speeches_with_topic": int(len(df)),
        "n_unique_topics": int(len(counts)),
        "top_n_requested": int(n),
        "top_n_used": int(k),
        "top_n_coverage": round(float(coverage), 6),
        "procedural_filtering_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return counts.head(k), metadata


def build_prompt(counts: pd.Series) -> str:
    lines = [f"{int(count)}\t{topic}" for topic, count in counts.items()]
    return "Topic strings with frequency counts (count\\ttopic):\n\n" + "\n".join(lines)


# ---------- minimal JSON parsing ----------
def strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> str:
    """Extract a JSON object from a response that may contain accidental wrapper text."""
    text = strip_json_fence(text)
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return text[start : end + 1]


# ---------- main ----------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--n-categories", type=int, default=N_CATEGORIES)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--raw-output", type=Path, default=RAW_RESPONSE_PATH)

    # Deprecated compatibility arguments. They are accepted but intentionally ignored.
    parser.add_argument(
        "--procedural-threshold",
        type=float,
        default=None,
        help="Deprecated; accepted for compatibility but ignored. This script does not filter by procedural score.",
    )
    parser.add_argument(
        "--include-high-procedural",
        action="store_true",
        help="Deprecated; accepted for compatibility but ignored. This script always uses all valid topics.",
    )
    args = parser.parse_args()

    if args.procedural_threshold is not None or args.include_high_procedural:
        print("Note: procedural filtering arguments are ignored in this version.")

    load_env_from_likely_locations()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set.")

    counts, metadata = top_topics(args.input, args.top_n)
    prompt = build_prompt(counts)

    system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(n_categories=args.n_categories, top_n=args.top_n)
    response_schema = make_response_schema(args.n_categories)

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        response_schema=response_schema,
    )

    print(f"Calling {args.model} with {len(counts):,} topics in prompt...")
    resp = client.models.generate_content(model=args.model, contents=prompt, config=config)
    raw_text = resp.text or ""

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(raw_text, encoding="utf-8")
    print(f"Saved raw model response to {args.raw_output}")

    # No separate JSON validation is run here. We only parse the JSON so it can be
    # saved as a formatted file and used by the next script.
    try:
        taxonomy = json.loads(extract_json_object(raw_text))
    except Exception as exc:
        sys.exit(f"Could not parse model response as JSON. See {args.raw_output}. Error: {exc}")

    # Keep `categories` at the top level for compatibility with the next script.
    taxonomy["metadata"] = metadata | {
        "model": args.model,
        "fixed_n_categories_requested": int(args.n_categories),
        "raw_response_path": str(args.raw_output),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output}")

    print("\nFixed taxonomy:")
    for i, category in enumerate(taxonomy.get("categories", []), 1):
        print(f"{i:2d}. {category.get('name', '')}")
        print(f"    {category.get('definition', '')}")

    print(f"\nNatural taxonomy: {taxonomy.get('natural_n_categories', 'unknown')} categories")
    for i, category in enumerate(taxonomy.get("natural_categories", []), 1):
        print(f"{i:2d}. {category.get('name', '')}")
        print(f"    {category.get('definition', '')}")


if __name__ == "__main__":
    main()





