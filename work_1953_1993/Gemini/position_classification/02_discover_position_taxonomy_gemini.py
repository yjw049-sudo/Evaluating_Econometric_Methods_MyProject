"""
Ask Gemini to propose a small institutional taxonomy for speaker positions.

Place and run this script inside:
    work_1953_1993/Gemini/position_classification

Default input:
    speakerposition_unique_values.csv

Default outputs:
    position_taxonomy_gemini.json
    position_taxonomy_gemini.raw_response.txt

Usage:
    python 02_discover_position_taxonomy_gemini.py
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
DEFAULT_INPUT_PATH = ROOT / "speakerposition_unique_values.csv"
DEFAULT_JSON_OUTPUT_PATH = ROOT / "position_taxonomy_gemini.json"
DEFAULT_RAW_OUTPUT_PATH = ROOT / "position_taxonomy_gemini.raw_response.txt"

MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.0

MIN_CATEGORIES = 3
MAX_CATEGORIES = 8


SYSTEM_INSTRUCTION = """\
You are assisting with a coding scheme for institutional roles in Canadian parliamentary speech data.

You will receive a list of unique speaker position strings. Each position has a stable numeric position_id. The id is only an identifier; do not treat it as meaningful.

Task:
Propose a concise taxonomy of institutional position types for these position strings.

Coding objective:
The taxonomy should group titles by the speaker's formal institutional role in parliamentary debate, especially features such as floor access, agenda-management responsibility, presiding responsibility, and whether the title indicates a government, opposition, support, or ordinary-member role.

Neutrality requirements:
- Do not use speech outcomes, procedural_score, cluster_score, disagreement, topics, party ideology, or any dependent variable to define categories.
- Do not classify by party name, province, electoral district, policy field, or ministerial department.
- Do not create categories such as Liberal, Conservative, Finance, Defence, Agriculture, Transport, Justice, or Foreign Affairs.
- Do not assume that one type necessarily speaks more, speaks longer, or uses more procedural language. Only define institutional-role categories.

Practical requirements:
- Propose between 3 and 8 categories.
- Each category must have a stable snake_case category_id.
- Categories should be broad enough for later statistical analysis.
- Include inclusion rules and exclusion rules.
- Include a priority rule for composite positions containing multiple titles.
- The priority rule should be based on formal institutional role, not on expected empirical results.

Return valid JSON in this exact shape:
{
  "taxonomy": [
    {
      "category_id": "example_category_id",
      "category_label": "Example category label",
      "definition": "Brief neutral definition.",
      "inclusion_rule": "How to recognize positions in this category.",
      "exclusion_rule": "What similar titles should not be placed here.",
      "institutional_role_basis": "Why these titles belong together as institutional roles.",
      "examples": ["Example position 1", "Example position 2"]
    }
  ],
  "priority_rule": ["category_id_1", "category_id_2"],
  "brief_rationale": "Short explanation of why this taxonomy is suitable."
}
"""


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "taxonomy": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category_id": {"type": "string"},
                    "category_label": {"type": "string"},
                    "definition": {"type": "string"},
                    "inclusion_rule": {"type": "string"},
                    "exclusion_rule": {"type": "string"},
                    "institutional_role_basis": {"type": "string"},
                    "examples": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "category_id",
                    "category_label",
                    "definition",
                    "inclusion_rule",
                    "exclusion_rule",
                    "institutional_role_basis",
                    "examples",
                ],
            },
        },
        "priority_rule": {"type": "array", "items": {"type": "string"}},
        "brief_rationale": {"type": "string"},
    },
    "required": ["taxonomy", "priority_rule", "brief_rationale"],
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


def get_client() -> genai.Client:
    load_env_from_likely_locations()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set. Put it in .env or your system environment.")
    return genai.Client(api_key=api_key)


def safe_category_id(text: Any) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> str:
    text = strip_json_fence(text)
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return text[start : end + 1]


def read_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"Input file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["position_id", "speakerposition"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        sys.exit(f"Missing columns in input file: {missing}")
    df = df[required].copy()
    df["position_id"] = df["position_id"].astype(int)
    df["speakerposition"] = df["speakerposition"].fillna("").astype(str).str.strip()
    df = df[df["speakerposition"].ne("")].sort_values("position_id").reset_index(drop=True)
    return df


def build_prompt(positions_df: pd.DataFrame) -> str:
    lines = []
    for _, row in positions_df.iterrows():
        lines.append(
            f"position_id={int(row['position_id'])} | "
            f"speakerposition={row['speakerposition']}"
        )
    return (
        f"There are {len(positions_df)} unique non-empty speakerposition values.\n"
        "Every value is listed below with a stable position_id.\n"
        "Please propose the institutional taxonomy using the instructions.\n\n"
        "Speaker positions:\n\n"
        + "\n".join(lines)
    )


def parse_taxonomy_response(raw_text: str) -> dict[str, Any]:
    obj = json.loads(extract_json_object(raw_text))
    taxonomy = obj.get("taxonomy")
    if not isinstance(taxonomy, list):
        raise ValueError("Taxonomy response must contain a taxonomy array.")

    cleaned_taxonomy: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in taxonomy:
        if not isinstance(item, dict):
            continue
        category_id = safe_category_id(item.get("category_id", ""))
        if category_id in seen:
            raise ValueError(f"Duplicate category_id: {category_id}")
        examples = item.get("examples", [])
        if not isinstance(examples, list):
            examples = [str(examples)]
        cleaned_taxonomy.append(
            {
                "category_id": category_id,
                "category_label": str(item.get("category_label", category_id)).strip(),
                "definition": str(item.get("definition", "")).strip(),
                "inclusion_rule": str(item.get("inclusion_rule", "")).strip(),
                "exclusion_rule": str(item.get("exclusion_rule", "")).strip(),
                "institutional_role_basis": str(item.get("institutional_role_basis", "")).strip(),
                "examples": [str(x).strip() for x in examples if str(x).strip()],
            }
        )
        seen.add(category_id)

    if not (MIN_CATEGORIES <= len(cleaned_taxonomy) <= MAX_CATEGORIES):
        print(
            "Warning: taxonomy has "
            f"{len(cleaned_taxonomy)} categories. Expected between "
            f"{MIN_CATEGORIES} and {MAX_CATEGORIES}."
        )

    taxonomy_ids = {item["category_id"] for item in cleaned_taxonomy}
    priority_rule = [safe_category_id(x) for x in obj.get("priority_rule", [])]
    priority_rule = [x for x in priority_rule if x in taxonomy_ids]

    for item in cleaned_taxonomy:
        if item["category_id"] not in priority_rule:
            priority_rule.append(item["category_id"])

    return {
        "taxonomy": cleaned_taxonomy,
        "priority_rule": priority_rule,
        "brief_rationale": str(obj.get("brief_rationale", "")).strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask Gemini to discover a speakerposition taxonomy.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT_PATH)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT_PATH)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    positions_df = read_positions(args.input)
    prompt = build_prompt(positions_df)

    client = get_client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=args.temperature,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    print(f"Input: {args.input}")
    print(f"Unique positions sent to Gemini: {len(positions_df):,}")
    print("Prompt uses only: position_id, speakerposition")
    print(f"Calling {args.model}...")

    response = client.models.generate_content(
        model=args.model,
        contents=prompt,
        config=config,
    )
    raw_text = response.text or ""

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(raw_text, encoding="utf-8")
    print(f"Saved raw response to: {args.raw_output}")

    taxonomy = parse_taxonomy_response(raw_text)
    result = {
        "metadata": {
            "input_path": str(args.input),
            "output_json": str(args.output_json),
            "model": args.model,
            "temperature": args.temperature,
            "n_unique_positions": int(len(positions_df)),
            "prompt_columns_used": ["position_id", "speakerposition"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "taxonomy": taxonomy["taxonomy"],
        "priority_rule": taxonomy["priority_rule"],
        "brief_rationale": taxonomy["brief_rationale"],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved taxonomy JSON to: {args.output_json}")

    print("\nDiscovered taxonomy:")
    for item in taxonomy["taxonomy"]:
        print(f"- {item['category_id']}: {item['category_label']}")
        print(f"  {item['definition']}")

    print("\nPriority rule:")
    print(" > ".join(taxonomy["priority_rule"]))


if __name__ == "__main__":
    main()
