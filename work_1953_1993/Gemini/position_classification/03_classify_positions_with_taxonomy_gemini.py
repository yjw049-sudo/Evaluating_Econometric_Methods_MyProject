"""
Classify every unique speakerposition using the taxonomy discovered by Gemini.

This version asks Gemini to return ONLY:
    position_id
    category_id

The original speakerposition text is merged back by this script after parsing,
so the model does not need to repeat titles, confidence scores, or rationales.

Place and run this script inside:
    work_1953_1993/Gemini/position_classification

Default inputs:
    speakerposition_unique_values.csv
    position_taxonomy_gemini.json

Default outputs:
    speakerposition_institutional_types_gemini.csv
    speakerposition_institutional_types_gemini.json
    speakerposition_institutional_types_gemini.raw_response.txt

Usage:
    python 03_classify_positions_with_taxonomy_gemini.py
    python 03_classify_positions_with_taxonomy_gemini.py --batch-size 60
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
DEFAULT_POSITIONS_INPUT_PATH = ROOT / "speakerposition_unique_values.csv"
DEFAULT_TAXONOMY_INPUT_PATH = ROOT / "position_taxonomy_gemini.json"
DEFAULT_CSV_OUTPUT_PATH = ROOT / "speakerposition_institutional_types_gemini.csv"
DEFAULT_JSON_OUTPUT_PATH = ROOT / "speakerposition_institutional_types_gemini.json"
DEFAULT_RAW_OUTPUT_PATH = ROOT / "speakerposition_institutional_types_gemini.raw_response.txt"

MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.0
DEFAULT_BATCH_SIZE = 80
DEFAULT_MAX_RETRIES = 2


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "positions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position_id": {"type": "integer"},
                    "category_id": {"type": "string"},
                },
                "required": ["position_id", "category_id"],
            },
        },
    },
    "required": ["positions"],
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
        sys.exit(f"Positions input file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["position_id", "speakerposition"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        sys.exit(f"Missing columns in positions input: {missing}")

    cols = ["position_id", "speakerposition"]
    if "n_speeches_with_position" in df.columns:
        cols.append("n_speeches_with_position")

    df = df[cols].copy()
    df["position_id"] = df["position_id"].astype(int)
    df["speakerposition"] = df["speakerposition"].fillna("").astype(str).str.strip()
    df = df[df["speakerposition"].ne("")].sort_values("position_id").reset_index(drop=True)
    return df


def load_taxonomy(path: Path) -> dict[str, Any]:
    if not path.exists():
        sys.exit(f"Taxonomy file not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    taxonomy = obj.get("taxonomy")
    priority_rule = obj.get("priority_rule")
    if not isinstance(taxonomy, list):
        sys.exit("Taxonomy JSON must contain a taxonomy array.")
    if not isinstance(priority_rule, list):
        sys.exit("Taxonomy JSON must contain a priority_rule array.")

    cleaned_taxonomy = []
    for item in taxonomy:
        category_id = safe_category_id(item.get("category_id", ""))
        cleaned_taxonomy.append(
            {
                "category_id": category_id,
                "category_label": str(item.get("category_label", category_id)).strip(),
                "definition": str(item.get("definition", "")).strip(),
                "inclusion_rule": str(item.get("inclusion_rule", "")).strip(),
                "exclusion_rule": str(item.get("exclusion_rule", "")).strip(),
                "institutional_role_basis": str(item.get("institutional_role_basis", "")).strip(),
                "examples": item.get("examples", []),
            }
        )

    cleaned_priority_rule = [safe_category_id(x) for x in priority_rule]
    allowed = {item["category_id"] for item in cleaned_taxonomy}
    cleaned_priority_rule = [x for x in cleaned_priority_rule if x in allowed]
    for item in cleaned_taxonomy:
        if item["category_id"] not in cleaned_priority_rule:
            cleaned_priority_rule.append(item["category_id"])

    return {
        "taxonomy": cleaned_taxonomy,
        "priority_rule": cleaned_priority_rule,
        "brief_rationale": str(obj.get("brief_rationale", "")).strip(),
    }


def build_system_instruction(taxonomy: dict[str, Any]) -> str:
    taxonomy_lines = []
    for item in taxonomy["taxonomy"]:
        examples = item.get("examples", [])
        if isinstance(examples, list):
            examples_text = ", ".join(str(x) for x in examples)
        else:
            examples_text = str(examples)

        taxonomy_lines.append(
            "\n".join(
                [
                    f"category_id: {item['category_id']}",
                    f"category_label: {item['category_label']}",
                    f"definition: {item['definition']}",
                    f"inclusion_rule: {item['inclusion_rule']}",
                    f"exclusion_rule: {item['exclusion_rule']}",
                    f"institutional_role_basis: {item['institutional_role_basis']}",
                    f"examples: {examples_text}",
                ]
            )
        )

    allowed_ids = [item["category_id"] for item in taxonomy["taxonomy"]]

    return f"""\
You are classifying Canadian parliamentary speakerposition values into institutional position types.

Use ONLY the taxonomy below. Do not invent new categories.

Taxonomy:

{chr(10).join(taxonomy_lines)}

Allowed category_id values:
{", ".join(allowed_ids)}

Priority rule for composite positions with multiple titles:
{", ".join(taxonomy["priority_rule"])}

Classification principles:
- The position_id is only a stable identifier. Return the same position_id for each input row.
- Classify by the formal institutional role expressed in the title.
- Do not classify by party name, province, electoral district, policy field, ministerial department, ideology, or expected speech outcome.
- For composite titles separated by semicolons, choose the category with the highest priority according to the priority rule.
- Return one classification for every input position_id and do not include unlisted ids.
- Return ONLY position_id and category_id. Do not return confidence, rationale, notes, explanations, or copied speakerposition text.

Return valid JSON in this exact shape:
{{
  "positions": [
    {{
      "position_id": 0,
      "category_id": "{allowed_ids[0]}"
    }}
  ]
}}
"""


def build_prompt(batch_df: pd.DataFrame) -> str:
    lines = []
    for _, row in batch_df.iterrows():
        lines.append(
            f"position_id={int(row['position_id'])} | "
            f"speakerposition={row['speakerposition']}"
        )
    return "Classify these speakerposition values:\n\n" + "\n".join(lines)


def split_into_batches(df: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    return [df.iloc[start : start + batch_size].copy() for start in range(0, len(df), batch_size)]


def parse_response(
    raw_text: str,
    expected_position_ids: set[int],
    allowed_category_ids: set[str],
) -> dict[str, Any]:
    obj = json.loads(extract_json_object(raw_text))
    response_items = obj.get("positions", obj.get("items"))
    if not isinstance(response_items, list):
        raise ValueError("Response must contain a positions array.")

    seen: set[int] = set()
    cleaned_items: list[dict[str, Any]] = []

    for item in response_items:
        if not isinstance(item, dict):
            continue
        position_id = int(item.get("position_id"))
        if position_id in seen:
            continue

        # Primary expected key: category_id.
        # The fallback makes parsing robust if the model uses the old key once.
        raw_category = item.get("category_id", item.get("institutional_position_type", ""))
        category_id = safe_category_id(raw_category)

        if category_id not in allowed_category_ids:
            raise ValueError(
                f"Invalid category_id: {category_id}. "
                f"Allowed values: {sorted(allowed_category_ids)}"
            )

        cleaned_items.append(
            {
                "position_id": position_id,
                "category_id": category_id,
            }
        )
        seen.add(position_id)

    missing = sorted(expected_position_ids - seen)
    extra = sorted(seen - expected_position_ids)
    if missing:
        raise ValueError(f"Missing position_id values in response: {missing}")
    if extra:
        raise ValueError(f"Response contains unexpected position_id values: {extra}")

    return {"positions": sorted(cleaned_items, key=lambda x: x["position_id"])}


def classify_one_batch(
    client: genai.Client,
    model: str,
    config: types.GenerateContentConfig,
    batch_df: pd.DataFrame,
    allowed_category_ids: set[str],
    batch_number: int,
    total_batches: int,
    max_retries: int,
) -> tuple[list[dict[str, Any]], str]:
    prompt = build_prompt(batch_df)
    expected_ids = set(batch_df["position_id"].astype(int).tolist())
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 2):
        print(
            f"Calling {model} for batch {batch_number}/{total_batches} "
            f"with {len(batch_df):,} positions, attempt {attempt}..."
        )
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        raw_text = response.text or ""
        try:
            result = parse_response(raw_text, expected_ids, allowed_category_ids)
            return result["positions"], raw_text
        except Exception as exc:
            last_error = exc
            print(f"Batch {batch_number} attempt {attempt} failed: {exc}")

    raise RuntimeError(f"Batch {batch_number} failed after retries. Last error: {last_error}")


def classify_positions(
    positions_df: pd.DataFrame,
    taxonomy: dict[str, Any],
    client: genai.Client,
    model: str,
    temperature: float,
    batch_size: int,
    max_retries: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    system_instruction = build_system_instruction(taxonomy)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )
    allowed_category_ids = {item["category_id"] for item in taxonomy["taxonomy"]}
    all_items: list[dict[str, Any]] = []
    raw_responses: list[str] = []
    batches = split_into_batches(positions_df, batch_size)

    for batch_number, batch_df in enumerate(batches, start=1):
        items, raw_text = classify_one_batch(
            client=client,
            model=model,
            config=config,
            batch_df=batch_df,
            allowed_category_ids=allowed_category_ids,
            batch_number=batch_number,
            total_batches=len(batches),
            max_retries=max_retries,
        )
        raw_responses.append(
            "\n".join(
                [
                    f"===== BATCH {batch_number} START =====",
                    raw_text,
                    f"===== BATCH {batch_number} END =====",
                ]
            )
        )
        all_items.extend(items)

    return sorted(all_items, key=lambda x: x["position_id"]), raw_responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify speakerposition values using a discovered taxonomy.")
    parser.add_argument("--positions-input", type=Path, default=DEFAULT_POSITIONS_INPUT_PATH)
    parser.add_argument("--taxonomy-input", type=Path, default=DEFAULT_TAXONOMY_INPUT_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV_OUTPUT_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT_PATH)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT_PATH)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = parser.parse_args()

    if args.batch_size <= 0:
        sys.exit("--batch-size must be positive.")
    if args.max_retries < 0:
        sys.exit("--max-retries must be non-negative.")

    positions_df = read_positions(args.positions_input)
    taxonomy = load_taxonomy(args.taxonomy_input)
    client = get_client()

    print(f"Positions input: {args.positions_input}")
    print(f"Taxonomy input: {args.taxonomy_input}")
    print(f"Unique positions: {len(positions_df):,}")
    print(f"Batch size: {args.batch_size}")
    print("Prompt uses only: position_id, speakerposition, and the taxonomy from stage 2")
    print("Gemini output requested: position_id and category_id only")

    classified_items, raw_responses = classify_positions(
        positions_df=positions_df,
        taxonomy=taxonomy,
        client=client,
        model=args.model,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
    )

    classification_df = pd.DataFrame(classified_items)
    output_df = positions_df.merge(classification_df, on="position_id", how="left", validate="one_to_one")

    base_cols = ["position_id", "speakerposition"]
    if "n_speeches_with_position" in output_df.columns:
        base_cols.append("n_speeches_with_position")

    output_df = output_df[
        base_cols + ["category_id"]
    ].sort_values("position_id").reset_index(drop=True)

    result = {
        "metadata": {
            "positions_input": str(args.positions_input),
            "taxonomy_input": str(args.taxonomy_input),
            "output_csv": str(args.output_csv),
            "model": args.model,
            "temperature": args.temperature,
            "batch_size": args.batch_size,
            "max_retries": args.max_retries,
            "n_unique_positions": int(len(positions_df)),
            "taxonomy": taxonomy["taxonomy"],
            "priority_rule": taxonomy["priority_rule"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "positions": output_df.to_dict(orient="records"),
    }

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text("\n\n".join(raw_responses), encoding="utf-8")
    print(f"\nSaved raw responses to: {args.raw_output}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON to: {args.output_json}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved CSV to: {args.output_csv}")

    print("\nCategory counts:")
    counts = (
        output_df["category_id"]
        .value_counts(dropna=False)
        .rename_axis("category_id")
        .reset_index(name="count")
    )
    print(counts.to_string(index=False))

    print("\nFirst 20 classified positions:")
    print(output_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
