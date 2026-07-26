"""
Score K-means clusters by procedural / partisan / formal tendency using Gemini.

This script reads K-means cluster top words and asks Gemini to assign a
cluster-level procedural_score using the same 0, 0.25, 0.50, 0.75, 1 scale used
for speech-level scoring.

Place and run this script inside:
    Gemini_merged

Default external input:
    ../Kmeans/Kmeans_merged/cluster_summary.csv

Default outputs written inside Gemini_merged:
    cluster_procedural_scores_gemini.csv
    cluster_procedural_scores_gemini.json
    cluster_procedural_scores_gemini.raw_response.txt

Important:
    The prompt uses only cluster id and top words.
    It does not send share, count, or cluster size to Gemini.

Output CSV columns:
    cluster
    top_words
    cluster_score

Usage:
    python 05_score_kmeans_clusters_gemini.py
    python 05_score_kmeans_clusters_gemini.py --model gemini-2.5-flash-lite
    python 05_score_kmeans_clusters_gemini.py --input ../Kmeans/Kmeans_merged/cluster_summary.csv
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
DEFAULT_INPUT_PATH = ROOT.parent / "Kmeans" / "Kmeans_merged" / "cluster_summary.csv"
DEFAULT_CSV_OUTPUT_PATH = ROOT / "cluster_procedural_scores_gemini.csv"
DEFAULT_JSON_OUTPUT_PATH = ROOT / "cluster_procedural_scores_gemini.json"
DEFAULT_RAW_OUTPUT_PATH = ROOT / "cluster_procedural_scores_gemini.raw_response.txt"

MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.0

CLUSTER_COL_CANDIDATES = ["cluster", "Cluster", "kmeans_cluster", "cluster_id"]
TOP_WORDS_COL_CANDIDATES = [
    "top_10_words",
    "top_words",
    "Top 10 words",
    "Top words",
    "keywords",
    "terms",
]

SYSTEM_INSTRUCTION = """\
You are a research assistant analysing K-means clusters from Canadian House of Commons speeches, 1963-1993.

You will be given a cluster summary table. Each row contains a cluster id and its top words or phrases. For each cluster, assign a cluster-level `procedural_score` based only on the top words and phrases.

Use this score rule:
- 0 = almost entirely substantive policy, administration, legislation, public spending, economic/social issues, or government action.
- 0.25 = mostly substantive, with minor procedural formulae or partisan rhetoric.
- 0.50 = mixed substantive and procedural/partisan/formal content.
- 0.75 = mostly procedural, partisan attack, obstruction/delay, or formal intervention, with some substantive content.
- 1 = almost entirely procedural, partisan, dilatory, or formal, with little substantive policy content.

Procedural, partisan, dilatory, or formal content includes points of order, privilege, Speaker rulings, adjournment, closure, time allocation, order of business, committee or bill-stage mechanics, divisions, standing orders, unanimous consent, tabling, leave of the House, short formal interventions, repeated objections, delay tactics, or clusters mainly indicating criticism of parties, governments, oppositions, leaders, or electoral records instead of substantive policy discussion.

When in doubt, give a lower score if the top words suggest meaningful substantive policy discussion.
Choose only one of these values: 0, 0.25, 0.50, 0.75, 1.
Return every cluster exactly once. Return no explanation.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster": {"type": "integer"},
                    "procedural_score": {"type": "string", "enum": ["0", "0.25", "0.5", "0.75", "1"]},
                },
                "required": ["cluster", "procedural_score"],
            },
        },
    },
    "required": ["clusters"],
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


def first_existing_column(df: pd.DataFrame, candidates: list[str], required_name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    sys.exit(
        f"Could not find {required_name} column. Expected one of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def read_cluster_summary(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    if not path.exists():
        sys.exit(f"Input file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    cluster_col = first_existing_column(df, CLUSTER_COL_CANDIDATES, "cluster")
    top_words_col = first_existing_column(df, TOP_WORDS_COL_CANDIDATES, "top words")

    out = df[[cluster_col, top_words_col]].copy()
    out = out.rename(columns={cluster_col: "cluster", top_words_col: "top_words"})
    out = out[out["cluster"].notna()].copy()
    out["cluster"] = out["cluster"].astype(int)
    out["top_words"] = out["top_words"].fillna("").astype(str)
    out = out.sort_values("cluster").reset_index(drop=True)

    source_cols = {
        "cluster_col": cluster_col,
        "top_words_col": top_words_col,
    }
    return out, source_cols


def build_prompt(df: pd.DataFrame) -> str:
    lines = ["cluster\ttop_words"]
    for _, row in df.iterrows():
        lines.append(f"{int(row['cluster'])}\t{row['top_words']}")
    return "\n".join(lines)


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
        raise ValueError("No JSON object found in model response")
    return text[start : end + 1]


def normalise_score(value: Any) -> float:
    allowed = [0.0, 0.25, 0.5, 0.75, 1.0]
    x = float(value)
    return min(allowed, key=lambda v: abs(v - x))


def parse_response(raw_text: str, expected_clusters: set[int]) -> dict[str, Any]:
    obj = json.loads(extract_json_object(raw_text))
    if not isinstance(obj, dict):
        raise ValueError("Response must be a JSON object")

    response_items = obj.get("clusters", obj.get("items"))
    if not isinstance(response_items, list):
        raise ValueError("Response must be a JSON object with a 'clusters' array")

    seen: set[int] = set()
    cleaned_items: list[dict[str, Any]] = []
    for item in response_items:
        if not isinstance(item, dict):
            continue
        cluster = int(item.get("cluster"))
        if cluster in seen:
            continue
        cleaned_items.append(
            {
                "cluster": cluster,
                "procedural_score": normalise_score(item.get("procedural_score")),
            }
        )
        seen.add(cluster)

    missing = sorted(expected_clusters - seen)
    extra = sorted(seen - expected_clusters)
    if missing:
        raise ValueError(f"Missing clusters in response: {missing}")
    if extra:
        print(f"Warning: response contains unexpected clusters: {extra}")

    obj["clusters"] = sorted(cleaned_items, key=lambda x: x["cluster"])
    obj.setdefault("notes", "")
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Score K-means clusters by procedural tendency using Gemini")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV_OUTPUT_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT_PATH)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT_PATH)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    df, source_cols = read_cluster_summary(args.input)
    if df.empty:
        sys.exit("No clusters found in input file.")

    prompt = build_prompt(df)
    expected_clusters = set(df["cluster"].astype(int).tolist())

    client = get_client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=args.temperature,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    print(f"Input: {args.input}")
    print(f"Clusters: {len(df):,}")
    print("Prompt uses only: cluster, top_words")
    print(f"Calling {args.model}...")
    resp = client.models.generate_content(model=args.model, contents=prompt, config=config)
    raw_text = resp.text or ""

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(raw_text, encoding="utf-8")
    print(f"Saved raw response to {args.raw_output}")

    try:
        result = parse_response(raw_text, expected_clusters)
    except Exception as exc:
        sys.exit(f"Could not parse Gemini response. See {args.raw_output}. Error: {exc}")

    result["metadata"] = {
        "input_path": str(args.input),
        "output_csv": str(args.output_csv),
        "model": args.model,
        "temperature": args.temperature,
        "n_clusters": int(len(df)),
        "source_columns": source_cols,
        "prompt_columns_used": ["cluster", "top_words"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_scale": [0, 0.25, 0.5, 0.75, 1],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON to {args.output_json}")

    score_df = pd.DataFrame(result["clusters"])
    out = df.merge(score_df, on="cluster", how="left")
    out = out.rename(columns={"procedural_score": "cluster_score"})
    out = out[["cluster", "top_words", "cluster_score"]]
    out = out.sort_values("cluster").reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved CSV to {args.output_csv}")

    print("\nCluster procedural scores:")
    for _, row in out.iterrows():
        print(f"cluster {int(row['cluster']):>2}: {row['cluster_score']}")


if __name__ == "__main__":
    main()







