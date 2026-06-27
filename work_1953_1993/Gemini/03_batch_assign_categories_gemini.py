"""
Step 3 (Batch): Assign each unique topic phrase to one taxonomy category.

Place and run this script inside:
    work_1953_1993/Gemini

Inputs:
    speeches_topics_procedural.csv
    topic_taxonomy.json

Final output after parse:
    topic_to_category.csv

Typical use:
    python 03_batch_assign_categories_gemini.py make-jsonl --limit 200
    python 03_batch_assign_categories_gemini.py submit
    python 03_batch_assign_categories_gemini.py wait
    python 03_batch_assign_categories_gemini.py download
    python 03_batch_assign_categories_gemini.py parse

Requires:
    GEMINI_API_KEY in .env or in the system environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent
TOPICS_PATH = ROOT / "speeches_topics_procedural.csv"
TAXONOMY_PATH = ROOT / "topic_taxonomy.json"
OUTPUT_PATH = ROOT / "topic_to_category.csv"
REQUEST_JSONL_PATH = ROOT / "batch_requests_topic_categories.jsonl"
JOB_INFO_PATH = ROOT / "batch_job_topic_categories.json"
RESULTS_JSONL_PATH = ROOT / "batch_results_topic_categories.jsonl"
STATUS_JSON_PATH = ROOT / "batch_status_topic_categories.json"

MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.0
BATCH_SIZE = 50
POLL_SECONDS = 60
COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
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


def load_taxonomy(path: Path) -> tuple[list[str], str]:
    if not path.exists():
        sys.exit(f"Taxonomy file not found: {path}")
    taxonomy = json.loads(path.read_text(encoding="utf-8"))
    names = [category["name"] for category in taxonomy["categories"]]
    lines = [f"- {category['name']}: {category['definition']}" for category in taxonomy["categories"]]
    body = "\n".join(lines)

    instruction = f"""\
You are a research assistant labelling topical descriptions of speeches in the Canadian House of Commons, 1953-1993.
Each input item is a short noun phrase describing one speech topic.
Assign EXACTLY ONE category from the list below to each item.

Categories:
{body}

Pick the SINGLE best fit.
If a topic touches several categories, choose the category with the most substantive content.
Use the residual category only when no listed category is a reasonable match.

You will receive a JSON array of topic strings.
Return a JSON array of the same length where each element is the chosen category name.
Use exact strings from the category list. Preserve the input order. Do not return anything else.
"""
    return names, instruction


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "topic" not in df.columns:
        return set()
    return set(df["topic"].dropna().astype(str).tolist())


def load_unique_topics(topics_path: Path, output_path: Path, limit: int | None) -> list[str]:
    if not topics_path.exists():
        sys.exit(f"Topics file not found: {topics_path}")
    df = pd.read_csv(topics_path, usecols=["topic"], encoding="utf-8-sig")
    unique_topics = df["topic"].dropna().astype(str).unique().tolist()
    done = load_done(output_path)
    if done:
        print(f"Skipping {len(done):,} topics already assigned in {output_path.name}")
        unique_topics = [topic for topic in unique_topics if topic not in done]
    if limit is not None:
        unique_topics = unique_topics[:limit]
    return unique_topics


def response_schema(category_names: list[str]) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "enum": category_names},
    }


def build_request(batch: list[str], system_instruction: str, category_names: list[str]) -> dict[str, Any]:
    prompt = json.dumps(batch, ensure_ascii=False)
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generation_config": {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "temperature": TEMPERATURE,
            "response_mime_type": "application/json",
            "response_schema": response_schema(category_names),
        },
    }


def make_jsonl(args: argparse.Namespace) -> None:
    category_names, system_instruction = load_taxonomy(args.taxonomy)
    unique_topics = load_unique_topics(args.topics, args.output, args.limit)
    if not unique_topics:
        print("Nothing to write.")
        return

    batches = [unique_topics[i:i + args.batch_size] for i in range(0, len(unique_topics), args.batch_size)]
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.jsonl, "w", encoding="utf-8") as f:
        for idx, batch in enumerate(batches, start=1):
            record = {
                "key": f"topic_batch_{idx:06d}",
                "request": build_request(batch, system_instruction, category_names),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(batches):,} batch requests covering {len(unique_topics):,} topics to {args.jsonl}")


def object_to_jsonable(obj: Any) -> Any:
    for method in ("model_dump", "to_json_dict"):
        if hasattr(obj, method):
            try:
                return getattr(obj, method)()
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        return {k: object_to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [object_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: object_to_jsonable(v) for k, v in obj.items()}
    return obj


def submit(args: argparse.Namespace) -> None:
    if not args.jsonl.exists():
        sys.exit(f"JSONL file not found: {args.jsonl}. Run make-jsonl first.")
    if args.job_file.exists() and not args.force:
        sys.exit(f"Job file already exists: {args.job_file}. Use --force only if you intentionally want to create a new paid batch job.")

    client = get_client()
    uploaded_file = client.files.upload(
        file=str(args.jsonl),
        config=types.UploadFileConfig(display_name=args.jsonl.stem, mime_type="jsonl"),
    )
    print(f"Uploaded input file: {uploaded_file.name}")

    batch_job = client.batches.create(
        model=args.model,
        src=uploaded_file.name,
        generation_config={"display_name": args.display_name},
    )
    print(f"Created batch job: {batch_job.name}")

    info = {
        "job_name": batch_job.name,
        "model": args.model,
        "uploaded_file_name": uploaded_file.name,
        "request_jsonl": str(args.jsonl),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.job_file.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved job info to {args.job_file}")


def read_job_name(job_file: Path) -> str:
    if not job_file.exists():
        sys.exit(f"Job info file not found: {job_file}. Run submit first.")
    info = json.loads(job_file.read_text(encoding="utf-8"))
    job_name = info.get("job_name")
    if not job_name:
        sys.exit(f"No job_name found in {job_file}")
    return job_name


def get_state_name(batch_job: Any) -> str:
    state = getattr(batch_job, "state", None)
    name = getattr(state, "name", None)
    if name:
        return str(name)
    if isinstance(state, str):
        return state
    return str(state)


def status(args: argparse.Namespace) -> Any:
    client = get_client()
    job_name = read_job_name(args.job_file)
    batch_job = client.batches.get(name=job_name)
    state = get_state_name(batch_job)
    print(f"{job_name}: {state}")
    stats = getattr(batch_job, "batch_stats", None) or getattr(batch_job, "batchStats", None)
    if stats:
        print(f"batch_stats: {stats}")
    args.status_file.write_text(json.dumps(object_to_jsonable(batch_job), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved full status to {args.status_file}")
    return batch_job


def wait(args: argparse.Namespace) -> None:
    client = get_client()
    job_name = read_job_name(args.job_file)
    while True:
        batch_job = client.batches.get(name=job_name)
        state = get_state_name(batch_job)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {job_name}: {state}")
        args.status_file.write_text(json.dumps(object_to_jsonable(batch_job), indent=2, ensure_ascii=False), encoding="utf-8")
        if state in COMPLETED_STATES:
            print(f"Final state: {state}")
            return
        time.sleep(args.poll_seconds)


def find_result_file_name(batch_job: Any) -> str | None:
    dest = getattr(batch_job, "dest", None)
    if dest is None:
        return None
    for attr in ("file_name", "fileName"):
        value = getattr(dest, attr, None)
        if value:
            return str(value)
    if isinstance(dest, dict):
        return dest.get("file_name") or dest.get("fileName")
    return None


def download(args: argparse.Namespace) -> None:
    client = get_client()
    job_name = read_job_name(args.job_file)
    batch_job = client.batches.get(name=job_name)
    state = get_state_name(batch_job)
    if state != "JOB_STATE_SUCCEEDED":
        sys.exit(f"Batch job is not succeeded yet. Current state: {state}")
    result_file_name = find_result_file_name(batch_job)
    if not result_file_name:
        sys.exit("Could not find result file name in batch job. Check the saved status JSON.")
    print(f"Downloading result file: {result_file_name}")
    content = client.files.download(file=result_file_name)
    text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
    args.results.write_text(text, encoding="utf-8")
    print(f"Wrote results to {args.results}")


def cancel(args: argparse.Namespace) -> None:
    client = get_client()
    job_name = read_job_name(args.job_file)
    client.batches.cancel(name=job_name)
    print(f"Cancel requested for {job_name}")

# ---------- parsing ----------
def strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_text_from_response(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        text = getattr(response, "text", None)
        return str(text or "")
    if "text" in response:
        return str(response.get("text") or "")
    candidates = response.get("candidates") or []
    parts_text: list[str] = []
    for cand in candidates:
        content = cand.get("content", {}) if isinstance(cand, dict) else {}
        parts = content.get("parts", []) if isinstance(content, dict) else []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                parts_text.append(str(part["text"]))
    return "".join(parts_text)


def read_request_batches(jsonl_path: Path) -> dict[str, list[str]]:
    if not jsonl_path.exists():
        sys.exit(f"Request JSONL not found: {jsonl_path}. It is needed to map result keys back to topics.")
    out: dict[str, list[str]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec["key"]
            text = rec["request"]["contents"][0]["parts"][0]["text"]
            out[key] = json.loads(text)
    return out


def parse_result_line(line: str, request_batches: dict[str, list[str]], category_names: list[str]) -> list[dict[str, str]]:
    raw = json.loads(line)
    key = raw.get("key") or raw.get("metadata", {}).get("key")
    response = raw.get("response") or raw.get("inlineResponse", {}).get("response")
    error = raw.get("error") or raw.get("status")
    topics = request_batches.get(str(key), [])
    if not key:
        return [{"topic": "", "category": "", "error": f"No key found: {str(raw)[:200]}"}]
    if error and not response:
        return [{"topic": topic, "category": "", "error": json.dumps(error, ensure_ascii=False)[:300]} for topic in topics]
    text = extract_text_from_response(response)
    if not text:
        return [{"topic": topic, "category": "", "error": "empty response text"} for topic in topics]
    try:
        labels = json.loads(strip_json_fence(text))
        if not isinstance(labels, list):
            raise ValueError("model output is not a JSON array")
        if len(labels) != len(topics):
            raise ValueError(f"expected {len(topics)} labels, got {len(labels)}")
        category_set = set(category_names)
        fallback = category_names[-1]
        rows = []
        for topic, label in zip(topics, labels):
            label = str(label)
            if label not in category_set:
                label = fallback
            rows.append({"topic": topic, "category": label, "error": ""})
        return rows
    except Exception as exc:
        return [{"topic": topic, "category": "", "error": f"ParseError: {exc}: {text[:200]}"} for topic in topics]


def parse(args: argparse.Namespace) -> None:
    if not args.results.exists():
        sys.exit(f"Results file not found: {args.results}. Run download first.")
    category_names, _ = load_taxonomy(args.taxonomy)
    request_batches = read_request_batches(args.jsonl)
    rows: list[dict[str, str]] = []
    with open(args.results, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.extend(parse_result_line(line, request_batches, category_names))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["topic", "category", "error"])
        writer.writeheader()
        writer.writerows(rows)
    n_error = sum(1 for r in rows if r.get("error"))
    print(f"Wrote {len(rows):,} rows to {args.output}; errors: {n_error:,}")


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jsonl", type=Path, default=REQUEST_JSONL_PATH)
    parser.add_argument("--job-file", type=Path, default=JOB_INFO_PATH)
    parser.add_argument("--status-file", type=Path, default=STATUS_JSON_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_JSONL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch workflow for assigning topic phrases to taxonomy categories")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-jsonl", help="create JSONL batch input file")
    p.add_argument("--topics", type=Path, default=TOPICS_PATH)
    p.add_argument("--taxonomy", type=Path, default=TAXONOMY_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--jsonl", type=Path, default=REQUEST_JSONL_PATH)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.set_defaults(func=make_jsonl)

    p = sub.add_parser("submit", help="upload JSONL and create batch job")
    add_common_paths(p)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--display-name", default="topic-to-category")
    p.add_argument("--force", action="store_true", help="allow creating another paid job even if job info exists")
    p.set_defaults(func=submit)

    p = sub.add_parser("status", help="check batch job status")
    add_common_paths(p)
    p.set_defaults(func=status)

    p = sub.add_parser("wait", help="poll until the batch job finishes")
    add_common_paths(p)
    p.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    p.set_defaults(func=wait)

    p = sub.add_parser("download", help="download batch result JSONL after success")
    add_common_paths(p)
    p.set_defaults(func=download)

    p = sub.add_parser("parse", help="parse result JSONL into CSV")
    add_common_paths(p)
    p.set_defaults(func=parse)

    p = sub.add_parser("cancel", help="cancel the current batch job")
    add_common_paths(p)
    p.set_defaults(func=cancel)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
