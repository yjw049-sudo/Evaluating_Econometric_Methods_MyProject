"""
Step 1 (Batch, auto-chunked): Generate a short topic phrase and a 0-1 procedural score
for each speech.

Place and run this script inside:
    Gemini_merged

External input (read directly; no intermediate copy is required):
    ../output/Speech_sample.csv

Required columns (all other CSV columns are ignored):
    basepk
    speechtext

Final output after merge:
    speeches_topics_procedural.csv

Output columns:
    basepk
    topic
    procedural_score
    error

Recommended chunked workflow:
    python 01_batch_classify_topics_gemini.py make-chunks --chunk-size 1000
    python 01_batch_classify_topics_gemini.py submit --suffix part001
    python 01_batch_classify_topics_gemini.py wait --suffix part001
    python 01_batch_classify_topics_gemini.py download --suffix part001
    python 01_batch_classify_topics_gemini.py parse --suffix part001
    python 01_batch_classify_topics_gemini.py merge-parts

For a quick single-batch test:
    python 01_batch_classify_topics_gemini.py make-jsonl --limit 20
    python 01_batch_classify_topics_gemini.py submit
    python 01_batch_classify_topics_gemini.py wait
    python 01_batch_classify_topics_gemini.py download
    python 01_batch_classify_topics_gemini.py parse

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

# ---------- paths and config ----------
ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT.parent / "output" / "Speech_sample.csv"
OUTPUT_PATH = ROOT / "speeches_topics_procedural.csv"

REQUEST_JSONL_PATH = ROOT / "batch_requests_topics_procedural.jsonl"
JOB_INFO_PATH = ROOT / "batch_job_topics_procedural.json"
RESULTS_JSONL_PATH = ROOT / "batch_results_topics_procedural.jsonl"
STATUS_JSON_PATH = ROOT / "batch_status_topics_procedural.json"

PARTS_DIR = ROOT / "batch_parts"
MANIFEST_PATH = PARTS_DIR / "chunk_manifest.csv"

BASEPK_COLUMN = "basepk"
TEXT_COLUMN = "speechtext"
TEXT_COLUMN_ALIASES = ["speechtext", "speechtext_original", "speechtext_oringinal"]

MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.0
MAX_CHARS = 6000
POLL_SECONDS = 60
# Keep each submitted job conservative under the model-wide queued-token quota.
# run-chunks submits jobs sequentially and retries temporary quota failures.
DEFAULT_CHUNK_SIZE = 1000

OUT_FIELDS = ["basepk", "topic", "procedural_score", "error"]
COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

SYSTEM_INSTRUCTION = """\
You are a research assistant analysing speeches from the Canadian House of Commons, 1953-1993.

For each speech, return valid JSON with two fields:

1. `topic`: a short noun phrase, maximum 8 words, describing the main substantive topic.
2. `procedural_score`: a number from 0 to 1 indicating how far the speech moves away from substantive policy discussion toward procedural, partisan, dilatory, or purely formal parliamentary language.

Topic rules:
- Be specific: "old-age pensions" not "social policy"; "railway freight rates" not "transportation".
- Lower-case unless a proper noun requires capitalisation.
- Ignore routine openings such as "Mr. Speaker".
- If substantive content is present, label the substantive topic.
- If no substantive topic can be identified, use "unclear".

Score rule:
- 0 = almost entirely substantive policy, administration, legislation, public spending, economic/social issues, or government action.
- 0.25 = mostly substantive, with minor procedural formulae or partisan rhetoric.
- 0.50 = mixed substantive and procedural/partisan/formal content.
- 0.75 = mostly procedural, partisan attack, obstruction/delay, or formal intervention, with some substantive content.
- 1 = almost entirely procedural, partisan, dilatory, or formal, with little substantive policy content.

Procedural, partisan, dilatory, or formal content includes points of order, privilege, Speaker rulings, adjournment, closure, time allocation, order of business, committee or bill-stage mechanics, divisions, standing orders, unanimous consent, tabling, leave of the House, short formal interventions, repeated objections, delay tactics, or speeches mainly criticising parties, governments, oppositions, leaders, or electoral records instead of developing a policy argument.

When in doubt, give a lower score if the speech contains meaningful substantive policy discussion.

Return only this JSON shape:
{"topic": "<short noun phrase>", "procedural_score": 0.0}
"""

PROMPT_TEMPLATE = """\
BasePK: {basepk}

Speech:
{speechtext}
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "procedural_score": {"type": "number"},
    },
    "required": ["topic", "procedural_score"],
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


def get_client() -> genai.Client:
    load_env_from_likely_locations()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set. Put it in .env or your system environment.")
    return genai.Client(api_key=api_key)


# ---------- data ----------
def read_input_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"Input file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if BASEPK_COLUMN not in df.columns:
        sys.exit(f"Required column '{BASEPK_COLUMN}' not found. Available columns: {list(df.columns)}")

    text_col = None
    for candidate in TEXT_COLUMN_ALIASES:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        sys.exit(f"Required text column not found. Expected one of {TEXT_COLUMN_ALIASES}. Available columns: {list(df.columns)}")

    if text_col != TEXT_COLUMN:
        print(f"Using text column '{text_col}' because '{TEXT_COLUMN}' was not found.")

    df = df[[BASEPK_COLUMN, text_col]].rename(columns={text_col: TEXT_COLUMN}).copy()
    df = df[df[BASEPK_COLUMN].notna()].copy()
    df = df[df[TEXT_COLUMN].notna() & (df[TEXT_COLUMN].astype(str).str.strip() != "")].copy()
    df = df.drop_duplicates(subset=[BASEPK_COLUMN], keep="first")
    df[BASEPK_COLUMN] = df[BASEPK_COLUMN].astype(int)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    return df.reset_index(drop=True)


def load_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    try:
        done = pd.read_csv(path, usecols=[BASEPK_COLUMN], encoding="utf-8-sig")[BASEPK_COLUMN].dropna().astype(int)
        return set(done.tolist())
    except Exception:
        return set()


def build_prompt(row: pd.Series) -> str:
    text = str(row.get(TEXT_COLUMN) or "")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + " [...truncated]"
    return PROMPT_TEMPLATE.format(basepk=row[BASEPK_COLUMN], speechtext=text)


def build_request(prompt: str) -> dict[str, Any]:
    """Build one JSONL request line for Gemini Batch API.

    Important: do not use the Python SDK keyword name ``config`` inside JSONL.
    File-based Batch JSONL expects ``generation_config``. ``system_instruction``
    is a top-level request field.
    """
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}],
        },
        "generation_config": {
            "temperature": TEMPERATURE,
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
        },
    }


def write_jsonl_for_df(df: pd.DataFrame, jsonl_path: Path) -> int:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            basepk = int(row[BASEPK_COLUMN])
            record = {
                "key": str(basepk),
                "request": build_request(build_prompt(row)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(df)


# ---------- chunk path helpers ----------
def normalise_suffix(suffix: str | None) -> str | None:
    if suffix is None:
        return None
    suffix = str(suffix).strip()
    if not suffix:
        return None
    suffix = suffix.replace("-", "_").replace(" ", "_")
    if not re.match(r"^[A-Za-z0-9_]+$", suffix):
        sys.exit("Suffix can only contain letters, numbers, and underscores.")
    return suffix


def chunk_paths(suffix: str | None) -> dict[str, Path]:
    suffix = normalise_suffix(suffix)
    if not suffix:
        return {
            "jsonl": REQUEST_JSONL_PATH,
            "job_file": JOB_INFO_PATH,
            "status_file": STATUS_JSON_PATH,
            "results": RESULTS_JSONL_PATH,
            "output": OUTPUT_PATH,
        }
    return {
        "jsonl": PARTS_DIR / f"batch_requests_topics_procedural_{suffix}.jsonl",
        "job_file": PARTS_DIR / f"batch_job_topics_procedural_{suffix}.json",
        "status_file": PARTS_DIR / f"batch_status_topics_procedural_{suffix}.json",
        "results": PARTS_DIR / f"batch_results_topics_procedural_{suffix}.jsonl",
        "output": PARTS_DIR / f"speeches_topics_procedural_{suffix}.csv",
    }


def apply_suffix_paths(args: argparse.Namespace) -> None:
    suffix = normalise_suffix(getattr(args, "suffix", None))
    if not suffix:
        return
    paths = chunk_paths(suffix)
    args.jsonl = paths["jsonl"]
    args.job_file = paths["job_file"]
    args.status_file = paths["status_file"]
    args.results = paths["results"]
    args.output = paths["output"]


def infer_suffix_from_path(path: Path, prefix: str, suffix: str) -> str:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return ""
    return name[len(prefix): -len(suffix)]


# ---------- JSON-safe conversion ----------
def object_to_jsonable(obj: Any) -> Any:
    """Convert Google SDK objects into plain JSON-safe Python values."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [object_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): object_to_jsonable(v) for k, v in obj.items()}
    for method in ("model_dump", "to_json_dict"):
        if hasattr(obj, method):
            try:
                return object_to_jsonable(getattr(obj, method)())
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        return {k: object_to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


# ---------- batch workflow ----------
def make_jsonl(args: argparse.Namespace) -> None:
    apply_suffix_paths(args)
    df = read_input_csv(args.input)

    # Resume from an existing final output or part output.
    done = load_done(args.output)
    if done:
        print(f"Skipping {len(done):,} basepks already in {args.output.name}")
        df = df[~df[BASEPK_COLUMN].astype(int).isin(done)].reset_index(drop=True)

    start = max(0, int(args.start or 0))
    if start:
        df = df.iloc[start:].reset_index(drop=True)
        print(f"Start offset applied: {start:,}")

    if args.limit is not None:
        df = df.head(args.limit).copy()
        print(f"Limit applied: {len(df):,} speeches")

    if df.empty:
        print("Nothing to write.")
        return

    n = write_jsonl_for_df(df, args.jsonl)
    print(f"Wrote {n:,} requests to {args.jsonl}")


def make_chunks(args: argparse.Namespace) -> None:
    """Split the input into many JSONL files.

    This command does not call Gemini and does not cost money.
    It creates batch_parts/chunk_manifest.csv and one JSONL per chunk.
    """
    df = read_input_csv(args.input)

    done = load_done(args.output)
    if done and not args.include_done:
        print(f"Skipping {len(done):,} basepks already in {args.output.name}")
        df = df[~df[BASEPK_COLUMN].astype(int).isin(done)].reset_index(drop=True)

    if args.start:
        df = df.iloc[args.start:].reset_index(drop=True)
        print(f"Start offset applied: {args.start:,}")

    if args.limit is not None:
        df = df.head(args.limit).copy()
        print(f"Global limit applied: {len(df):,} speeches")

    if df.empty:
        print("Nothing to chunk.")
        return

    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    chunk_size = int(args.chunk_size)
    if chunk_size <= 0:
        sys.exit("--chunk-size must be positive.")

    manifest_rows: list[dict[str, Any]] = []
    total = len(df)
    n_chunks = (total + chunk_size - 1) // chunk_size
    width = max(3, len(str(n_chunks)))

    for i in range(n_chunks):
        if args.max_chunks is not None and i >= args.max_chunks:
            break
        suffix = f"part{i + 1:0{width}d}"
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total)
        sub = df.iloc[start_idx:end_idx].copy()
        paths = chunk_paths(suffix)

        if paths["jsonl"].exists() and not args.overwrite:
            print(f"Skip existing {paths['jsonl'].name}; use --overwrite to rewrite")
        else:
            n = write_jsonl_for_df(sub, paths["jsonl"])
            print(f"Wrote {n:,} requests to {paths['jsonl']}")

        manifest_rows.append({
            "suffix": suffix,
            "start_index": start_idx,
            "end_index_exclusive": end_idx,
            "n_requests": len(sub),
            "jsonl": str(paths["jsonl"]),
            "job_file": str(paths["job_file"]),
            "status_file": str(paths["status_file"]),
            "results": str(paths["results"]),
            "output": str(paths["output"]),
        })

    pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    print(f"\nWrote manifest: {MANIFEST_PATH}")
    print(f"Chunks written: {len(manifest_rows):,}")
    print("\nNext example:")
    if manifest_rows:
        first = manifest_rows[0]["suffix"]
        print(f"  python {Path(__file__).name} submit --suffix {first}")
        print(f"  python {Path(__file__).name} wait --suffix {first}")
        print(f"  python {Path(__file__).name} download --suffix {first}")
        print(f"  python {Path(__file__).name} parse --suffix {first}")


def submit(args: argparse.Namespace) -> None:
    apply_suffix_paths(args)

    if not args.jsonl.exists():
        sys.exit(f"JSONL file not found: {args.jsonl}. Run make-jsonl or make-chunks first.")
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
        config={"display_name": args.display_name},
    )
    job_name = batch_job.name
    print(f"Created batch job: {job_name}")

    info = {
        "job_name": job_name,
        "model": args.model,
        "uploaded_file_name": uploaded_file.name,
        "request_jsonl": str(args.jsonl),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suffix": normalise_suffix(getattr(args, "suffix", None)),
    }
    args.job_file.parent.mkdir(parents=True, exist_ok=True)
    args.job_file.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved job info to {args.job_file}")


def submit_next(args: argparse.Namespace) -> None:
    """Submit the first chunk in the manifest that does not yet have a job file.

    This is safer than submitting all chunks at once, because your Batch queued-token
    limit may be low. Run submit-next, then wait/download/parse that suffix, then run
    submit-next again.
    """
    if not MANIFEST_PATH.exists():
        sys.exit(f"Manifest not found: {MANIFEST_PATH}. Run make-chunks first.")

    manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig")
    for _, row in manifest.iterrows():
        suffix = str(row["suffix"])
        paths = chunk_paths(suffix)
        if not paths["job_file"].exists():
            print(f"Next unsubmitted chunk: {suffix}")
            args.suffix = suffix
            args.jsonl = paths["jsonl"]
            args.job_file = paths["job_file"]
            args.status_file = paths["status_file"]
            args.results = paths["results"]
            args.output = paths["output"]
            submit(args)
            print(f"\nAfter it is submitted, use:")
            print(f"  python {Path(__file__).name} wait --suffix {suffix}")
            print(f"  python {Path(__file__).name} download --suffix {suffix}")
            print(f"  python {Path(__file__).name} parse --suffix {suffix}")
            return
    print("All chunks in the manifest already have job files.")


def _is_quota_or_rate_error(exc: BaseException) -> bool:
    msg = str(exc)
    signals = ("429", "RESOURCE_EXHAUSTED", "quota", "rate limit", "Too Many Requests")
    return any(s in msg for s in signals)


def _manifest_suffixes(from_suffix: str | None = None, to_suffix: str | None = None) -> list[str]:
    if not MANIFEST_PATH.exists():
        sys.exit(f"Manifest not found: {MANIFEST_PATH}. Run make-chunks first.")
    manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig")
    suffixes = [str(x) for x in manifest["suffix"].tolist()]
    if from_suffix:
        from_suffix = normalise_suffix(from_suffix)
        if from_suffix not in suffixes:
            sys.exit(f"from-suffix not found in manifest: {from_suffix}")
        suffixes = suffixes[suffixes.index(from_suffix):]
    if to_suffix:
        to_suffix = normalise_suffix(to_suffix)
        if to_suffix not in suffixes:
            sys.exit(f"to-suffix not found in selected manifest range: {to_suffix}")
        suffixes = suffixes[:suffixes.index(to_suffix) + 1]
    return suffixes


def _get_batch_state_for_suffix(client: genai.Client, suffix: str) -> tuple[str, Any | None]:
    paths = chunk_paths(suffix)
    if not paths["job_file"].exists():
        return "NOT_SUBMITTED", None
    job_name = read_job_name(paths["job_file"])
    batch_job = client.batches.get(name=job_name)
    state = get_state_name(batch_job)
    paths["status_file"].parent.mkdir(parents=True, exist_ok=True)
    paths["status_file"].write_text(
        json.dumps(object_to_jsonable(batch_job), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return state, batch_job


def _wait_for_suffix(client: genai.Client, suffix: str, poll_seconds: int) -> str:
    paths = chunk_paths(suffix)
    job_name = read_job_name(paths["job_file"])
    while True:
        batch_job = client.batches.get(name=job_name)
        state = get_state_name(batch_job)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {suffix}  {job_name}: {state}")
        paths["status_file"].parent.mkdir(parents=True, exist_ok=True)
        paths["status_file"].write_text(
            json.dumps(object_to_jsonable(batch_job), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if state in COMPLETED_STATES:
            return state
        time.sleep(poll_seconds)


def run_chunks(args: argparse.Namespace) -> None:
    """Sequentially submit, wait, download, and parse chunks.

    This command is designed for low Batch queued-token limits. It keeps at most
    one chunk submitted/running at a time: submit one part, wait until it finishes,
    download and parse it, then move to the next part.

    The command is resumable. If a part already has a parsed CSV, it is skipped.
    If a part already has a job file but no parsed CSV, the existing job is checked
    and continued instead of creating a new paid job.
    """
    suffixes = _manifest_suffixes(args.from_suffix, args.to_suffix)
    if args.max_chunks is not None:
        suffixes = suffixes[:args.max_chunks]
    if not suffixes:
        print("No chunks selected.")
        return

    print(f"Selected chunks: {suffixes[0]} to {suffixes[-1]} ({len(suffixes):,} chunks)")
    print("This will run sequentially: submit -> wait -> download -> parse for each chunk.")
    print("Do not close the terminal. Disable sleep mode if this will run overnight.\n")

    client = get_client()
    completed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for idx, suffix in enumerate(suffixes, 1):
        paths = chunk_paths(suffix)
        print("=" * 80)
        print(f"Chunk {idx:,}/{len(suffixes):,}: {suffix}")
        print("=" * 80)

        if paths["output"].exists() and not args.reparse:
            print(f"Parsed CSV already exists, skipping: {paths['output']}")
            skipped.append(suffix)
            continue

        # Submit only if no job file exists. If quota is temporarily exhausted,
        # wait and retry without creating duplicate jobs.
        if not paths["job_file"].exists():
            submit_args = argparse.Namespace(
                suffix=suffix,
                jsonl=paths["jsonl"],
                job_file=paths["job_file"],
                status_file=paths["status_file"],
                results=paths["results"],
                output=paths["output"],
                model=args.model,
                display_name=f"{args.display_name}-{suffix}",
                force=False,
            )
            attempt = 0
            while True:
                attempt += 1
                try:
                    print(f"Submitting {suffix} (attempt {attempt})...")
                    submit(submit_args)
                    break
                except Exception as exc:
                    if _is_quota_or_rate_error(exc) and attempt <= args.max_submit_retries:
                        print(f"Submit hit quota/rate limit: {type(exc).__name__}: {str(exc)[:250]}")
                        print(f"Waiting {args.submit_retry_seconds} seconds before retrying...")
                        time.sleep(args.submit_retry_seconds)
                        continue
                    failed.append(suffix)
                    print(f"Submit failed for {suffix}: {type(exc).__name__}: {exc}")
                    if args.continue_on_failure:
                        break
                    sys.exit(f"Stopping at {suffix}. Fix the error, then rerun run-chunks to resume.")
            if suffix in failed and args.continue_on_failure:
                continue
        else:
            print(f"Existing job file found; will continue existing job: {paths['job_file']}")

        # Wait until this chunk reaches a terminal state.
        state = _wait_for_suffix(client, suffix, args.poll_seconds)
        if state != "JOB_STATE_SUCCEEDED":
            failed.append(suffix)
            print(f"Chunk {suffix} ended with state {state}")
            if args.continue_on_failure:
                continue
            sys.exit(f"Stopping at {suffix}. Check {paths['status_file']}")

        # Download results unless already downloaded.
        if paths["results"].exists() and not args.redownload:
            print(f"Result JSONL already exists, skipping download: {paths['results']}")
        else:
            download_args = argparse.Namespace(
                suffix=suffix,
                jsonl=paths["jsonl"],
                job_file=paths["job_file"],
                status_file=paths["status_file"],
                results=paths["results"],
                output=paths["output"],
            )
            download(download_args)

        # Parse results.
        parse_args = argparse.Namespace(
            suffix=suffix,
            jsonl=paths["jsonl"],
            job_file=paths["job_file"],
            status_file=paths["status_file"],
            results=paths["results"],
            output=paths["output"],
        )
        parse(parse_args)
        completed.append(suffix)

    print("\n" + "=" * 80)
    print("run-chunks finished")
    print(f"Completed this run: {len(completed):,}")
    print(f"Skipped existing parsed CSVs: {len(skipped):,}")
    print(f"Failed: {len(failed):,}")

    if args.merge:
        print("\nMerging parsed part CSV files...")
        merge_args = argparse.Namespace(output=args.output)
        merge_parts(merge_args)
    else:
        print("Merge skipped. Run merge-parts later when ready.")


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
    apply_suffix_paths(args)
    client = get_client()
    job_name = read_job_name(args.job_file)
    batch_job = client.batches.get(name=job_name)
    state = get_state_name(batch_job)
    print(f"{job_name}: {state}")
    stats = getattr(batch_job, "batch_stats", None) or getattr(batch_job, "batchStats", None)
    if stats:
        print(f"batch_stats: {stats}")
    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    args.status_file.write_text(json.dumps(object_to_jsonable(batch_job), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved full status to {args.status_file}")
    return batch_job


def status_all(args: argparse.Namespace) -> None:
    if not MANIFEST_PATH.exists():
        sys.exit(f"Manifest not found: {MANIFEST_PATH}. Run make-chunks first.")
    client = get_client()
    manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig")
    rows = []
    for _, row in manifest.iterrows():
        suffix = str(row["suffix"])
        paths = chunk_paths(suffix)
        state = "NOT_SUBMITTED"
        job_name = ""
        if paths["job_file"].exists():
            job_name = read_job_name(paths["job_file"])
            try:
                batch_job = client.batches.get(name=job_name)
                state = get_state_name(batch_job)
                paths["status_file"].write_text(json.dumps(object_to_jsonable(batch_job), indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                state = f"STATUS_ERROR: {type(exc).__name__}: {str(exc)[:120]}"
        parsed = paths["output"].exists()
        rows.append({"suffix": suffix, "state": state, "parsed_csv_exists": parsed, "job_name": job_name})
        print(f"{suffix}: {state} | parsed={parsed}")

    status_table = PARTS_DIR / "chunk_status_summary.csv"
    pd.DataFrame(rows).to_csv(status_table, index=False, encoding="utf-8-sig")
    print(f"\nWrote {status_table}")


def wait(args: argparse.Namespace) -> None:
    apply_suffix_paths(args)
    client = get_client()
    job_name = read_job_name(args.job_file)
    while True:
        batch_job = client.batches.get(name=job_name)
        state = get_state_name(batch_job)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {job_name}: {state}")
        args.status_file.parent.mkdir(parents=True, exist_ok=True)
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
    apply_suffix_paths(args)
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
    if isinstance(content, bytes):
        text = content.decode("utf-8")
    else:
        text = str(content)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(text, encoding="utf-8")
    print(f"Wrote results to {args.results}")


# ---------- result parsing ----------
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


def parse_model_json(text: str) -> tuple[str, float]:
    text = strip_json_fence(text)
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("model output is not a JSON object")
    topic = str(obj.get("topic", "")).strip().strip('"').strip("'").rstrip(".").strip()
    if not topic:
        topic = "unclear"
    raw_score = obj.get("procedural_score", obj.get("procedural_rhetorical_score"))
    score = float(raw_score)
    score = round(max(0.0, min(1.0, score)), 3)
    return topic, score


def parse_result_line(line: str) -> dict[str, Any]:
    raw = json.loads(line)
    key = raw.get("key") or raw.get("metadata", {}).get("key") or raw.get("custom_id")
    response = raw.get("response") or raw.get("inlineResponse", {}).get("response")
    error = raw.get("error") or raw.get("status")
    if not key:
        raise ValueError(f"No key found in result line: {raw}")
    basepk = int(str(key))
    if error and not response:
        return {"basepk": basepk, "topic": "", "procedural_score": "", "error": json.dumps(error, ensure_ascii=False)[:300]}
    text = extract_text_from_response(response)
    if not text:
        return {"basepk": basepk, "topic": "", "procedural_score": "", "error": "empty response text"}
    try:
        topic, score = parse_model_json(text)
        return {"basepk": basepk, "topic": topic, "procedural_score": score, "error": ""}
    except Exception as exc:
        return {"basepk": basepk, "topic": "", "procedural_score": "", "error": f"ParseError: {exc}: {text[:200]}"}


def parse(args: argparse.Namespace) -> None:
    apply_suffix_paths(args)
    if not args.results.exists():
        sys.exit(f"Results file not found: {args.results}. Run download first.")
    rows = []
    with open(args.results, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(parse_result_line(line))
            except Exception as exc:
                rows.append({"basepk": "", "topic": "", "procedural_score": "", "error": f"Line {line_no}: {exc}"})

    rows = sorted(rows, key=lambda r: (r["basepk"] == "", int(r["basepk"]) if r["basepk"] != "" else 10**18))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    n_error = sum(1 for r in rows if r.get("error"))
    print(f"Wrote {len(rows):,} rows to {args.output}; errors: {n_error:,}")


def merge_parts(args: argparse.Namespace) -> None:
    if not PARTS_DIR.exists():
        sys.exit(f"Parts folder not found: {PARTS_DIR}. Run make-chunks first.")

    if MANIFEST_PATH.exists():
        manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig")
        files = [Path(p) for p in manifest["output"].tolist() if Path(p).exists()]
    else:
        files = sorted(PARTS_DIR.glob("speeches_topics_procedural_part*.csv"))

    if not files:
        sys.exit("No part output CSV files found. Run parse --suffix partXXX first.")

    frames = []
    for file in files:
        df = pd.read_csv(file, encoding="utf-8-sig")
        if BASEPK_COLUMN not in df.columns:
            print(f"Skipping {file}: no basepk column")
            continue
        frames.append(df)

    if not frames:
        sys.exit("No valid part CSV files found.")

    out = pd.concat(frames, ignore_index=True)
    before = len(out)
    out = out.drop_duplicates(subset=[BASEPK_COLUMN], keep="last")
    out = out.sort_values(BASEPK_COLUMN).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")

    n_error = int(out["error"].fillna("").astype(str).str.len().gt(0).sum()) if "error" in out.columns else 0
    print(f"Merged {len(files):,} part files")
    print(f"Rows before dedup: {before:,}; after dedup: {len(out):,}; errors: {n_error:,}")
    print(f"Wrote final output: {args.output}")


def cancel(args: argparse.Namespace) -> None:
    apply_suffix_paths(args)
    client = get_client()
    job_name = read_job_name(args.job_file)
    client.batches.cancel(name=job_name)
    print(f"Cancel requested for {job_name}")


# ---------- CLI ----------
def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suffix", default=None, help="chunk suffix, for example part001")
    parser.add_argument("--jsonl", type=Path, default=REQUEST_JSONL_PATH)
    parser.add_argument("--job-file", type=Path, default=JOB_INFO_PATH)
    parser.add_argument("--status-file", type=Path, default=STATUS_JSON_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_JSONL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch workflow for topic + procedural_score classification")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-jsonl", help="create one JSONL batch input file")
    p.add_argument("--input", type=Path, default=INPUT_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--jsonl", type=Path, default=REQUEST_JSONL_PATH)
    p.add_argument("--suffix", default=None, help="chunk suffix, for example part001")
    p.add_argument("--start", type=int, default=0, help="start row offset after cleaning/deduplication")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=make_jsonl)

    p = sub.add_parser("make-chunks", help="split input into many JSONL files; no API call and no cost")
    p.add_argument("--input", type=Path, default=INPUT_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="global row limit before chunking")
    p.add_argument("--max-chunks", type=int, default=None, help="write only the first N chunks for testing")
    p.add_argument("--overwrite", action="store_true", help="rewrite existing JSONL chunk files")
    p.add_argument("--include-done", action="store_true", help="do not skip basepks already in final output")
    p.set_defaults(func=make_chunks)

    p = sub.add_parser("submit", help="upload JSONL and create batch job")
    add_common_paths(p)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--display-name", default="topics-procedural-score")
    p.add_argument("--force", action="store_true", help="allow creating another paid job even if job info exists")
    p.set_defaults(func=submit)

    p = sub.add_parser("submit-next", help="submit the first chunk without a job file in the manifest")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--display-name", default="topics-procedural-score")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=submit_next)

    p = sub.add_parser("run-chunks", help="automatically submit, wait, download, and parse chunks sequentially")
    p.add_argument("--from-suffix", default=None, help="first chunk to run, e.g. part001")
    p.add_argument("--to-suffix", default=None, help="last chunk to run, e.g. part061")
    p.add_argument("--max-chunks", type=int, default=None, help="run only the first N selected chunks")
    p.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    p.add_argument("--submit-retry-seconds", type=int, default=600, help="seconds to wait before retrying submit after quota/rate errors")
    p.add_argument("--max-submit-retries", type=int, default=24)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--display-name", default="topics-procedural-score")
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--redownload", action="store_true", help="download result JSONL again even if it already exists")
    p.add_argument("--reparse", action="store_true", help="parse again even if part CSV already exists")
    p.add_argument("--no-merge", dest="merge", action="store_false", help="do not merge part CSVs at the end")
    p.add_argument("--continue-on-failure", action="store_true", help="continue with later chunks if one chunk fails")
    p.set_defaults(func=run_chunks, merge=True)

    p = sub.add_parser("status", help="check batch job status")
    add_common_paths(p)
    p.set_defaults(func=status)

    p = sub.add_parser("status-all", help="check all submitted chunks in the manifest")
    p.set_defaults(func=status_all)

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

    p = sub.add_parser("merge-parts", help="merge parsed chunk CSV files into final output")
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.set_defaults(func=merge_parts)

    p = sub.add_parser("cancel", help="cancel the current batch job")
    add_common_paths(p)
    p.set_defaults(func=cancel)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()







