"""Merge all LIPAD speech files with Parliamentarians.xlsx."""

from datetime import datetime
from pathlib import Path
import re
import unicodedata

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
LIPAD_DIR = PROJECT_DIR / "input" / "lipad"
PARLIAMENTARIANS_FILE = PROJECT_DIR / "input" / "Parliamentarians.xlsx"
OUTPUT_FILE = PROJECT_DIR / "output" / "Speech_merged.csv"
TEMP_OUTPUT_FILE = PROJECT_DIR / "output" / "Speech_merged.tmp.csv"

LIPAD_COLUMNS = [
    "basepk",
    "speechdate",
    "speechtext",
    "speakername",
    "speakerparty",
    "speakerposition",
    "maintopic",
]


def normalize_name(value):
    """Normalize names while keeping a readable, explicit matching rule."""
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower().replace("?", "'").replace(":", " ")

    titles = [
        r"\bthe\s+right\s+honourable\b",
        r"\bright\s+honourable\b",
        r"\bthe\s+honourable\b",
        r"\bhonourable\b",
        r"\brt\.?\s*hon\.?\b",
        r"\bhon\.?\b",
        r"\bmr\.?\b",
        r"\bmrs\.?\b",
        r"\bms\.?\b",
        r"\bmiss\b",
        r"\bdr\.?\b",
        r"\bsir\b",
        r"\bmadam\b",
        r"\bmme\.?\b",
        r"\bm\.?\b",
    ]
    for title in titles:
        text = re.sub(title, " ", text)

    text = re.sub(r"[^a-z,\s\-']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_last_name(value):
    """Convert 'Last, First Middle' to 'First Middle Last'."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if "," not in text:
        return text

    last_name, given_names = text.split(",", 1)
    return f"{given_names.strip()} {last_name.strip()}"


def last_name(value):
    """Extract a normalized last name."""
    normalized = normalize_name(value).replace(",", " ")
    words = normalized.split()
    return words[-1] if words else ""


def parse_intervals(value):
    """Parse lines such as 'MP (1963/01/01 - 1968/06/24)'."""
    if pd.isna(value):
        return []

    pattern = re.compile(
        r"^(?P<value>.*?)\s*"
        r"\((?P<start>\d{4}/\d{2}/\d{2})\s*-\s*"
        r"(?P<end>\d{4}/\d{2}/\d{2})\)\s*$"
    )
    intervals = []

    for line in str(value).replace("\r", "\n").split("\n"):
        match = pattern.match(line.strip())
        if match:
            intervals.append(
                (
                    match.group("value").strip(),
                    pd.Timestamp(match.group("start")),
                    pd.Timestamp(match.group("end")),
                )
            )

    return intervals


def build_parliamentarian_indexes(parliamentarians):
    """Build small lookup tables used while reading the daily speech files."""
    exact_mp_index = {}
    last_name_mp_index = {}
    party_index = {}
    riding_index = {}
    person_details = {}

    for person_id, (_, row) in enumerate(parliamentarians.iterrows(), start=1):
        name = row["Name"]
        exact_keys = {
            normalize_name(name),
            normalize_name(first_last_name(name)),
        }
        family_name = last_name(name.split(",", 1)[0] if "," in str(name) else name)

        for parliamentarian_type, start_date, end_date in parse_intervals(
            row["Type of Parliamentarian"]
        ):
            if parliamentarian_type != "MP":
                continue

            interval = (person_id, start_date, end_date)
            for key in exact_keys:
                if key:
                    exact_mp_index.setdefault(key, []).append(interval)
            if family_name:
                last_name_mp_index.setdefault(family_name, []).append(interval)

        for party, start_date, end_date in parse_intervals(row["Political Affiliation"]):
            party_index.setdefault(person_id, []).append((start_date, end_date, party))

        riding_intervals = parse_intervals(row["Riding/Senatorial Division"])
        provinces = [
            line.strip()
            for line in str(row["Province/Territory"]).replace("\r", "\n").split("\n")
            if line.strip() and line.strip().lower() != "nan"
        ]

        for position, (riding, start_date, end_date) in enumerate(riding_intervals):
            province = None
            if len(provinces) == len(riding_intervals):
                province = provinces[position]
            elif len(provinces) == 1:
                province = provinces[0]

            riding_index.setdefault(person_id, []).append(
                (start_date, end_date, riding, province)
            )

        person_details[person_id] = {
            "parliamentarian_name": name,
            "gender": row["Gender"],
            "date_of_birth": row["Date of Birth"],
            "date_of_death": row["Date of Death"],
        }

    return exact_mp_index, last_name_mp_index, party_index, riding_index, person_details


def active_person(candidates, speech_date):
    """Return a person only when exactly one MP candidate is active."""
    active_ids = {
        person_id
        for person_id, start_date, end_date in candidates
        if start_date <= speech_date <= end_date
    }
    return next(iter(active_ids)) if len(active_ids) == 1 else None


def match_person(name, speech_date, exact_mp_index, last_name_mp_index):
    """Match by full name, then by a unique active MP with the same last name."""
    name_key = normalize_name(name)
    person_id = active_person(exact_mp_index.get(name_key, []), speech_date)

    if person_id is not None:
        return person_id

    family_name = last_name(name)
    return active_person(last_name_mp_index.get(family_name, []), speech_date)


def value_at_date(intervals, speech_date, value_position):
    """Return the value from the interval active on the speech date."""
    matching = [
        interval
        for interval in intervals
        if interval[0] <= speech_date <= interval[1]
    ]
    if not matching:
        return None

    latest = max(matching, key=lambda interval: interval[0])
    return latest[value_position]


def value_for_person(index, person_id, speech_date, value_position):
    """Return dated metadata for a matched person; keep unmatched rows empty."""
    if pd.isna(person_id):
        return None
    return value_at_date(
        index.get(int(person_id), []),
        speech_date,
        value_position,
    )


def process_lipad_file(
    file_path,
    exact_mp_index,
    last_name_mp_index,
    party_index,
    riding_index,
    person_details,
):
    """Read one daily Lipad file and keep only speeches matched to an MP."""
    speeches = pd.read_csv(
        file_path,
        usecols=LIPAD_COLUMNS,
        dtype={"basepk": "string"},
        low_memory=False,
    )
    # Match the old preparation logic: remove missing speaker/text rows and
    # deduplicate identical raw speeches within each daily Lipad file.
    speeches = speeches.dropna(subset=["speakername", "speechtext"]).copy()
    speeches = speeches.drop_duplicates(subset=["speechtext"], keep="first")
    speeches["_speech_date"] = pd.to_datetime(speeches["speechdate"], errors="coerce")
    speeches = speeches.dropna(subset=["_speech_date"]).copy()

    speeches["person_id"] = [
        match_person(
            name,
            speech_date,
            exact_mp_index,
            last_name_mp_index,
        )
        for name, speech_date in zip(speeches["speakername"], speeches["_speech_date"])
    ]
    if speeches.empty:
        return speeches

    speeches["is_mp_matched"] = speeches["person_id"].notna()
    speeches = speeches.loc[speeches["is_mp_matched"]].copy()

    if speeches.empty:
        return speeches

    speeches["person_id"] = pd.to_numeric(
        speeches["person_id"], errors="coerce"
    ).astype("Int64")
    speeches["year"] = speeches["_speech_date"].dt.year.astype(int)

    details = speeches["person_id"].map(person_details)
    speeches["parliamentarian_name"] = details.map(
        lambda item: item["parliamentarian_name"] if isinstance(item, dict) else None
    )
    speeches["gender"] = details.map(
        lambda item: item["gender"] if isinstance(item, dict) else None
    )
    speeches["date_of_birth"] = details.map(
        lambda item: item["date_of_birth"] if isinstance(item, dict) else None
    )
    speeches["date_of_death"] = details.map(
        lambda item: item["date_of_death"] if isinstance(item, dict) else None
    )

    speeches["party"] = [
        value_for_person(party_index, person_id, speech_date, 2)
        for person_id, speech_date in zip(speeches["person_id"], speeches["_speech_date"])
    ]
    speeches["riding"] = [
        value_for_person(riding_index, person_id, speech_date, 2)
        for person_id, speech_date in zip(speeches["person_id"], speeches["_speech_date"])
    ]
    speeches["province"] = [
        value_for_person(riding_index, person_id, speech_date, 3)
        for person_id, speech_date in zip(speeches["person_id"], speeches["_speech_date"])
    ]

    output_columns = [
        "basepk",
        "speechdate",
        "speechtext",
        "speakername",
        "speakerparty",
        "speakerposition",
        "maintopic",
        "year",
        "person_id",
        "is_mp_matched",
        "parliamentarian_name",
        "gender",
        "date_of_birth",
        "date_of_death",
        "party",
        "riding",
        "province",
    ]
    return speeches[output_columns]


def file_date(file_path):
    return datetime.strptime(file_path.stem, "%Y-%m-%d")


def main():
    parliamentarians = pd.read_excel(PARLIAMENTARIANS_FILE, sheet_name="Parliamentarians")
    indexes = build_parliamentarian_indexes(parliamentarians)

    lipad_files = sorted(LIPAD_DIR.rglob("*.csv"), key=file_date)
    if not lipad_files:
        raise FileNotFoundError(f"No CSV files found in {LIPAD_DIR}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if TEMP_OUTPUT_FILE.exists():
        TEMP_OUTPUT_FILE.unlink()

    input_rows = 0
    output_rows = 0
    matched_mp_rows = 0
    first_write = True

    for file_number, file_path in enumerate(lipad_files, start=1):
        daily_row_count = sum(1 for _ in open(file_path, encoding="utf-8")) - 1
        input_rows += max(daily_row_count, 0)

        matched = process_lipad_file(file_path, *indexes)
        if not matched.empty:
            matched.to_csv(
                TEMP_OUTPUT_FILE,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
                encoding="utf-8-sig" if first_write else "utf-8",
            )
            output_rows += len(matched)
            matched_mp_rows += int(matched["is_mp_matched"].sum())
            first_write = False

        if file_number % 250 == 0 or file_number == len(lipad_files):
            print(
                f"Processed {file_number}/{len(lipad_files)} files; "
                f"output rows: {output_rows:,}; matched MPs: {matched_mp_rows:,}"
            )

    if first_write:
        raise ValueError("No speeches remained after the initial preparation rules")

    TEMP_OUTPUT_FILE.replace(OUTPUT_FILE)
    print(f"Raw input rows: {input_rows:,}")
    print(f"Rows after missing speaker/text, duplicate-text, and date checks: {output_rows:,}")
    print(f"Matched MP rows: {matched_mp_rows:,}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
