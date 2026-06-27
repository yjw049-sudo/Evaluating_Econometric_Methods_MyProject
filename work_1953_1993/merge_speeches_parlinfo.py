# -*- coding: utf-8 -*-
"""
Merge Canadian parliamentary speech data with ParlInfo parliamentarian metadata.

Core logic:
1. Parse ParlInfo cells like "MP (1953/08/10 - 1954/06/30)" into date intervals.
2. Match speeches to people by speaker name + speech date.
3. Attach party/riding/province/gender at the date of each speech.

Author: generated for research workflow
"""

# =========================
# 0. KEY PARAMETERS
# =========================

from pathlib import Path

# Project paths
WORK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORK_DIR / "output"
INPUT_DIR = WORK_DIR / "input"

# Input files
SPEECH_FILE = OUTPUT_DIR / "processed_lipad_1963_1993_filtered.csv"
PARLINFO_FILE = INPUT_DIR / "Parliamentarians.xlsx"

# Output files
OUTPUT_MERGED_FILE = OUTPUT_DIR / "speeches_with_parlinfo.csv"
OUTPUT_UNMATCHED_FILE = OUTPUT_DIR / "unmatched_speeches.csv"
OUTPUT_AMBIGUOUS_FILE = OUTPUT_DIR / "ambiguous_lastname_matches.csv"
OUTPUT_PERSON_PERIOD_FILE = OUTPUT_DIR / "parlinfo_person_period_debug.csv"

# Create the output directory if it does not already exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Speech file settings
SPEECH_FILE_TYPE = "csv"       # "csv" or "xlsx"
SPEECH_CSV_SEP = ";"           # your sample uses semicolon
SPEECH_CSV_ENCODING = "utf-8-sig"

# Column names in speech data
SPEECH_ID_COL = "basepk"
SPEAKER_NAME_COL = "speakername"
SPEECH_DATE_COL = "speechdate"
SPEECH_TEXT_COL = "speechtext"  # optional, used only for speech length if available

# Column names in ParlInfo data
PARLINFO_SHEET_NAME = 0  # 0 = first sheet; or use "Parliamentarians"
PARL_NAME_COL = "Name"
PARL_TYPE_COL = "Type of Parliamentarian"
PARL_HOC_COL = "House of Commons"
PARL_RIDING_COL = "Riding/Senatorial Division"
PARL_PROVINCE_COL = "Province/Territory"
PARL_GENDER_COL = "Gender"
PARL_PARTY_COL = "Political Affiliation"
PARL_DOB_COL = "Date of Birth"
PARL_DOD_COL = "Date of Death"

# Research period filter. Set to None if you do not want to filter.
# Example for 1953-1993: "1953-01-01", "1993-12-31"
START_DATE = None
END_DATE = None

# Matching behavior
ALLOW_LASTNAME_FALLBACK = True
# If speaker is "Mr. Argue" and only one active MP named Argue exists on that date, match it.
# If more than one active MP has the same last name on that date, leave it unmatched and save candidates.

# Drop non-House-of-Commons speakers after matching?
# In this script, matching is already based on MP intervals, so Senators will not match unless also MPs.
KEEP_ONLY_MATCHED_MP = False

# =========================
# 1. IMPORTS
# =========================

import re
import unicodedata
import pandas as pd
import numpy as np

# =========================
# 2. HELPER FUNCTIONS
# =========================

def read_speech_file(path):
    if SPEECH_FILE_TYPE.lower() == "csv":
        return pd.read_csv(path, sep=SPEECH_CSV_SEP, encoding=SPEECH_CSV_ENCODING)
    elif SPEECH_FILE_TYPE.lower() in {"xlsx", "excel"}:
        return pd.read_excel(path)
    else:
        raise ValueError("SPEECH_FILE_TYPE must be 'csv' or 'xlsx'.")


def strip_accents(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_name(name):
    """Normalize names for matching."""
    if pd.isna(name):
        return ""
    x = strip_accents(str(name)).lower().strip()
    x = x.replace(":", " ")
    x = x.replace("’", "'")

    # remove common parliamentary titles/honorifics
    title_patterns = [
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
        r"\bm\.?\b",
        r"\bmme\.?\b",
    ]
    for pat in title_patterns:
        x = re.sub(pat, " ", x)

    # keep letters, comma, apostrophe, hyphen and spaces
    x = re.sub(r"[^a-z,\s\-']", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def comma_to_firstlast(name):
    """Convert 'Abbott, Douglas Charles' -> 'Douglas Charles Abbott'."""
    if pd.isna(name):
        return ""
    s = str(name).strip()
    if "," not in s:
        return s
    last, first = s.split(",", 1)
    return f"{first.strip()} {last.strip()}"


def extract_last_name_from_parlinfo(name):
    if pd.isna(name):
        return ""
    s = strip_accents(str(name)).strip()
    if "," in s:
        last = s.split(",", 1)[0]
    else:
        last = s.split()[-1] if s.split() else ""
    return normalize_name(last).replace(",", "")


def extract_last_name_from_speech(name):
    """Extract likely last name from speech speaker name."""
    x = normalize_name(name).replace(",", " ")
    if not x:
        return ""
    tokens = x.split()
    if not tokens:
        return ""
    return tokens[-1]


def parse_interval_lines(cell):
    """
    Parse lines of the form:
        MP (1953/08/10 - 1954/06/30)
        Liberal Party of Canada (1940/03/26 - 1954/06/30)
        Conservative (1867-1942) (1940/03/26 - 1942/12/10)
    Returns list of dicts: value, start_date, end_date.
    """
    rows = []
    if pd.isna(cell):
        return rows

    text = str(cell).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    # Match the final date interval in parentheses.
    pattern = re.compile(
        r"^(?P<value>.*?)\s*\((?P<start>\d{4}/\d{2}/\d{2})\s*-\s*(?P<end>\d{4}/\d{2}/\d{2})\)\s*$"
    )

    for line in lines:
        m = pattern.match(line)
        if not m:
            continue
        rows.append({
            "value": m.group("value").strip(),
            "start_date": pd.to_datetime(m.group("start"), errors="coerce"),
            "end_date": pd.to_datetime(m.group("end"), errors="coerce"),
            "raw_line": line,
        })
    return rows


def split_plain_lines(cell):
    if pd.isna(cell):
        return []
    text = str(cell).replace("\r\n", "\n").replace("\r", "\n")
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def build_interval_table(parl, source_col, variable_name):
    """Build long interval table from a ParlInfo column containing value(date-date) lines."""
    out = []
    for _, row in parl.iterrows():
        intervals = parse_interval_lines(row.get(source_col, np.nan))
        for item in intervals:
            out.append({
                "person_id": row["person_id"],
                "parlinfo_name": row[PARL_NAME_COL],
                "variable": variable_name,
                "value": item["value"],
                "start_date": item["start_date"],
                "end_date": item["end_date"],
                "raw_line": item["raw_line"],
            })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.dropna(subset=["start_date", "end_date"])
    return df


def build_riding_interval_table(parl):
    """
    Build riding interval table and attach province/territory where possible.
    Province/Territory column often contains one line per riding interval, without dates.
    """
    out = []
    for _, row in parl.iterrows():
        intervals = parse_interval_lines(row.get(PARL_RIDING_COL, np.nan))
        provinces = split_plain_lines(row.get(PARL_PROVINCE_COL, np.nan))

        for i, item in enumerate(intervals):
            if len(provinces) == len(intervals):
                province = provinces[i]
            elif len(provinces) == 1:
                province = provinces[0]
            else:
                province = np.nan

            out.append({
                "person_id": row["person_id"],
                "parlinfo_name": row[PARL_NAME_COL],
                "riding_at_date": item["value"],
                "province_at_date": province,
                "start_date": item["start_date"],
                "end_date": item["end_date"],
                "raw_line": item["raw_line"],
            })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.dropna(subset=["start_date", "end_date"])
    return df


def interval_join_by_person(speeches_with_person, interval_df, value_col, output_col):
    """
    Attach interval value by person_id and speech date.
    speeches_with_person must contain person_id and speech_date.
    interval_df must contain person_id, start_date, end_date, and value_col.
    """
    if interval_df.empty:
        speeches_with_person[output_col] = np.nan
        return speeches_with_person

    base_cols = [SPEECH_ID_COL, "person_id", "speech_date"]
    tmp = speeches_with_person[base_cols].merge(
        interval_df[["person_id", value_col, "start_date", "end_date"]],
        on="person_id",
        how="left"
    )
    tmp = tmp[
        (tmp["speech_date"] >= tmp["start_date"]) &
        (tmp["speech_date"] <= tmp["end_date"])
    ].copy()

    # If multiple intervals match, keep the interval with the latest start date.
    tmp = tmp.sort_values([SPEECH_ID_COL, "start_date"])
    tmp = tmp.drop_duplicates(subset=[SPEECH_ID_COL], keep="last")
    tmp = tmp[[SPEECH_ID_COL, value_col]].rename(columns={value_col: output_col})

    result = speeches_with_person.merge(tmp, on=SPEECH_ID_COL, how="left")
    return result

# =========================
# 3. LOAD DATA
# =========================

speeches = read_speech_file(SPEECH_FILE)
parl = pd.read_excel(PARLINFO_FILE, sheet_name=PARLINFO_SHEET_NAME)

# Ensure speech id exists
if SPEECH_ID_COL not in speeches.columns:
    speeches[SPEECH_ID_COL] = range(1, len(speeches) + 1)

# Standardize dates
speeches["speech_date"] = pd.to_datetime(speeches[SPEECH_DATE_COL], errors="coerce")

if START_DATE is not None:
    speeches = speeches[speeches["speech_date"] >= pd.to_datetime(START_DATE)].copy()
if END_DATE is not None:
    speeches = speeches[speeches["speech_date"] <= pd.to_datetime(END_DATE)].copy()

# Basic speech length if text exists
if SPEECH_TEXT_COL in speeches.columns:
    speeches["speech_length_words"] = speeches[SPEECH_TEXT_COL].fillna("").astype(str).str.split().str.len()

# Create person id
parl = parl.copy()
parl["person_id"] = range(1, len(parl) + 1)

# Normalize ParlInfo names
parl["name_key_comma"] = parl[PARL_NAME_COL].apply(normalize_name)
parl["name_firstlast"] = parl[PARL_NAME_COL].apply(comma_to_firstlast)
parl["name_key_firstlast"] = parl["name_firstlast"].apply(normalize_name)
parl["last_name_key"] = parl[PARL_NAME_COL].apply(extract_last_name_from_parlinfo)

# Normalize speech names
speeches["speaker_name_key"] = speeches[SPEAKER_NAME_COL].apply(normalize_name)
speeches["speaker_last_name_key"] = speeches[SPEAKER_NAME_COL].apply(extract_last_name_from_speech)

# =========================
# 4. BUILD PARLINFO INTERVAL TABLES
# =========================

type_intervals = build_interval_table(parl, PARL_TYPE_COL, "parliamentarian_type")
party_intervals = build_interval_table(parl, PARL_PARTY_COL, "party")
riding_intervals = build_riding_interval_table(parl)

# MP intervals only. This excludes Senators unless they also have MP intervals.
mp_intervals = type_intervals[type_intervals["value"].eq("MP")].copy()
mp_intervals = mp_intervals.merge(
    parl[["person_id", PARL_NAME_COL, "name_key_comma", "name_key_firstlast", "last_name_key",
          PARL_GENDER_COL, PARL_DOB_COL, PARL_DOD_COL]],
    on="person_id",
    how="left",
    suffixes=("", "_person")
)

# Debug person-period file
person_period_debug = pd.concat([
    type_intervals.assign(source="type"),
    party_intervals.assign(source="party")
], ignore_index=True)
person_period_debug.to_csv(OUTPUT_PERSON_PERIOD_FILE, index=False, encoding="utf-8-sig")

# =========================
# 5. MATCH SPEECHES TO MP PERSON_ID
# =========================

# 5A. Exact/full-name matching using both ParlInfo name formats.
mp_keys_a = mp_intervals.copy()
mp_keys_a["match_key"] = mp_keys_a["name_key_firstlast"]
mp_keys_b = mp_intervals.copy()
mp_keys_b["match_key"] = mp_keys_b["name_key_comma"]
mp_keys = pd.concat([mp_keys_a, mp_keys_b], ignore_index=True)
mp_keys = mp_keys.drop_duplicates(subset=["person_id", "match_key", "start_date", "end_date"])

exact_candidates = speeches.merge(
    mp_keys[["person_id", PARL_NAME_COL, "match_key", "start_date", "end_date"]],
    left_on="speaker_name_key",
    right_on="match_key",
    how="left"
)
exact_candidates = exact_candidates[
    (exact_candidates["speech_date"] >= exact_candidates["start_date"]) &
    (exact_candidates["speech_date"] <= exact_candidates["end_date"])
].copy()

# Keep the latest active MP interval if duplicate
exact_candidates = exact_candidates.sort_values([SPEECH_ID_COL, "start_date"])
exact_match = exact_candidates.drop_duplicates(subset=[SPEECH_ID_COL], keep="last")
exact_match = exact_match[[SPEECH_ID_COL, "person_id", PARL_NAME_COL, "start_date", "end_date"]].rename(
    columns={PARL_NAME_COL: "matched_parlinfo_name", "start_date": "mp_start_date", "end_date": "mp_end_date"}
)
exact_match["match_method"] = "exact_full_name"

matched = speeches.merge(exact_match, on=SPEECH_ID_COL, how="left")

# 5B. Last-name fallback for unmatched rows.
ambiguous_candidates_out = pd.DataFrame()
if ALLOW_LASTNAME_FALLBACK:
    unmatched = matched[matched["person_id"].isna()].copy()

    last_candidates = unmatched.merge(
        mp_intervals[["person_id", PARL_NAME_COL, "last_name_key", "start_date", "end_date"]],
        left_on="speaker_last_name_key",
        right_on="last_name_key",
        how="left"
    )
    last_candidates = last_candidates[
        (last_candidates["speech_date"] >= last_candidates["start_date"]) &
        (last_candidates["speech_date"] <= last_candidates["end_date"])
    ].copy()

    if not last_candidates.empty:
        # Count unique active people per speech.
        counts = last_candidates.groupby(SPEECH_ID_COL)["person_id_y"].nunique().reset_index(name="n_candidates")
        last_candidates = last_candidates.merge(counts, on=SPEECH_ID_COL, how="left")

        unique_last = last_candidates[last_candidates["n_candidates"].eq(1)].copy()
        unique_last = unique_last.sort_values([SPEECH_ID_COL, "start_date"])
        unique_last = unique_last.drop_duplicates(subset=[SPEECH_ID_COL], keep="last")
        unique_last = unique_last[[SPEECH_ID_COL, "person_id_y", PARL_NAME_COL, "start_date", "end_date"]].rename(
            columns={
                "person_id_y": "person_id_fallback",
                PARL_NAME_COL: "matched_parlinfo_name_fallback",
                "start_date": "mp_start_date_fallback",
                "end_date": "mp_end_date_fallback",
            }
        )

        ambiguous_candidates_out = last_candidates[last_candidates["n_candidates"].gt(1)].copy()
        if not ambiguous_candidates_out.empty:
            keep_cols = [SPEECH_ID_COL, SPEAKER_NAME_COL, SPEECH_DATE_COL, "speech_date",
                         "speaker_last_name_key", "person_id_y", PARL_NAME_COL, "start_date", "end_date", "n_candidates"]
            keep_cols = [c for c in keep_cols if c in ambiguous_candidates_out.columns]
            ambiguous_candidates_out[keep_cols].to_csv(OUTPUT_AMBIGUOUS_FILE, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame().to_csv(OUTPUT_AMBIGUOUS_FILE, index=False, encoding="utf-8-sig")

        matched = matched.merge(unique_last, on=SPEECH_ID_COL, how="left")

        # Fill unmatched person_id with fallback person_id.
        fill_mask = matched["person_id"].isna() & matched["person_id_fallback"].notna()
        matched.loc[fill_mask, "person_id"] = matched.loc[fill_mask, "person_id_fallback"]
        matched.loc[fill_mask, "matched_parlinfo_name"] = matched.loc[fill_mask, "matched_parlinfo_name_fallback"]
        matched.loc[fill_mask, "mp_start_date"] = matched.loc[fill_mask, "mp_start_date_fallback"]
        matched.loc[fill_mask, "mp_end_date"] = matched.loc[fill_mask, "mp_end_date_fallback"]
        matched.loc[fill_mask, "match_method"] = "unique_last_name_active_on_date"

        # Drop fallback helper columns
        drop_cols = [c for c in matched.columns if c.endswith("_fallback")]
        matched = matched.drop(columns=drop_cols)
    else:
        pd.DataFrame().to_csv(OUTPUT_AMBIGUOUS_FILE, index=False, encoding="utf-8-sig")

# =========================
# 6. ATTACH STATIC PERSON ATTRIBUTES
# =========================

static_cols = ["person_id", PARL_GENDER_COL, PARL_DOB_COL, PARL_DOD_COL]
static_cols = [c for c in static_cols if c in parl.columns]
matched = matched.merge(parl[static_cols], on="person_id", how="left")

matched = matched.rename(columns={
    PARL_GENDER_COL: "gender",
    PARL_DOB_COL: "date_of_birth",
    PARL_DOD_COL: "date_of_death",
})

matched["is_mp_matched"] = matched["person_id"].notna()

# =========================
# 7. ATTACH PARTY/RIDING/PROVINCE AT SPEECH DATE
# =========================

# Party interval table
party_for_join = party_intervals.rename(columns={"value": "party_value"})
matched = interval_join_by_person(matched, party_for_join, "party_value", "party_at_date")

# Riding/province interval table
matched = interval_join_by_person(matched, riding_intervals, "riding_at_date", "riding_at_date")
matched = interval_join_by_person(matched, riding_intervals, "province_at_date", "province_at_date")

# =========================
# 8. OPTIONAL SIMPLE PARTY STANDARDIZATION
# =========================

def simplify_party(party):
    if pd.isna(party):
        return np.nan
    x = str(party).lower()
    if "liberal" in x:
        return "Liberal"
    if "progressive conservative" in x or "conservative" in x:
        return "Conservative/Progressive Conservative"
    if "new democratic" in x or "n.d.p" in x or "ndp" in x:
        return "NDP"
    if "bloc" in x:
        return "Bloc Québécois"
    if "reform" in x:
        return "Reform"
    if "social credit" in x:
        return "Social Credit"
    if "c.c.f" in x or "co-operative commonwealth" in x:
        return "CCF"
    if "independent" in x:
        return "Independent"
    return str(party)

matched["party_simplified"] = matched["party_at_date"].apply(simplify_party)

# =========================
# 9. OPTIONAL: FILTER ONLY MATCHED MPs
# =========================

if KEEP_ONLY_MATCHED_MP:
    matched = matched[matched["is_mp_matched"]].copy()

# =========================
# 10. DIAGNOSTICS AND OUTPUT
# =========================

# Save unmatched speeches
unmatched_out = matched[~matched["is_mp_matched"]].copy()
unmatched_out.to_csv(OUTPUT_UNMATCHED_FILE, index=False, encoding="utf-8-sig")

# Save merged data
matched.to_csv(OUTPUT_MERGED_FILE, index=False, encoding="utf-8-sig")

# Print diagnostics
n_total = len(matched)
n_mp = int(matched["is_mp_matched"].sum())
n_party = int(matched["party_at_date"].notna().sum())
n_riding = int(matched["riding_at_date"].notna().sum())

print("================ MERGE DIAGNOSTICS ================")
print(f"Total speeches:          {n_total}")
print(f"Matched to MP:           {n_mp} ({n_mp / n_total:.1%})" if n_total else "Matched to MP: 0")
print(f"Matched party_at_date:   {n_party} ({n_party / n_total:.1%})" if n_total else "Matched party_at_date: 0")
print(f"Matched riding_at_date:  {n_riding} ({n_riding / n_total:.1%})" if n_total else "Matched riding_at_date: 0")
print("\nMatch methods:")
print(matched["match_method"].fillna("unmatched").value_counts(dropna=False).to_string())
print("\nOutput files:")
print(f"Merged data:             {OUTPUT_MERGED_FILE}")
print(f"Unmatched speeches:      {OUTPUT_UNMATCHED_FILE}")
print(f"Ambiguous candidates:    {OUTPUT_AMBIGUOUS_FILE}")
print(f"Person-period debug:     {OUTPUT_PERSON_PERIOD_FILE}")

# Show a compact preview
preview_cols = [SPEECH_ID_COL, SPEAKER_NAME_COL, SPEECH_DATE_COL, "matched_parlinfo_name",
                "match_method", "party_at_date", "riding_at_date", "province_at_date", "gender"]
preview_cols = [c for c in preview_cols if c in matched.columns]
print("\nPreview:")
print(matched[preview_cols].head(20).to_string(index=False))
