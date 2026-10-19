import re
import json
import warnings
import numpy as np
import pandas as pd
from paths import tokens_file
from collections import Counter

TOKENS = json.loads(tokens_file.read_text(encoding="utf-8"))
NUMERIC_TYPES = {"numeric_continuous", "numeric_discrete"}
CATEGORY_TYPES = {"categorical_nominal", "categorical_ordinal"}
PARSE_SAMPLE_ROWS = 5000
DISCRETE_MAX_UNIQUE = 30
NUMBER_SHARE = 0.95
DATE_SHARE = 0.9
FIXED_LENGTH_SHARE = 0.99
ID_MIN_ROWS = 50
ID_MIN_UNIQUE_SHARE = 0.99
ID_MIN_LENGTH = 4
TEXT_MIN_WORDS = 4
TEXT_MIN_UNIQUE_SHARE = 0.5
# date formats tried in order when a column is parsed: ISO, compact, then mixed
DATE_ORDER = ["ISO8601", "%Y%m%dT%H%M%S", "mixed"]
# the written date patterns with their strptime format (None: day order unknown)
DATE_PATTERNS = {
    "YYYY-MM-DD": (re.compile(r"^\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$"), "%Y-%m-%d"),
    "YYYYMMDDTHHMMSS": (re.compile(r"^\d{8}T\d{6}$"), "%Y%m%dT%H%M%S"),
    "DD/MM/YYYY or MM/DD/YYYY": (re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"), None),
    "DD.MM.YYYY": (re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$"), "%d.%m.%Y"),
    "YYYY/MM/DD": (re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$"), "%Y/%m/%d"),
    "Mon DD YYYY": (re.compile(r"^[A-Za-z]{3,9}\.? \d{1,2},? \d{4}$"), "%b %d %Y"),
    "DD Mon YYYY": (re.compile(r"^\d{1,2} [A-Za-z]{3,9}\.? \d{4}$"), "%d %b %Y"),
}
BOOLEAN_TOKENS = set(TOKENS["boolean_true"] + TOKENS["boolean_false"]) - {"0", "1"}
PLAIN_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
NUMBER_UNIT = re.compile(r"^((?:[+-]?\d[\d,]*\.?\d*|[+-]?\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z°µ\"'][A-Za-z°µ/²³\"'.]*)$")
CURRENCY = re.compile(r"^([$€£¥]\s*[+-]?[\d,]+(\.\d+)?|[+-]?[\d,]+(\.\d+)?\s*([$€£¥]|USD|EUR|GBP))$")
CURRENCY_SYMBOLS = re.compile(r"[$€£¥]|USD|EUR|GBP")
PERCENT = re.compile(r"^[+-]?\d[\d,]*\.?\d*\s*%$")
NUMBERS = re.compile(r"\d+(?:[.,]\d+)?")
DECIMAL = re.compile(r"\d\.\d")

# splits a column name into lowercase words (camelCase and separators)
def name_tokens(name):
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [token for token in re.split(r"[^a-z0-9]+", spaced.lower()) if token]

# marks values that are hidden missing tokens such as ' ', '?' or 'unknown'
def hidden_missing_mask(text, tokens=None):
    lowered = {token.strip().lower() for token in tokens or TOKENS["hidden_missing"]}
    return text.str.strip().str.lower().isin(lowered)

# a random sample of a Series, or the whole Series if it is small
def sample_values(series, size, seed=0):
    return series if len(series) <= size else series.sample(n=size, random_state=seed)

# parses text to numbers; values that fail are cleaned (',', '$', '%',...
# spaces) and parsed again
def parse_numeric(values, numbers=None):
    numbers = (pd.to_numeric(values, errors="coerce").astype(float) if numbers is None else numbers).copy()
    failed = numbers.isna()
    if failed.any():
        cleaned = values[failed]
        for char in TOKENS["numeric_cleaning"]:
            cleaned = cleaned.str.replace(char, "", regex=False)
        numbers[failed] = pd.to_numeric(cleaned, errors="coerce")
    return numbers

# parses text to dates with one format and hides the pandas warnings
def parse_dates(values, date_format):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(values, errors="coerce", format=date_format)

# the first date format that parses enough values of the sample, or None
def best_date_format(sample):
    if sample.str.contains(r"\d", regex=True).mean() < DATE_SHARE:
        return None
    return next((name for name in DATE_ORDER if parse_dates(sample, name).notna().mean() >= DATE_SHARE), None)

# maps yes/no-like tokens to True and False, or None if the column is not...
# boolean
def boolean_map(present):
    uniques = present.str.lower().unique()
    mapping = {token: True for token in uniques if token in TOKENS["boolean_true"]}
    mapping.update({token: False for token in uniques if token in TOKENS["boolean_false"]})
    return mapping if len(mapping) == len(uniques) and len(set(mapping.values())) == 2 else None

# the share of values with the most common length, and that length
def fixed_length_share(present):
    lengths = present.str.len()
    length = lengths.mode().iloc[0]
    return float((lengths == length).mean()), int(length)

# an id has high confidence when it is monotonic or has a long fixed length
def id_confidence(present, numbers=None):
    share, length = fixed_length_share(present)
    monotonic = numbers is not None and (numbers.is_monotonic_increasing or numbers.is_monotonic_decreasing)
    return "high" if (share >= FIXED_LENGTH_SHARE and length >= ID_MIN_LENGTH) or monotonic else "low"

# a column is id-like when it has enough rows and almost all values are unique
def is_id_like(unique_share, present_count):
    return present_count >= ID_MIN_ROWS and unique_share >= ID_MIN_UNIQUE_SHARE

# the type of a numeric column: id, numeric_code, numeric_discrete or...
# numeric_continuous
def numeric_type(name, present, numbers, unique_share):
    is_integer = bool((numbers.to_numpy() == np.floor(numbers.to_numpy())).all())
    if is_integer and is_id_like(unique_share, len(present)):
        return "id", id_confidence(present, numbers)
    if is_integer and set(TOKENS["name_hints"]["code"]) & set(name_tokens(name)):
        return "numeric_code", "high" if fixed_length_share(present)[0] >= FIXED_LENGTH_SHARE else "low"
    if is_integer and numbers.nunique() <= DISCRETE_MAX_UNIQUE:
        return "numeric_discrete", "high"
    return "numeric_continuous", "high"

# infers the type of one raw text column, builds its typed values and...
# collects details (missing tokens, parse share, failed values) for the audit
def infer_column(name, raw):
    text = raw.astype(str).where(raw.notna(), "")
    quick = pd.to_numeric(text, errors="coerce").astype(float)
    missing = np.zeros(len(raw), dtype=bool)
    missing[quick.isna().to_numpy()] = hidden_missing_mask(text[quick.isna()]).to_numpy()
    missing = pd.Series(missing, index=raw.index)
    present = text[~missing].copy()
    present[quick.isna() & ~missing] = present[quick.isna() & ~missing].str.strip()
    details = {"missing_tokens": text[missing].value_counts().to_dict(), "present": len(present), "confidence": "high"}
    if present.empty:
        return "empty", pd.Series(np.nan, index=raw.index, dtype=object), details
    unique_share = present.nunique() / len(present)
    details["unique"] = int(present.nunique())
    if details["unique"] == 1:
        return "constant", raw.where(~missing, np.nan), details
    mapping = boolean_map(present) if details["unique"] <= len(TOKENS["boolean_true"]) + len(TOKENS["boolean_false"]) else None
    if mapping is not None:
        return "boolean", text.str.strip().str.lower().map(mapping).where(~missing).astype("boolean"), details
    sample = sample_values(present, PARSE_SAMPLE_ROWS)
    if parse_numeric(sample).notna().mean() >= NUMBER_SHARE:
        numbers = parse_numeric(present, quick[~missing])
        details.update(parse_share=float(numbers.notna().mean()), cleaned=int((quick[~missing].isna() & numbers.notna()).sum()))
        details["failed_values"] = present[numbers.isna()].value_counts().head(5).to_dict()
        if details["parse_share"] >= NUMBER_SHARE:
            parsed = numbers.dropna()
            details["is_integer"] = bool((parsed.to_numpy() == np.floor(parsed.to_numpy())).all())
            column_type, details["confidence"] = numeric_type(name, present, parsed, unique_share)
            if column_type in {"id", "numeric_code"}:
                return column_type, raw.where(~missing, np.nan), details
            typed = pd.Series(np.nan, index=raw.index, dtype=float)
            typed[numbers.index] = numbers
            return column_type, typed, details
    date_format = best_date_format(sample)
    if date_format is not None:
        dates = parse_dates(present, date_format)
        details.update(parse_share=float(dates.notna().mean()), date_format=date_format)
        details["failed_values"] = present[dates.isna()].value_counts().head(5).to_dict()
        if details["parse_share"] >= DATE_SHARE:
            typed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
            typed[dates.index] = dates
            return "datetime", typed, details
    typed = raw.where(~missing, np.nan)
    if sample.str.split().str.len().mean() >= TEXT_MIN_WORDS and unique_share >= TEXT_MIN_UNIQUE_SHARE:
        return "text", typed, details
    if is_id_like(unique_share, len(present)):
        if sample.str.contains(DECIMAL).mean() >= 0.5:
            details["confidence"] = "low"
            return "categorical_nominal", typed, details
        details["confidence"] = id_confidence(present)
        return "id", typed, details
    return "categorical_nominal", typed, details

# infers the type of every column; returns the types, the typed view and the...
# details per column
def infer_types(raw):
    types, typed, details = {}, {}, {}
    for name in raw.columns:
        types[name], typed[name], details[name] = infer_column(name, raw[name])
    return types, pd.DataFrame(typed, index=raw.index), details

# the name of the first date pattern a value matches, or None
def date_pattern(value):
    return next((name for name, (pattern, _) in DATE_PATTERNS.items() if pattern.match(value)), None)

# replaces every run of digits by "#", so "3 bd / 2 ba" becomes "# bd / # ba"
def template(value):
    return NUMBERS.sub("#", value)

# the unit or suffix of a "number + unit" value in lowercase, or None
def unit_of(value):
    match = NUMBER_UNIT.match(value)
    return match.group(2).lower() if match else None

# measures the text patterns of one column: shares of plain numbers, numbers...
# with a unit (top units), currency, percent, date-like values (top formats),...
# boolean tokens, packed values (top digit templates) and long text
def pattern_stats(values, sample_rows=2000, seed=0):
    present = pd.Series(values, dtype=object).astype(str).str.strip()
    present = present[present != ""]
    present = sample_values(present, sample_rows, seed)
    if len(present) == 0:
        return {"count": 0}
    units, formats, templates, counts = Counter(), Counter(), Counter(), Counter()
    for value in present:
        if PLAIN_NUMBER.match(value):
            counts["number"] += 1
            continue
        if PERCENT.match(value):
            counts["percent"] += 1
        elif CURRENCY.match(value):
            counts["currency"] += 1
        unit = unit_of(value)
        if unit is not None and not PERCENT.match(value):
            counts["number_with_unit"] += 1
            units[unit] += 1
        found_date = date_pattern(value)
        if found_date is not None:
            counts["date_like"] += 1
            formats[found_date] += 1
        elif len(NUMBERS.findall(value)) >= 2 and not CURRENCY_SYMBOLS.search(value):
            counts["packed"] += 1
            templates[template(value)] += 1
        counts["boolean_like"] += value.lower() in BOOLEAN_TOKENS
        counts["long_text"] += len(value.split()) >= 5
    names = ["number", "number_with_unit", "currency", "percent", "date_like", "boolean_like", "packed", "long_text"]
    total = len(present)
    return {
        "count": total,
        "shares": {name: round(counts[name] / total, 4) for name in names},
        "units": dict(units.most_common(5)),
        "date_formats": dict(formats.most_common(3)),
        "templates": [[name, round(count / total, 4)] for name, count in templates.most_common(3)],
    }

# a short text of the pattern shares above 5% for prompts (for example "unit...
# 98% (kg 60%, g 38%)")
def pattern_text(stats):
    if not stats.get("count"):
        return ""
    labels = {"number": "number", "number_with_unit": "unit", "currency": "currency", "percent": "percent", "date_like": "date", "boolean_like": "boolean", "packed": "packed", "long_text": "long text"}
    parts = []
    for key, label in labels.items():
        share = stats["shares"].get(key, 0.0)
        if share < 0.05:
            continue
        text = f"{label} {share:.0%}"
        if key == "number_with_unit" and stats["units"]:
            text += " (" + ", ".join(f"{unit} {count / stats['count']:.0%}" for unit, count in list(stats["units"].items())[:3]) + ")"
        if key == "date_like" and stats["date_formats"]:
            text += " (" + ", ".join(stats["date_formats"]) + ")"
        if key == "packed" and stats["templates"]:
            text += " (" + ", ".join(f"'{name}'" for name, _ in stats["templates"][:2]) + ")"
        parts.append(text)
    return ", ".join(parts)
