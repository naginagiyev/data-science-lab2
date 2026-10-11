import re
import warnings
import numpy as np
import pandas as pd
from edatool.schemas import ColumnInfo
from edatool.core.sampling import sample_values

# date formats tried in order: ISO, compact (20141013T000000), then mixed
date_formats = ["ISO8601", "%Y%m%dT%H%M%S", "mixed"]

# reads one threshold value from the settings
def value(settings, code, name):
    return settings["thresholds"][code][name]["value"]

# splits a column name into lowercase words (camelCase and separators)
def name_tokens(name):
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [token for token in re.split(r"[^a-z0-9]+", spaced.lower()) if token]

# marks values that are hidden missing tokens such as ' ', '?' or 'unknown'
def hidden_missing_mask(text, tokens):
    lowered = {token.strip().lower() for token in tokens}
    return text.str.strip().str.lower().isin(lowered)

# parses text to numbers; values that fail are cleaned (',', '$', '%',...
# spaces) and parsed again
def parse_numeric(values, cleaning_chars, numbers=None):
    if numbers is None:
        numbers = pd.to_numeric(values, errors="coerce").astype(float)
    numbers = numbers.copy()
    failed = numbers.isna()
    if failed.any():
        cleaned = values[failed]
        for char in cleaning_chars:
            cleaned = cleaned.str.replace(char, "", regex=False)
        numbers[failed] = pd.to_numeric(cleaned, errors="coerce")
    return numbers

# parses text to dates with one format and hides the pandas warnings
def parse_dates(values, date_format):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(values, errors="coerce", format=date_format)

# returns the first date format that parses enough values of the sample, or...
# None
def best_date_format(sample, min_share):
    if sample.str.contains(r"\d", regex=True).mean() < min_share:
        return None
    for date_format in date_formats:
        if parse_dates(sample, date_format).notna().mean() >= min_share:
            return date_format
    return None

# maps yes/no-like tokens to True and False, or returns None if the column is...
# not boolean
def boolean_map(present, settings):
    uniques = present.str.lower().unique()
    true_tokens = set(settings["tokens"]["boolean_true"])
    false_tokens = set(settings["tokens"]["boolean_false"])
    mapping = {token: True for token in uniques if token in true_tokens}
    mapping.update({token: False for token in uniques if token in false_tokens})
    if len(mapping) != len(uniques) or len(set(mapping.values())) != 2:
        return None
    return mapping

# returns the share of values with the most common length, and that length
def fixed_length_share(present):
    lengths = present.str.len()
    return float((lengths == lengths.mode().iloc[0]).mean()), int(lengths.mode().iloc[0])

# an id has high confidence when it is monotonic or has a long fixed length
def id_confidence(present, numbers, settings):
    share, length = fixed_length_share(present)
    fixed = share >= value(settings, "B04", "min_fixed_length_share")
    fixed = fixed and length >= value(settings, "E03", "min_pattern_length")
    monotonic = numbers is not None and (numbers.is_monotonic_increasing or numbers.is_monotonic_decreasing)
    return "high" if fixed or monotonic else "low"

# a column is id-like when it has enough rows and almost all values are...
# unique
def is_id_like(unique_share, present_count, settings):
    enough_rows = present_count >= value(settings, "E03", "min_rows")
    return enough_rows and unique_share >= value(settings, "E03", "min_unique_share")

# decides the type of a numeric column: id, numeric_code, numeric_discrete or...
# numeric_continuous
def infer_numeric(name, present, numbers, unique_share, settings):
    values = numbers.to_numpy()
    is_integer = bool((values == np.floor(values)).all())
    if is_integer and is_id_like(unique_share, len(present), settings):
        return "id", id_confidence(present, numbers, settings)
    code_hints = set(settings["tokens"]["name_hints"]["code"])
    if is_integer and code_hints & set(name_tokens(name)):
        share, _ = fixed_length_share(present)
        confidence = "high" if share >= value(settings, "B04", "min_fixed_length_share") else "low"
        return "numeric_code", confidence
    if is_integer and numbers.nunique() <= value(settings, "type_inference", "discrete_max_unique"):
        return "numeric_discrete", "high"
    return "numeric_continuous", "high"

# infers the semantic type of one column, builds its typed values and...
# collects details (missing tokens, parse share) for the detectors
def infer_column(name, raw, settings):
    text = raw.astype(str).where(raw.notna(), "")
    quick_numbers = pd.to_numeric(text, errors="coerce").astype(float)
    not_numeric = quick_numbers.isna()
    tokens = settings["tokens"]["hidden_missing"]
    missing = np.zeros(len(raw), dtype=bool)
    missing[not_numeric.to_numpy()] = hidden_missing_mask(text[not_numeric], tokens).to_numpy()
    missing = pd.Series(missing, index=raw.index)
    present = text[~missing].copy()
    present[not_numeric & ~missing] = present[not_numeric & ~missing].str.strip()
    tokens_found = text[missing].value_counts().to_dict()
    details = {"missing_tokens": tokens_found, "present": len(present)}
    empty = pd.Series(np.nan, index=raw.index, dtype=object)
    if present.empty:
        return "empty", "high", empty, details
    unique_count = present.nunique()
    unique_share = unique_count / len(present)
    details["unique"] = int(unique_count)
    if unique_count == 1:
        return "constant", "high", raw.where(~missing, np.nan), details

    bool_token_count = len(settings["tokens"]["boolean_true"]) + len(settings["tokens"]["boolean_false"])
    mapping = boolean_map(present, settings) if unique_count <= bool_token_count else None
    if mapping is not None:
        typed = text.str.strip().str.lower().map(mapping).where(~missing).astype("boolean")
        return "boolean", "high", typed, details

    sample_size = value(settings, "sampling", "parse_sample_rows")
    sample = sample_values(present, sample_size)
    cleaning_chars = settings["tokens"]["numeric_cleaning"]
    numeric_share = value(settings, "B01", "min_parse_share")
    if parse_numeric(sample, cleaning_chars).notna().mean() >= numeric_share:
        numbers = parse_numeric(present, cleaning_chars, quick_numbers[~missing])
        details["parse_share"] = float(numbers.notna().mean())
        details["cleaned"] = int((quick_numbers[~missing].isna() & numbers.notna()).sum())
        details["failed_values"] = present[numbers.isna()].value_counts().head(5).to_dict()
        if details["parse_share"] >= numeric_share:
            parsed = numbers.dropna()
            details["is_integer"] = bool((parsed.to_numpy() == np.floor(parsed.to_numpy())).all())
            semantic_type, confidence = infer_numeric(name, present, parsed, unique_share, settings)
            if semantic_type in {"id", "numeric_code"}:
                return semantic_type, confidence, raw.where(~missing, np.nan), details
            typed = pd.Series(np.nan, index=raw.index, dtype=float)
            typed[numbers.index] = numbers
            return semantic_type, confidence, typed, details

    date_share = value(settings, "B02", "min_parse_share")
    date_format = best_date_format(sample, date_share)
    if date_format is not None:
        dates = parse_dates(present, date_format)
        details["parse_share"] = float(dates.notna().mean())
        details["failed_values"] = present[dates.isna()].value_counts().head(5).to_dict()
        details["date_format"] = date_format
        if details["parse_share"] >= date_share:
            typed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
            typed[dates.index] = dates
            return "datetime", "high", typed, details

    typed = raw.where(~missing, np.nan)
    mean_words = sample.str.split().str.len().mean()
    if mean_words >= value(settings, "E05", "min_mean_words"):
        if unique_share >= value(settings, "E05", "min_unique_share"):
            return "text", "high", typed, details
    if is_id_like(unique_share, len(present), settings):
        return "id", id_confidence(present, None, settings), typed, details
    return "categorical_nominal", "high", typed, details

# infers the type and role of every column and builds the typed view
def infer_schema(raw, target, settings):
    schema = []
    typed_columns = {}
    details = {}
    for name in raw.columns:
        semantic_type, confidence, typed, column_details = infer_column(name, raw[name], settings)
        role = "target" if name == target else "id" if semantic_type == "id" else "feature"
        schema.append(ColumnInfo(name=str(name), type=semantic_type, role=role, confidence=confidence))
        typed_columns[name] = typed
        details[name] = column_details
    typed_view = pd.DataFrame(typed_columns, index=raw.index)
    return schema, typed_view, details
