import logging
import pandas as pd
from tools.columntypes import infer_types, pattern_stats, pattern_text

SAMPLE_ROWS = 50000
FULL_COLUMNS = 40
WIDE_TABLE = 60
NUMERIC_TYPES = {"numeric_continuous", "numeric_discrete"}
logger = logging.getLogger("automl.glance")

# shortens a value for display
def short(value, length=40):
    text = str(value)
    return text if len(text) <= length else text[: length - 3] + "..."

# the quick look at one column: type, missing share, uniques, top and random...
# values, range and pattern stats
def column_glance(name, raw_column, column_type, typed_column, details, seed):
    text = raw_column.astype(str)
    missing_tokens = {str(token): int(count) for token, count in details.get("missing_tokens", {}).items()}
    missing = sum(missing_tokens.values())
    present = text[text.str.strip() != ""]
    distinct = pd.Series(present.unique())
    examples = distinct.sample(min(5, len(distinct)), random_state=seed).tolist() if len(distinct) else []
    entry = {
        "name": str(name),
        "type": column_type,
        "confidence": details.get("confidence", "high"),
        "missing_share": round(missing / max(1, len(text)), 4),
        "missing_tokens": missing_tokens,
        "unique": int(present.nunique()),
        "top_values": [[str(value), int(count)] for value, count in present.value_counts().head(5).items()],
        "examples": [str(value) for value in examples],
        "patterns": pattern_stats(present, seed=seed),
    }
    if column_type in NUMERIC_TYPES:
        numbers = pd.to_numeric(typed_column, errors="coerce")
        if numbers.notna().any():
            entry["min"] = float(numbers.min())
            entry["max"] = float(numbers.max())
    return entry

# how interesting a column is for the prompt: hidden missing tokens, text...
# types and unit, packed, currency or date patterns count
def interest(entry):
    shares = entry["patterns"].get("shares", {})
    score = 2.0 * bool(set(entry["missing_tokens"]) - {""})
    score += 1.0 * (entry["type"] not in NUMERIC_TYPES)
    score += sum(shares.get(name, 0.0) for name in ["number_with_unit", "packed", "currency", "percent", "date_like"]) * 3
    return score

# the quick look at the whole table (a sample of at most 50,000 rows for types)
def glance(raw, seed=0):
    sample = raw.sample(SAMPLE_ROWS, random_state=seed) if len(raw) > SAMPLE_ROWS else raw
    types, typed, details = infer_types(sample)
    columns = []
    for position, name in enumerate(sample.columns):
        try:
            columns.append(column_glance(name, sample.iloc[:, position], types[name], typed.iloc[:, position], details[name], seed))
        except Exception as error:
            logger.warning(f"Glance of column {name!r} failed: {type(error).__name__}: {error}")
            columns.append({"name": str(name), "type": types[name], "error": f"{type(error).__name__}: {error}", "patterns": {}, "missing_tokens": {}})
    return {"rows": int(len(raw)), "sampled_rows": int(len(sample)), "columns": columns}

# one compact line per column for prompts (about 60 tokens): type, missing...
# share and tokens, uniques, examples, range and patterns
def column_line(entry, examples=True):
    if "error" in entry:
        return f"- `{entry['name']}` {entry['type']} (stats failed)"
    parts = [f"- `{entry['name']}` {entry['type']}"]
    if entry["missing_share"] > 0:
        tokens = ", ".join(f"{token!r}:{count}" for token, count in list(entry["missing_tokens"].items())[:3])
        parts.append(f"missing {entry['missing_share']:.1%} ({tokens})")
    parts.append(f"{entry['unique']} unique")
    if examples and entry["top_values"]:
        values = entry["top_values"] if entry["unique"] <= 5 else [[value, 0] for value in entry["examples"][:4]]
        parts.append("e.g. " + ", ".join(short(value, 25) for value, _ in values))
    if "min" in entry:
        parts.append(f"range {entry['min']:.4g}..{entry['max']:.4g}")
    patterns = pattern_text(entry["patterns"])
    if patterns:
        parts.append(f"patterns: {patterns}")
    return " | ".join(parts)

# the compact text view of the table for prompts; tables with more than 60...
# columns show the 40 most interesting columns in full and the rest as names
def glance_text(result, examples=True):
    columns = result["columns"]
    lines = [f"{result['rows']} rows, {len(columns)} columns"]
    if len(columns) <= WIDE_TABLE:
        return "\n".join(lines + [column_line(entry, examples) for entry in columns])
    ranked = sorted(range(len(columns)), key=lambda index: -interest(columns[index]))
    full = set(ranked[:FULL_COLUMNS])
    for index, entry in enumerate(columns):
        lines.append(column_line(entry, examples) if index in full else f"- `{entry['name']}` {entry['type']}")
    return "\n".join(lines)
