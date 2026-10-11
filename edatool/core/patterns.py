import re
import pandas as pd
from collections import Counter

PLAIN_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
NUMBER_UNIT = re.compile(r"^((?:[+-]?\d[\d,]*\.?\d*|[+-]?\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z°µ\"'][A-Za-z°µ/²³\"'.]*)$")
CURRENCY = re.compile(r"^([$€£¥]\s*[+-]?[\d,]+(\.\d+)?|[+-]?[\d,]+(\.\d+)?\s*([$€£¥]|USD|EUR|GBP))$")
PERCENT = re.compile(r"^[+-]?\d[\d,]*\.?\d*\s*%$")
NUMBERS = re.compile(r"\d+(?:[.,]\d+)?")
WORDS = re.compile(r"\S+")
DATE_PATTERNS = {
    "YYYY-MM-DD": re.compile(r"^\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$"),
    "YYYYMMDDTHHMMSS": re.compile(r"^\d{8}T\d{6}$"),
    "DD/MM/YYYY or MM/DD/YYYY": re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
    "DD.MM.YYYY": re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$"),
    "YYYY/MM/DD": re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$"),
    "Mon DD YYYY": re.compile(r"^[A-Za-z]{3,9}\.? \d{1,2},? \d{4}$"),
    "DD Mon YYYY": re.compile(r"^\d{1,2} [A-Za-z]{3,9}\.? \d{4}$"),
}
BOOLEAN_TOKENS = {"true", "false", "yes", "no", "y", "n", "t", "f"}
CURRENCY_SYMBOLS = re.compile(r"[$€£¥]|USD|EUR|GBP")

# returns the name of the first date pattern a value matches, or None
def date_pattern(value):
    for name, pattern in DATE_PATTERNS.items():
        if pattern.match(value):
            return name
    return None

# replaces every run of digits by "#", so "3 bd / 2 ba" becomes "# bd / # ba"
def template(value):
    return re.sub(r"\d+(?:[.,]\d+)?", "#", value)

# returns the unit or suffix of a "number + unit" value in lowercase, or None
def unit_of(value):
    match = NUMBER_UNIT.match(value)
    return match.group(2).lower() if match else None

# measures the text patterns of one column: shares of plain numbers, numbers...
# with a unit or suffix (with the top units), currency, percent, date-like...
# values (with the top formats), boolean-like tokens, values with two or more...
# numbers inside (packed) and long text, plus the top digit templates
def pattern_stats(values, sample_rows=2000, seed=0):
    present = pd.Series(values, dtype=object).astype(str).str.strip()
    present = present[present != ""]
    if len(present) > sample_rows:
        present = present.sample(sample_rows, random_state=seed)
    total = len(present)
    if total == 0:
        return {"count": 0}
    units, formats, templates = Counter(), Counter(), Counter()
    counts = Counter()
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
        if value.lower() in BOOLEAN_TOKENS:
            counts["boolean_like"] += 1
        if len(WORDS.findall(value)) >= 5:
            counts["long_text"] += 1
    shares = {name: round(counts[name] / total, 4) for name in ["number", "number_with_unit", "currency", "percent", "date_like", "boolean_like", "packed", "long_text"]}
    top_templates = [[name, round(count / total, 4)] for name, count in templates.most_common(3)]
    return {
        "count": total,
        "shares": shares,
        "units": dict(units.most_common(5)),
        "date_formats": dict(formats.most_common(3)),
        "templates": top_templates,
    }

# a short text of the pattern shares above 5%, for prompts (for example...
# "unit 98% (kg 60%, g 38%)")
def pattern_text(stats):
    if not stats.get("count"):
        return ""
    parts = []
    names = {"number": "number", "number_with_unit": "unit", "currency": "currency", "percent": "percent", "date_like": "date", "boolean_like": "boolean", "packed": "packed", "long_text": "long text"}
    for key, label in names.items():
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
