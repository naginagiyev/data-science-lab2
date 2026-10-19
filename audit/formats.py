import re
import difflib
import pandas as pd
from audit.context import label_variants, mask_index, normalize_labels
from tools.columntypes import CATEGORY_TYPES, DATE_PATTERNS, NUMERIC_TYPES, TOKENS, date_pattern
from tools.columntypes import hidden_missing_mask, name_tokens, sample_values, template, unit_of

# the present values of a column as stripped text (no blanks and no hidden...
# missing tokens), a sample of at most 5,000 values
def present_text(context, column):
    text = context.text(column).str.strip()
    return sample_values(text[(text != "") & ~hidden_missing_mask(text)], 5000)

# B01: numbers stored as text that needed cleaning, had blocking tokens or...
# had values that failed to parse
def numbers_as_text(context):
    for column in context.columns_of(NUMERIC_TYPES):
        details = context.details[column]
        blocking = {token: count for token, count in details.get("missing_tokens", {}).items() if token != ""}
        cleaned, failed = details.get("cleaned", 0), details.get("failed_values", {})
        if not (blocking or cleaned or failed):
            continue
        evidence = {"parse_share": round(details.get("parse_share", 1.0), 4)}
        evidence.update({key: value for key, value in [("blocking_tokens", blocking), ("values_needing_cleaning", cleaned), ("failed_values", failed)] if value})
        values = context.text(column)
        rows = mask_index(pd.to_numeric(values, errors="coerce").isna() & (values.str.strip() != ""))
        context.add("B01", [column], "parse_after_cleaning", evidence, rows=rows, params={"cleaning": list(TOKENS["numeric_cleaning"]) if cleaned else []})

# B02: dates stored as text, with the found date format
def dates_as_text(context):
    for column in context.columns_of({"datetime"}):
        details = context.details[column]
        evidence = {"format": details.get("date_format"), "parse_share": round(details.get("parse_share", 1.0), 4)}
        if details.get("failed_values"):
            evidence["failed_values"] = details["failed_values"]
        context.add("B02", [column], "datetime_parse", evidence, params={"format": details.get("date_format")})

# B04: integer codes such as a zipcode, whose size and order mean nothing
def numeric_codes(context):
    for column in context.columns_of({"numeric_code"}):
        values = context.typed[column].dropna().astype(str)
        hints = sorted(set(name_tokens(column)) & set(TOKENS["name_hints"]["code"]))
        context.add("B04", [column], "name_hint", {"levels": int(values.nunique()), "length": int(values.str.len().mode().iloc[0]), "name_hints": hints})

# B05: the same category written in several ways (a mapping is given), or...
# very similar labels (no mapping: the cleaner decides)
def inconsistent_labels(context):
    for column in context.columns_of(CATEGORY_TYPES):
        values = context.typed[column].dropna()
        if values.empty:
            continue
        variants = label_variants(values)
        if variants:
            mapping = {label: canonical for group in variants.values() for label, canonical in group.items()}
            rows = mask_index(context.typed[column].astype(str).isin(list(mapping)) & context.typed[column].notna())
            context.add("B05", [column], "normalization", {"variant_groups": len(variants), "mapping": dict(list(mapping.items())[:10])}, rows=rows, params={"mapping": mapping})
        labels = sorted(normalize_labels(pd.Series(values.unique())).unique())
        if len(labels) > 200:
            continue
        pairs = []
        for index, first in enumerate(labels):
            for second in labels[index + 1 :]:
                ratio = difflib.SequenceMatcher(None, first, second).ratio()
                if ratio >= context.value("B05", "min_similarity"):
                    pairs.append([first, second, round(ratio, 3)])
        if pairs:
            context.add("B05", [column], "similarity", {"similar_pairs": pairs[:5], "pair_count": len(pairs)})

# B06: yes/no columns that use more than one token pair (Yes/No and Y/N)
def mixed_booleans(context):
    for column in context.columns_of({"boolean"}):
        values = context.raw[column].astype(str).str.strip()[context.typed[column].notna().to_numpy()]
        tokens = set(values.str.lower().unique())
        pairs = [f"{yes}/{no}" for yes, no in zip(TOKENS["boolean_true"], TOKENS["boolean_false"]) if {yes, no} & tokens]
        if len(pairs) > 1:
            context.add("B06", [column], "token_pairs", {"token_pairs": pairs, "tokens": sorted(values.unique().tolist())})

# B07: numbers written with a unit or suffix ("1.37kg", "8GB", "3.2k")
def numbers_with_units(context):
    for column in context.columns_of(CATEGORY_TYPES | {"text", "id"}):
        values = present_text(context, column)
        if len(values) < context.value("B07", "min_present"):
            continue
        units = values.map(unit_of)
        share = float(units.notna().mean())
        if share >= context.value("B07", "min_share"):
            counts = units.dropna().value_counts()
            evidence = {"unit_share": round(share, 4), "units": counts.head(10).to_dict(), "examples": values[units.notna()].drop_duplicates().head(5).tolist()}
            context.add("B07", [column], "number_unit", evidence, params={"units": counts.index.tolist()[:10]})

# turns a digit template ("# bd / # ba") into a regex with one group per number
def template_pattern(text):
    pattern = re.escape(text).replace("\\#", "#").replace("\\ ", " ")
    return re.sub(r"\s+", lambda match: r"\s*", pattern).replace("#", r"(\d+(?:[.,]\d+)?)")

# B08: several numbers packed in one value with a fixed text pattern ("3 bd /...
# 2 ba", "2560x1600"); the top templates must cover most values
def packed_values(context):
    for column in context.columns_of(CATEGORY_TYPES | {"text", "id"}):
        values = present_text(context, column)
        if len(values) < context.value("B08", "min_present"):
            continue
        top = values.map(template).value_counts(normalize=True).head(3)
        if top.sum() < context.value("B08", "min_template_share") or top.index.str.count("#").min() < context.value("B08", "min_slots"):
            continue
        if values.map(date_pattern).notna().mean() >= 0.5:
            continue
        names = [f"{column}_{index + 1}" for index in range(top.index[0].count("#"))]
        evidence = {"templates": {name: round(float(share), 4) for name, share in top.items()}, "examples": values.drop_duplicates().head(5).tolist()}
        context.add("B08", [column], "packed_template", evidence, params={"pattern": template_pattern(top.index[0]), "into": names})

# day-first, month-first or ambiguous for values like 03/04/2020
def day_order(values):
    parts = values[values.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$")].str.split("/")
    first, second = parts.str[0].astype(int), parts.str[1].astype(int)
    if (first > 12).any() and not (second > 12).any():
        return "day_first"
    if (second > 12).any() and not (first > 12).any():
        return "month_first"
    return "ambiguous"

# B09: dates written in two or more formats in one column, with the strptime...
# formats of the parse step
def mixed_dates(context):
    for column in context.columns_of(CATEGORY_TYPES | {"text", "datetime", "id"}):
        values = present_text(context, column)
        if len(values) < context.value("B09", "min_present"):
            continue
        formats = values.map(date_pattern)
        if formats.notna().mean() < context.value("B09", "min_date_share"):
            continue
        shares = formats.value_counts(normalize=True)
        shares = shares[shares >= context.value("B09", "min_minority_share")]
        if len(shares) < 2:
            continue
        order = day_order(values) if "DD/MM/YYYY or MM/DD/YYYY" in shares.index else "not_needed"
        slash = "%m/%d/%Y" if order == "month_first" else "%d/%m/%Y"
        evidence = {"formats": shares.round(4).to_dict(), "day_order": order, "examples": values.drop_duplicates().head(5).tolist()}
        context.add("B09", [column], "date_formats", evidence, params={"formats": [DATE_PATTERNS[name][1] or slash for name in shares.index], "dayfirst": order == "day_first"})

CHECKS = [numbers_as_text, dates_as_text, numeric_codes, inconsistent_labels, mixed_booleans, numbers_with_units, packed_values, mixed_dates]
