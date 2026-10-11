import difflib
import pandas as pd
from edatool.core.registry import register
from edatool.core.typeinference import name_tokens
from edatool.detectors.common import category_types, label_variants, mask_index, normalize_labels, numeric_types, text_values

# value pattern classes used by B03, checked in this order
pattern_classes = [
    ("date", r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}.*"),
    ("number", r"[-+]?(\d+\.?\d*|\.\d+)"),
    ("alpha", r"[^\W\d_]+([\s\-'&().,/]+[^\W\d_]*)*"),
]

# checks if a column was stored as text in the input
def is_text_stored(context, column):
    return context.raw[column].dtype == object

# B01: numeric columns stored as text that needed cleaning, had blocking...
# tokens or had values that failed to parse
@register("B01", applies_to=numeric_types)
def numbers_as_text(context, columns):
    for column in columns:
        if not is_text_stored(context, column):
            continue
        details = context.details[column]
        blocking = {token: count for token, count in details.get("missing_tokens", {}).items() if token != ""}
        cleaned = details.get("cleaned", 0)
        failed = details.get("failed_values", {})
        if blocking or cleaned or failed:
            evidence = {"parse_share": round(details.get("parse_share", 1.0), 4)}
            if blocking:
                evidence["blocking_tokens"] = blocking
            if cleaned:
                evidence["values_needing_cleaning"] = cleaned
            if failed:
                evidence["failed_values"] = failed
            cleaning = [char for char in context.settings["tokens"]["numeric_cleaning"] if cleaned]
            values = text_values(context, column)
            blocked = pd.to_numeric(values, errors="coerce").isna() & (values.str.strip() != "")
            rows = mask_index(blocked)
            context.ledger.add("B01", [column], "parse_after_cleaning", evidence, rows=rows, fix_params={"cleaning": cleaning})

# B02: date columns stored as text, with the found date format
@register("B02", applies_to={"datetime"})
def dates_as_text(context, columns):
    for column in columns:
        if not is_text_stored(context, column):
            continue
        details = context.details[column]
        evidence = {"format": details.get("date_format"), "parse_share": round(details.get("parse_share", 1.0), 4)}
        if details.get("failed_values"):
            evidence["failed_values"] = details["failed_values"]
        fix_params = {"format": details.get("date_format")}
        context.ledger.add("B02", [column], "datetime_parse", evidence, fix_params=fix_params)

# B03: columns that mix value patterns (numbers, words, dates) with a...
# minority of at least 1%
@register("B03", applies_to=category_types | {"id", "numeric_code"})
def mixed_formats(context, columns):
    for column in columns:
        counts = context.typed[column].dropna().astype(str).str.strip().value_counts()
        if counts.empty:
            continue
        labels = pd.Series("mixed", index=counts.index)
        for name, pattern in reversed(pattern_classes):
            labels[counts.index.str.fullmatch(pattern)] = name
        shares = counts.groupby(labels.to_numpy()).sum() / counts.sum()
        shares = shares.sort_values(ascending=False)
        if len(shares) > 1 and shares.iloc[1] >= context.value("B03", "min_minority_share"):
            examples = {name: labels[labels == name].index[0] for name in shares.index}
            evidence = {"pattern_shares": shares.round(4).to_dict(), "examples": examples}
            context.ledger.add("B03", [column], "pattern_classes", evidence)

# B04: integer columns that are codes (for example zipcode), so their size...
# and order mean nothing
@register("B04", applies_to={"numeric_code"})
def numeric_code(context, columns):
    for column in columns:
        values = context.typed[column].dropna().astype(str)
        hints = sorted(set(name_tokens(column)) & set(context.settings["tokens"]["name_hints"]["code"]))
        evidence = {"levels": int(values.nunique()), "length": int(values.str.len().mode().iloc[0]), "name_hints": hints}
        context.ledger.add("B04", [column], "name_hint", evidence, fix_params={"method": "onehot"})

# B05: the same category written in several ways, and very similar labels...
# (review)
@register("B05", applies_to=category_types)
def inconsistent_labels(context, columns):
    for column in columns:
        values = context.typed[column].dropna()
        if values.empty:
            continue
        variants = label_variants(values)
        if variants:
            mapping = {label: canonical for group in variants.values() for label, canonical in group.items()}
            evidence = {"variant_groups": len(variants), "mapping": dict(list(mapping.items())[:10])}
            rows = mask_index(context.typed[column].astype(str).isin(list(mapping)) & context.typed[column].notna())
            context.ledger.add("B05", [column], "normalization", evidence, rows=rows, fix_params={"mapping": mapping})
        labels = sorted(normalize_labels(pd.Series(values.unique())).unique())
        if len(labels) > 200:
            continue
        min_similarity = context.value("B05", "min_similarity")
        pairs = []
        for index, first in enumerate(labels):
            for second in labels[index + 1 :]:
                ratio = difflib.SequenceMatcher(None, first, second).ratio()
                if ratio >= min_similarity:
                    pairs.append([first, second, round(ratio, 3)])
        if pairs:
            evidence = {"similar_pairs": pairs[:5], "pair_count": len(pairs)}
            context.ledger.add("B05", [column], "similarity", evidence, needs_review=True)

# B06: yes/no columns that use more than one token pair (for example Yes/No...
# and Y/N)
@register("B06", applies_to={"boolean"})
def mixed_boolean_tokens(context, columns):
    true_tokens = context.settings["tokens"]["boolean_true"]
    false_tokens = context.settings["tokens"]["boolean_false"]
    for column in columns:
        values = context.raw[column].astype(str).str.strip()
        tokens = set(values[context.typed[column].notna().to_numpy()].str.lower().unique())
        pairs = [f"{yes}/{no}" for yes, no in zip(true_tokens, false_tokens) if {yes, no} & tokens]
        if len(pairs) > 1:
            raw_tokens = sorted(values[context.typed[column].notna().to_numpy()].unique().tolist())
            evidence = {"token_pairs": pairs, "tokens": raw_tokens}
            context.ledger.add("B06", [column], "token_pairs", evidence)
