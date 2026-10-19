import numpy as np
import pandas as pd
from datetime import datetime
from edatool.core.registry import register
from edatool.core.typeinference import name_tokens
from edatool.detectors.common import category_types, iqr, mask_index, numeric_types, robust_z, text_values

# D01: a few negatives in a non-negative column, percentages above 100 and...
# rates outside 0..1
@register("D01", applies_to=numeric_types)
def impossible_values(context, columns):
    hints = context.settings["tokens"]["name_hints"]
    for column in columns:
        values = context.typed[column].dropna().astype(float)
        if values.empty:
            continue
        negative = values < 0
        share = float(negative.mean())
        if 0 < share <= context.value("D01", "max_negative_share"):
            rows = values.index[negative.to_numpy()]
            evidence = {"negatives": len(rows), "share": round(share, 4), "min": float(values.min())}
            context.ledger.add("D01", [column], "few_negatives", evidence, rows=rows)
        tokens = set(name_tokens(column))
        if tokens & set(hints["percent"]):
            over = values > context.value("D01", "percent_max")
            if over.any():
                rows = values.index[over.to_numpy()]
                context.ledger.add("D01", [column], "percent_over_100", {"count": len(rows), "max": float(values.max())}, rows=rows, needs_review=True)
        if tokens & set(hints["rate"]):
            outside = (values < context.value("D01", "rate_min")) | (values > context.value("D01", "rate_max"))
            if 0 < outside.mean() < 0.5:
                rows = values.index[outside.to_numpy()]
                context.ledger.add("D01", [column], "rate_outside_0_1", {"count": len(rows)}, rows=rows, needs_review=True)

# marks the values that a big gap (in IQR units) separates from the rest
def isolated_mask(values, min_gap_iqr):
    spread = iqr(values)
    flagged = np.zeros(len(values), dtype=bool)
    if spread == 0:
        return flagged
    uniques = np.unique(values)
    gaps = np.diff(uniques)
    first, third = np.percentile(values, [25, 75])
    for position in np.where(gaps > min_gap_iqr * spread)[0]:
        if uniques[position + 1] > third:
            flagged |= values >= uniques[position + 1]
        if uniques[position] < first:
            flagged |= values <= uniques[position]
    return flagged

# D02: extreme or isolated values (robust z above 10 or a big gap), likely...
# typos (review)
@register("D02", applies_to=numeric_types)
def extreme_isolated_values(context, columns):
    for column in columns:
        values = context.typed[column].dropna().astype(float)
        if values.nunique() < 3:
            continue
        array = values.to_numpy()
        scores = robust_z(array)
        flagged = isolated_mask(array, context.value("D02", "min_gap_iqr"))
        if scores is not None:
            flagged |= np.abs(scores) > context.value("D02", "min_robust_z")
        if not flagged.any() or flagged.mean() > context.value("D02", "max_share"):
            continue
        rows = values.index[flagged]
        evidence = {
            "count": len(rows),
            "values": sorted(set(array[flagged].tolist()), key=abs, reverse=True)[:5],
            "median": float(np.median(array)),
            "max_robust_z": round(float(np.abs(scores).max()), 2) if scores is not None else None,
        }
        context.ledger.add("D02", [column], "robust_z_or_gap", evidence, rows=rows)

# D03: dates in the future and placeholder dates like 1900-01-01
@register("D03", applies_to={"datetime"})
def invalid_dates(context, columns):
    placeholders = context.settings["tokens"]["placeholder_dates"]
    now = pd.Timestamp(datetime.now())
    for column in columns:
        values = context.typed[column].dropna()
        future = values > now
        if future.any():
            rows = values.index[future.to_numpy()]
            context.ledger.add("D03", [column], "future", {"count": len(rows), "max": str(values.max())}, rows=rows)
        dates = text_values(context, column).str.strip().str[:10]
        placeholder = dates.isin(placeholders)
        if placeholder.any():
            rows = mask_index(placeholder)
            found = sorted(dates[placeholder].unique().tolist())
            context.ledger.add("D03", [column], "placeholder", {"count": len(rows), "dates": found}, rows=rows)

# D04: rules that hold in at least 95% of rows but not in all (date order, A...
# = B + C) (review)
@register("D04", applies_to=numeric_types | {"datetime"}, cost="S")
def cross_column_rules(context, columns):
    min_share = context.value("D04", "min_rule_share")
    dates = [column for column in columns if context.types[column] == "datetime"]
    for index, first in enumerate(dates):
        for second in dates[index + 1 :]:
            both = context.typed[[first, second]].dropna()
            if both.empty:
                continue
            for early, late in [(first, second), (second, first)]:
                holds = both[early] <= both[late]
                if min_share <= holds.mean() < 1:
                    rows = both.index[(~holds).to_numpy()]
                    context.ledger.add("D04", [early, late], f"{early}<={late}", {"holds_share": round(float(holds.mean()), 4), "violations": len(rows)}, rows=rows)
    integers = [column for column in columns if context.types[column] in numeric_types and context.details[column].get("is_integer")]
    if len(integers) < 3 or len(integers) > 30:
        return
    full = context.typed[integers].astype(float).dropna()
    sample = full.loc[full.index.intersection(context.sample_rows())].to_numpy()
    for result_index, result in enumerate(integers):
        for first_index in range(len(integers)):
            for second_index in range(first_index + 1, len(integers)):
                if result_index in {first_index, second_index}:
                    continue
                holds = sample[:, result_index] == sample[:, first_index] + sample[:, second_index]
                if not min_share <= holds.mean() < 1:
                    continue
                first, second = integers[first_index], integers[second_index]
                full_holds = full[result] == full[first] + full[second]
                if min_share <= full_holds.mean() < 1:
                    rows = full.index[(~full_holds).to_numpy()]
                    rule = f"{result}={first}+{second}"
                    context.ledger.add("D04", [result, first, second], rule, {"rule": rule, "holds_share": round(float(full_holds.mean()), 4), "violations": len(rows)}, rows=rows)

# D05: categories with fewer than 10 rows
@register("D05", applies_to=category_types)
def rare_categories(context, columns):
    for column in columns:
        counts = context.typed[column].dropna().astype(str).value_counts()
        rare = counts[counts < context.value("D05", "min_count")]
        if len(counts) < 2 or rare.empty:
            continue
        values = context.typed[column].astype(str)
        rows = mask_index(values.isin(rare.index) & context.typed[column].notna())
        evidence = {"rare_levels": len(rare), "levels": len(counts), "examples": rare.head(5).to_dict()}
        fix_params = {"min_count": context.value("D05", "min_count")}
        context.ledger.add("D05", [column], "min_count", evidence, rows=rows, fix_params=fix_params)
