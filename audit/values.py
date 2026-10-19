import numpy as np
import pandas as pd
from audit.context import ALL_TYPES, iqr, mask_index, robust_z
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from tools.columntypes import CATEGORY_TYPES, NUMERIC_TYPES, TOKENS, hidden_missing_mask, name_tokens

# C02: missing values written as tokens such as ' ', '?' or 'unknown'
def hidden_missing(context):
    for column in context.columns_of(ALL_TYPES):
        found = {token: count for token, count in context.details[column].get("missing_tokens", {}).items() if token != ""}
        if found:
            values = context.text(column)
            rows = mask_index(hidden_missing_mask(values) & (values != ""))
            context.add("C02", [column], "tokens", {"tokens": found, "count": int(sum(found.values()))}, rows=rows)

# C03: special numbers like -999, or a spike at the minimum or maximum far...
# from the other values
def sentinel_numbers(context):
    sentinels = set(TOKENS["sentinel_numbers"])
    for column in context.columns_of(NUMERIC_TYPES):
        if context.details[column].get("unique", 0) < 3:
            continue
        values = context.typed[column].dropna().astype(float)
        array = values.to_numpy()
        candidates = {float(array.min()), float(array.max())} | set(np.unique(array[np.isin(array, list(sentinels))]).tolist())
        for value in sorted(candidates):
            count = int((array == value).sum())
            if count / len(array) < context.value("C03", "min_spike_share"):
                continue
            rest = values[values != value]
            spread = iqr(rest) or float(rest.max() - rest.min()) or 1.0
            gap = max(float(rest.min()) - value, value - float(rest.max()), 0.0)
            if gap > context.value("C03", "min_gap_iqr") * spread or (value in sentinels and gap > 0):
                label = int(value) if float(value).is_integer() else value
                evidence = {"value": label, "count": count, "share": round(count / len(array), 4), "gap_in_iqr": round(gap / spread, 2)}
                context.add("C03", [column], f"value={label}", evidence, rows=values.index[(values == value).to_numpy()])

# Cohen's h effect size between two proportions
def cohen_h(first, second):
    return abs(2 * np.arcsin(np.sqrt(first)) - 2 * np.arcsin(np.sqrt(second)))

# how much the target differs between rows with and without a missing value:...
# effect size and p-value (Fisher or chi-square for classes, Mann-Whitney for...
# numbers)
def missing_effect(context, missing):
    target = context.typed[context.target]
    known = target.notna()
    missing, present = missing & known, ~missing & known
    if missing.sum() < 2 or present.sum() < 2:
        return None
    if context.is_classification:
        table = pd.crosstab(missing[known], target[known].astype(str))
        if table.shape[0] != 2 or table.shape[1] < 2:
            return None
        rates = table.div(table.sum(axis=1), axis=0)
        effect = max(cohen_h(rates.loc[True, label], rates.loc[False, label]) for label in table.columns)
        p_value = fisher_exact(table.to_numpy())[1] if table.shape[1] == 2 else chi2_contingency(table)[1]
        return float(effect), float(p_value)
    first, second = target[missing].astype(float), target[present].astype(float)
    pooled = np.sqrt((first.var() + second.var()) / 2) or 1.0
    return float(abs(first.mean() - second.mean()) / pooled), float(mannwhitneyu(first, second).pvalue)

# C04: missing values whose rows have a clearly different target
def informative_missing(context):
    for column in context.columns_of(ALL_TYPES):
        missing = context.typed[column].isna()
        result = missing_effect(context, missing) if 0 < missing.sum() < context.n_rows else None
        if result is None:
            continue
        effect, p_value = result
        if p_value <= context.value("C04", "max_p_value") and effect >= context.value("C04", "min_effect_size"):
            context.add("C04", [column], "target_differs", {"effect_size": round(effect, 3), "p_value": float(f"{p_value:.3g}")}, rows=mask_index(missing))

# C05: groups of columns that are missing in the same rows (Jaccard >= 0.9)
def co_missing(context):
    candidates = [column for column in context.columns_of(ALL_TYPES) if 0 < context.typed[column].isna().sum() < context.n_rows]
    if len(candidates) < 2:
        return
    masks = context.typed[candidates].isna().to_numpy(dtype=np.float64)
    both = masks.T @ masks
    sizes = np.diag(both)
    jaccard = both / (sizes[:, None] + sizes[None, :] - both)
    parent = list(range(len(candidates)))
    def find(item):
        while parent[item] != item:
            item = parent[item]
        return item
    for first, second in np.argwhere(np.triu(np.nan_to_num(jaccard) >= context.value("C05", "min_jaccard"), k=1)):
        parent[find(second)] = find(first)
    blocks = {}
    for item in range(len(candidates)):
        blocks.setdefault(find(item), []).append(item)
    for members in blocks.values():
        if len(members) >= 2:
            rows = context.raw.index[masks[:, members].all(axis=1)]
            context.add("C05", [candidates[item] for item in members], "jaccard", {"columns": len(members)}, rows=rows)

# C06: columns with at least 95% missing values
def mostly_missing(context):
    for column in context.columns_of(ALL_TYPES):
        share = float(context.typed[column].isna().mean()) if context.n_rows else 0.0
        if share >= context.value("C06", "min_missing_share"):
            context.add("C06", [column], "missing_share", {"missing_share": round(share, 4)})

# D01: a few negatives in a non-negative column and percentages above 100
def impossible_values(context):
    for column in context.columns_of(NUMERIC_TYPES):
        values = context.typed[column].dropna().astype(float)
        if values.empty:
            continue
        negative = values < 0
        if 0 < negative.mean() <= context.value("D01", "max_negative_share"):
            rows = values.index[negative.to_numpy()]
            context.add("D01", [column], "few_negatives", {"negatives": len(rows), "min": float(values.min())}, rows=rows)
        over = values > 100
        if set(name_tokens(column)) & set(TOKENS["name_hints"]["percent"]) and over.any():
            rows = values.index[over.to_numpy()]
            context.add("D01", [column], "percent_over_100", {"count": len(rows), "max": float(values.max())}, rows=rows)

# marks the values that a big gap (in IQR units) separates from the rest
def isolated_mask(values, min_gap_iqr):
    spread = iqr(values)
    flagged = np.zeros(len(values), dtype=bool)
    if spread == 0:
        return flagged
    uniques = np.unique(values)
    first, third = np.percentile(values, [25, 75])
    for position in np.where(np.diff(uniques) > min_gap_iqr * spread)[0]:
        if uniques[position + 1] > third:
            flagged |= values >= uniques[position + 1]
        if uniques[position] < first:
            flagged |= values <= uniques[position]
    return flagged

# D02: extreme or isolated values (robust z above 10 or a big gap), likely typos
def extreme_values(context):
    for column in context.columns_of(NUMERIC_TYPES):
        values = context.typed[column].dropna().astype(float)
        if values.nunique() < 3:
            continue
        array = values.to_numpy()
        scores = robust_z(array)
        flagged = isolated_mask(array, context.value("D02", "min_gap_iqr"))
        if scores is not None:
            flagged |= np.abs(scores) > context.value("D02", "min_robust_z")
        if flagged.any() and flagged.mean() <= context.value("D02", "max_share"):
            evidence = {"count": int(flagged.sum()), "values": sorted(set(array[flagged].tolist()), key=abs, reverse=True)[:5], "median": float(np.median(array))}
            context.add("D02", [column], "robust_z_or_gap", evidence, rows=values.index[flagged])

# D03: placeholder dates like 1900-01-01
def placeholder_dates(context):
    for column in context.columns_of({"datetime"}):
        dates = context.text(column).str.strip().str[:10]
        placeholder = dates.isin(TOKENS["placeholder_dates"])
        if placeholder.any():
            context.add("D03", [column], "placeholder", {"count": int(placeholder.sum()), "dates": sorted(dates[placeholder].unique().tolist())}, rows=mask_index(placeholder))

# D05: categories with fewer than 10 rows
def rare_categories(context):
    for column in context.columns_of(CATEGORY_TYPES):
        counts = context.typed[column].dropna().astype(str).value_counts()
        rare = counts[counts < context.value("D05", "min_count")]
        if len(counts) >= 2 and not rare.empty:
            rows = mask_index(context.typed[column].astype(str).isin(rare.index) & context.typed[column].notna())
            evidence = {"rare_levels": len(rare), "levels": len(counts), "examples": rare.head(5).to_dict()}
            context.add("D05", [column], "min_count", evidence, rows=rows, params={"min_count": context.value("D05", "min_count")})

CHECKS = [hidden_missing, sentinel_numbers, informative_missing, co_missing, mostly_missing, impossible_values, extreme_values, placeholder_dates, rare_categories]
