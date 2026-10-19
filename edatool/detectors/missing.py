import numpy as np
import pandas as pd
from edatool.core.registry import register
from edatool.core.typeinference import hidden_missing_mask
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from edatool.detectors.common import all_types, iqr, mask_index, numeric_types, text_values

# Cohen's h effect size between two proportions
def cohen_h(first, second):
    return abs(2 * np.arcsin(np.sqrt(first)) - 2 * np.arcsin(np.sqrt(second)))

# C01: empty cells per column; the priority grows with the missing share
@register("C01", applies_to=all_types)
def missing_values(context, columns):
    for column in columns:
        blank = context.details[column].get("missing_tokens", {}).get("", 0)
        if not blank:
            continue
        share = blank / context.n_rows
        priority = "consider"
        if share >= context.value("C01", "must_fix_min_share"):
            priority = "must_fix"
        elif share >= context.value("C01", "should_fix_min_share"):
            priority = "should_fix"
        rows = mask_index(text_values(context, column).str.strip() == "")
        strategy = "median" if context.types[column] in numeric_types else "mode"
        evidence = {"missing": blank, "share": round(share, 4)}
        context.ledger.add("C01", [column], "blank", evidence, rows=rows, priority=priority, fix_params={"strategy": strategy})

# C02: missing values written as tokens like ' ', '?' or 'unknown'
@register("C02", applies_to=all_types)
def hidden_missing(context, columns):
    tokens = context.settings["tokens"]["hidden_missing"]
    for column in columns:
        found = {token: count for token, count in context.details[column].get("missing_tokens", {}).items() if token != ""}
        if not found:
            continue
        values = text_values(context, column)
        mask = hidden_missing_mask(values, tokens) & (values != "")
        evidence = {"tokens": found, "count": int(sum(found.values()))}
        mapping = {token: None for token in found}
        context.ledger.add("C02", [column], "tokens", evidence, rows=mask_index(mask), fix_params={"mapping": mapping})

# C03: special numbers like -999, or a spike at the min or max far from the...
# other values (review)
@register("C03", applies_to=numeric_types)
def sentinel_numbers(context, columns):
    sentinels = set(context.settings["tokens"]["sentinel_numbers"])
    for column in columns:
        values = context.typed[column].dropna().astype(float)
        if context.details[column].get("unique", 0) < 3:
            continue
        array = values.to_numpy()
        present_sentinels = set(np.unique(array[np.isin(array, list(sentinels))]).tolist())
        candidates = {float(array.min()), float(array.max())} | present_sentinels
        for value in sorted(candidates):
            label = int(value) if float(value).is_integer() else value
            count = int((array == value).sum())
            share = count / len(array)
            if share < context.value("C03", "min_spike_share"):
                continue
            rest = values[values != value]
            spread = iqr(rest) or float(rest.max() - rest.min()) or 1.0
            gap = max(float(rest.min()) - value, value - float(rest.max()), 0.0)
            listed = value in sentinels
            if gap > context.value("C03", "min_gap_iqr") * spread or (listed and gap > 0):
                rows = values.index[(values == value).to_numpy()]
                evidence = {
                    "value": label,
                    "count": count,
                    "share": round(share, 4),
                    "next_value": float(rest.min() if value < rest.min() else rest.max()),
                    "gap_in_iqr": round(gap / spread, 2),
                }
                fix_params = {"mapping": {label: None}}
                context.ledger.add("C03", [column], f"value={label}", evidence, rows=rows, fix_params=fix_params)

# measures how much the target differs between rows with and without a...
# missing value (effect size and p-value)
def missing_effect(context, missing):
    target = context.typed[context.target]
    known = target.notna()
    missing = missing & known
    present = ~missing & known
    if missing.sum() < 2 or present.sum() < 2:
        return None
    if context.is_classification:
        labels = target[known].astype(str)
        table = pd.crosstab(missing[known], labels)
        if table.shape != (2, table.shape[1]) or table.shape[1] < 2:
            return None
        rates = table.div(table.sum(axis=1), axis=0)
        effect = max(cohen_h(rates.loc[True, label], rates.loc[False, label]) for label in table.columns)
        p_value = fisher_exact(table.to_numpy())[1] if table.shape[1] == 2 else chi2_contingency(table)[1]
        return float(effect), float(p_value), {"class_share_missing": rates.loc[True].round(4).to_dict()}
    values = target.astype(float)
    first = values[missing]
    second = values[present]
    pooled = np.sqrt((first.var() + second.var()) / 2) or 1.0
    effect = abs(first.mean() - second.mean()) / pooled
    p_value = mannwhitneyu(first, second).pvalue
    return float(effect), float(p_value), {"target_mean_missing": round(float(first.mean()), 4), "target_mean_present": round(float(second.mean()), 4)}

# C04: missing values whose rows have a clearly different target
@register("C04", applies_to=all_types, needs_target=True)
def informative_missingness(context, columns):
    for column in columns:
        missing = context.typed[column].isna()
        if not 0 < missing.sum() < context.n_rows:
            continue
        result = missing_effect(context, missing)
        if result is None:
            continue
        effect, p_value, extra = result
        if p_value <= context.value("C04", "max_p_value") and effect >= context.value("C04", "min_effect_size"):
            evidence = {"effect_size": round(effect, 3), "p_value": float(f"{p_value:.3g}"), **extra}
            fix_params = {"condition": f"{column} is missing", "new_name": f"{column}_missing"}
            context.ledger.add("C04", [column], "target_differs", evidence, rows=mask_index(missing), fix_params=fix_params)

# C05: groups of columns that are missing in the same rows (jaccard >= 0.9)
@register("C05", applies_to=all_types)
def co_missing_block(context, columns):
    candidates = [column for column in columns if 0 < context.typed[column].isna().sum() < context.n_rows]
    if len(candidates) < 2:
        return
    masks = context.typed[candidates].isna().to_numpy(dtype=np.float64)
    both = masks.T @ masks
    sizes = np.diag(both)
    jaccard = both / (sizes[:, None] + sizes[None, :] - both)
    min_jaccard = context.value("C05", "min_jaccard")
    parent = list(range(len(candidates)))
    def find(item):
        while parent[item] != item:
            item = parent[item]
        return item
    for first, second in np.argwhere(np.triu(np.nan_to_num(jaccard) >= min_jaccard, k=1)):
        parent[find(second)] = find(first)
    blocks = {}
    for item in range(len(candidates)):
        blocks.setdefault(find(item), []).append(item)
    for members in blocks.values():
        if len(members) < 2:
            continue
        names = [candidates[item] for item in members]
        all_missing = masks[:, members].all(axis=1)
        rows = context.raw.index[all_missing]
        pair_values = [jaccard[a, b] for index, a in enumerate(members) for b in members[index + 1 :]]
        evidence = {"columns": len(names), "min_jaccard": round(float(min(pair_values)), 4)}
        context.ledger.add("C05", names, "jaccard", evidence, rows=rows)

# C06: columns with at least 95% missing values
@register("C06", applies_to=all_types)
def almost_fully_missing(context, columns):
    for column in columns:
        share = float(context.typed[column].isna().mean()) if context.n_rows else 0.0
        if share >= context.value("C06", "min_missing_share"):
            context.ledger.add("C06", [column], "missing_share", {"missing_share": round(share, 4)})
