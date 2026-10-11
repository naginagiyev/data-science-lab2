import numpy as np
import pandas as pd
from edatool.schemas import SplitAdvice
from edatool.core.registry import register
from scipy.stats import chi2_contingency, spearmanr
from edatool.detectors.common import mask_index, normalize_labels, numeric_types

# returns the target as numbers (classes become sorted codes)
def target_numbers(context):
    target = context.typed[context.target]
    if context.task == "regression":
        return target.astype(float)
    codes, _ = pd.factorize(target.astype(str).where(target.notna()), sort=True)
    numbers = pd.Series(codes.astype(float), index=target.index)
    return numbers.where(codes >= 0)

# H01: id values that repeat, so a group split is needed
@register("H01", applies_to={"id"}, mode="presplit")
def repeated_entities(context, columns):
    for column in columns:
        values = context.typed[column].dropna().astype(str)
        repeated = values.duplicated(keep=False)
        if repeated.any():
            rows = values.index[repeated.to_numpy()]
            evidence = {"repeated_ids": int(values[repeated.to_numpy()].nunique()), "rows": len(rows)}
            context.ledger.add("H01", [column], "repeats", evidence, rows=rows, fix_params={"key": column})

# H02: a target trend over a date column, so a time split is needed
@register("H02", applies_to={"datetime"}, needs_target=True, mode="presplit")
def time_structure(context, columns):
    target = target_numbers(context)
    for column in columns:
        frame = pd.DataFrame({"time": context.typed[column], "target": target}).dropna()
        if len(frame) < 20 or frame["time"].nunique() < 2:
            continue
        times = frame["time"].astype("int64")
        if context.task == "multi-class-classification":
            bins = pd.qcut(times.rank(method="first"), context.value("H02", "time_bins"), labels=False)
            table = pd.crosstab(bins, frame["target"])
            chi2, p_value = chi2_contingency(table)[:2]
            strength = float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))
            metric = "cramers_v"
        else:
            strength, p_value = spearmanr(times, frame["target"])
            metric = "spearman"
        if abs(strength) >= context.value("H02", "min_abs_rank_corr") and p_value <= context.value("H02", "max_p_value"):
            evidence = {metric: round(float(strength), 4), "p_value": float(f"{p_value:.3g}")}
            context.ledger.add("H02", [column, context.target], metric, evidence, fix_params={"key": column})

# H03: the target or a date follows the row order, so the rows must be...
# shuffled
@register("H03", needs_target=True, mode="presplit")
def sorted_file(context, columns):
    if context.n_rows < 20:
        return
    position = np.arange(context.n_rows)
    minimum = context.value("H03", "min_abs_rank_corr")
    checks = [(context.target, target_numbers(context))]
    checks += [(column, context.typed[column].astype("int64").where(context.typed[column].notna())) for column in context.columns_of({"datetime"})]
    for column, values in checks:
        known = values.notna().to_numpy()
        if known.sum() < 20 or values[known].nunique() < 2:
            continue
        value = spearmanr(position[known], values[known])[0]
        if abs(value) >= minimum:
            context.ledger.add("H03", [column], "row_order", {"spearman_with_row_order": round(float(value), 4)})

# H04: rows that are equal after rounding numbers and normalizing text
@register("H04", cost="S", mode="presplit")
def near_duplicate_rows(context, columns):
    used = [column for column in context.raw.columns if column != context.target and context.types[column] != "id"]
    if not used:
        return
    rows = context.sample_rows()
    exact = pd.Series(context.row_hash(used), index=context.raw.index).loc[rows].duplicated()
    parts = {}
    for column in used:
        values = context.typed.loc[rows, column]
        if context.types[column] in numeric_types:
            parts[column] = values.astype(float).round(context.value("H04", "round_decimals"))
        else:
            parts[column] = normalize_labels(values.astype(str))
    near = pd.DataFrame(parts).duplicated()
    extra = near & ~exact
    if extra.any():
        found = mask_index(extra)
        evidence = {"near_duplicate_rows": len(found), "sample_rows": len(rows)}
        context.ledger.add("H04", [], "rounded", evidence, rows=found)

# H05: too few rows, or too few rows per feature
@register("H05", mode="presplit")
def small_dataset(context, columns):
    features = len([column for column in context.raw.columns if column != context.target and context.types[column] != "id"])
    too_few = context.n_rows < context.value("H05", "min_rows")
    per_feature = context.n_rows < context.value("H05", "min_rows_per_feature") * max(features, 1)
    if too_few or per_feature:
        evidence = {"rows": context.n_rows, "features": features}
        context.ledger.add("H05", [], "rows", evidence)

# turns the split-risk issues into one SplitAdvice: group first, then time,...
# then stratified, else random
def advise_split(context, issues):
    codes = {issue.code: issue for issue in issues}
    if "H01" in codes:
        key = codes["H01"].columns[0]
        return SplitAdvice(strategy="group", key=key, shuffle=True, reason=f"Rows repeat the same `{key}`; keep each group in one split.", issues=issues)
    if "H02" in codes:
        key = codes["H02"].columns[0]
        return SplitAdvice(strategy="time", key=key, shuffle=False, reason=f"The target changes over `{key}`; split by time.", issues=issues)
    if context.is_classification:
        return SplitAdvice(strategy="stratified", key=context.target, shuffle=True, reason="Classification task; keep class shares equal in each split.", issues=issues)
    return SplitAdvice(strategy="random", shuffle=True, reason="No group, time or class structure found.", issues=issues)
