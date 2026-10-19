import numpy as np
from edatool.core.registry import register
from edatool.detectors.common import label_variants, mask_index, robust_z, skewness

# names the imbalance level of a majority/minority ratio
def imbalance_level(ratio, levels):
    level = "no imbalance"
    for name, limit in levels.items():
        if ratio > limit:
            level = f"{name.replace('_', ' ')} imbalance"
    return level

# F01: rows without a target value
@register("F01", needs_target=True)
def missing_target(context, columns):
    missing = context.typed[context.target].isna()
    if missing.any():
        rows = mask_index(missing)
        evidence = {"missing": len(rows), "share": round(len(rows) / context.n_rows, 4)}
        context.ledger.add("F01", [context.target], "missing", evidence, rows=rows, fix_params={"condition": f"{context.target} is missing"})

# F02: the target does not fit the task (few values for regression, ordered...
# numbers for classification) (review)
@register("F02", needs_target=True)
def task_type_doubt(context, columns):
    target_type = context.types[context.target]
    values = context.typed[context.target].dropna()
    levels = int(values.nunique())
    evidence = {"task": context.task, "target_type": target_type, "levels": levels}
    if context.task == "regression":
        if target_type not in {"numeric_continuous", "numeric_discrete"}:
            context.ledger.add("F02", [context.target], "regression_non_numeric", evidence)
        elif levels <= context.value("F02", "regression_max_unique"):
            context.ledger.add("F02", [context.target], "regression_few_values", evidence)
        return
    if context.task == "binary-classification" and levels != 2:
        context.ledger.add("F02", [context.target], "binary_level_count", evidence)
    if context.task == "multi-class-classification" and levels == 2:
        context.ledger.add("F02", [context.target], "multi_class_two_levels", evidence)
    min_levels = context.value("F02", "classification_min_numeric_levels")
    if target_type in {"numeric_discrete", "numeric_continuous"} and levels >= min_levels:
        evidence["values"] = sorted(float(value) for value in values.unique())[:20]
        context.ledger.add("F02", [context.target], "ordered_numeric_classes", evidence)

# F03: the same class written in several ways
@register("F03", needs_target=True)
def inconsistent_class_labels(context, columns):
    if not context.is_classification or context.types[context.target] == "boolean":
        return
    values = context.typed[context.target].dropna()
    variants = label_variants(values)
    if variants:
        mapping = {label: canonical for group in variants.values() for label, canonical in group.items()}
        context.ledger.add("F03", [context.target], "normalization", {"mapping": mapping}, fix_params={"mapping": mapping})

# F04: a majority/minority class ratio above 2
@register("F04", needs_target=True)
def class_imbalance(context, columns):
    if not context.is_classification:
        return
    counts = context.typed[context.target].dropna().astype(str).value_counts()
    if len(counts) < 2:
        return
    ratio = counts.iloc[0] / counts.iloc[-1]
    if ratio > context.value("F04", "min_ratio"):
        evidence = {
            "counts": counts.to_dict(),
            "ratio": round(float(ratio), 2),
            "level": imbalance_level(ratio, context.value("F04", "levels")),
            "minority_share": round(float(counts.iloc[-1] / counts.sum()), 4),
        }
        context.ledger.add("F04", [context.target], "ratio", evidence)

# F05: classes with too few rows for cross-validation
@register("F05", needs_target=True)
def rare_classes(context, columns):
    if not context.is_classification:
        return
    counts = context.typed[context.target].dropna().astype(str).value_counts()
    limit = max(context.value("F05", "min_rows"), context.value("F05", "cv_folds"))
    rare = counts[counts <= limit]
    if not rare.empty:
        evidence = {"rare_classes": rare.to_dict(), "limit": limit}
        context.ledger.add("F05", [context.target], "class_rows", evidence, fix_params={"min_count": limit})

# F06: a skewed regression target, or zeros and negatives that block a log...
# transform
@register("F06", needs_target=True)
def regression_target_shape(context, columns):
    if context.is_classification:
        return
    values = context.typed[context.target].dropna().astype(float).to_numpy()
    if len(values) < 3:
        return
    value = skewness(values)
    zeros = int((values == 0).sum())
    negatives = int((values < 0).sum())
    if abs(value) > context.value("F06", "min_abs_skew") or zeros or negatives:
        evidence = {"skew": round(value, 3), "zeros": zeros, "negatives": negatives}
        fix_action = None
        if values.min() > 0:
            evidence["skew_after_log"] = round(skewness(np.log(values)), 3)
        else:
            fix_action = "no_action"
        context.ledger.add("F06", [context.target], "shape", evidence, fix_action=fix_action)

# F07: extreme target values by robust z-score (review)
@register("F07", needs_target=True)
def target_outliers(context, columns):
    if context.is_classification:
        return
    values = context.typed[context.target].dropna().astype(float)
    scores = robust_z(values.to_numpy())
    if scores is None:
        return
    flagged = np.abs(scores) > context.value("F07", "min_robust_z")
    if flagged.any():
        rows = values.index[flagged]
        evidence = {"count": len(rows), "max_robust_z": round(float(np.abs(scores).max()), 2), "median": float(np.median(values)), "top_values": sorted(values[flagged].tolist(), reverse=True)[:5]}
        context.ledger.add("F07", [context.target], "robust_z", evidence, rows=rows)
