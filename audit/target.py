import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tools.columntypes import CATEGORY_TYPES, NUMERIC_TYPES
from audit.context import ALL_TYPES, label_variants, mask_index
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

# F03: the same class written in several ways (the mapping is a protected...
# target step)
def class_spellings(context):
    if not context.is_classification or context.types[context.target] == "boolean":
        return
    variants = label_variants(context.typed[context.target].dropna())
    if variants:
        mapping = {label: canonical for group in variants.values() for label, canonical in group.items()}
        rows = mask_index(context.typed[context.target].astype(str).isin(list(mapping)))
        context.add("F03", [context.target], "normalization", {"mapping": mapping}, rows=rows, params={"mapping": mapping})

# turns one column into numbers for the tree (categories become codes, missing...
# stays NaN so the tree can use it)
def encode_feature(values, column_type):
    if column_type in NUMERIC_TYPES | {"boolean"}:
        return values.astype(float).to_numpy()
    if column_type == "datetime":
        numbers = values.astype("int64").astype(float)
        numbers[values.isna().to_numpy()] = np.nan
        return numbers.to_numpy()
    codes = pd.factorize(values.astype(object))[0].astype(float)
    codes[codes < 0] = np.nan
    return codes

# cross-validated AUC or R2 of a depth-3 tree on one feature
def single_feature_score(feature, target, task, depth, folds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "regression":
            model, splitter, scoring = DecisionTreeRegressor(max_depth=depth, random_state=0), KFold(folds, shuffle=True, random_state=0), "r2"
        else:
            model, splitter = DecisionTreeClassifier(max_depth=depth, random_state=0), StratifiedKFold(folds, shuffle=True, random_state=0)
            scoring = "roc_auc" if task == "binary-classification" else "roc_auc_ovr"
        scores = cross_val_score(model, feature.reshape(-1, 1), target, cv=splitter, scoring=scoring, error_score=np.nan)
    return np.nan if np.isnan(scores).all() else float(np.nanmean(scores))

# the target of the sample rows as numbers (rare classes left out) and the...
# positions of the rows used, or None when there is too little to test
def leak_target(context, rows):
    target = context.typed.loc[rows, context.target]
    known = np.where(target.notna().to_numpy())[0]
    if len(known) < 10:
        return None
    if context.task == "regression":
        return target.iloc[known].astype(float).to_numpy(), known
    codes = pd.factorize(target.iloc[known].astype(str))[0]
    keep = np.isin(codes, np.where(np.bincount(codes) >= context.value("G01", "cv_folds"))[0])
    codes = pd.factorize(codes[keep])[0]
    return (codes, known[keep]) if len(np.unique(codes)) >= 2 else None

# G01, rule "tree": one feature alone predicts the target almost perfectly (a...
# depth-3 tree in 5-fold CV on a row sample, in parallel; missing values are a...
# signal too)
def single_feature_leaks(context):
    columns = context.columns_of(ALL_TYPES - {"id", "text", "empty", "constant"})
    found = leak_target(context, context.sample_rows())
    if not columns or len(columns) > context.value("G01", "max_features") or found is None:
        return
    target, positions = found
    rows = context.sample_rows()[positions]
    task = "binary-classification" if context.is_classification and len(np.unique(target)) == 2 else context.task
    depth, folds = context.value("G01", "tree_depth"), context.value("G01", "cv_folds")
    jobs = (delayed(single_feature_score)(encode_feature(context.typed.loc[rows, column], context.types[column]), target, task, depth, folds) for column in columns)
    scores = Parallel(n_jobs=context.value("G01", "n_jobs"))(jobs)
    metric = "r2" if context.task == "regression" else "auc"
    limit = context.value("G01", "min_r2" if metric == "r2" else "min_auc")
    for column, score in zip(columns, scores):
        if not np.isnan(score) and score >= limit:
            context.add("G01", [column], "tree", {metric: round(score, 4), "model": f"tree depth {depth}, {folds}-fold CV", "sample_rows": len(target)})

# G01, rule "purity": categories that map almost purely to one class and cover...
# most rows (many levels, which a small tree can not split)
def pure_categories(context):
    if not context.is_classification:
        return
    target = context.typed[context.target].astype(str).where(context.typed[context.target].notna())
    for column in context.columns_of(CATEGORY_TYPES | {"boolean", "numeric_code"}):
        frame = pd.DataFrame({"value": context.typed[column], "target": target}).dropna()
        if frame.empty:
            continue
        table = pd.crosstab(frame["value"], frame["target"])
        sizes = table.sum(axis=1)
        pure = (table.max(axis=1) / sizes >= context.value("G01", "min_purity")) & (sizes >= context.value("D05", "min_count"))
        covered = sizes[pure].sum() / sizes.sum()
        if covered >= context.value("G01", "min_row_share") and not any(issue.code == "G01" and issue.columns == [str(column)] for issue in context.issues):
            context.add("G01", [column], "purity", {"pure_levels": int(pure.sum()), "levels": len(table), "rows_covered": round(float(covered), 4)})

CHECKS = [class_spellings, single_feature_leaks, pure_categories]
