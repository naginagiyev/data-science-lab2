import warnings
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from joblib import Parallel, delayed
from edatool.core.registry import register
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from edatool.detectors.common import all_types, category_types, mask_index, numeric_types

# turns one column into numbers for the tree model (categories become codes)
def encode_feature(values, semantic_type):
    if semantic_type in numeric_types:
        return values.astype(float).to_numpy()
    if semantic_type == "boolean":
        return values.astype(float).to_numpy()
    if semantic_type == "datetime":
        numbers = values.astype("int64").astype(float)
        numbers[values.isna().to_numpy()] = np.nan
        return numbers.to_numpy()
    codes, _ = pd.factorize(values.astype(object))
    codes = codes.astype(float)
    codes[codes < 0] = np.nan
    return codes

# cross-validated AUC or R2 of a depth-3 tree that uses only one feature
def single_feature_score(features, target, task, depth, folds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "regression":
            model = DecisionTreeRegressor(max_depth=depth, random_state=0)
            splitter = KFold(folds, shuffle=True, random_state=0)
            scoring = "r2"
        else:
            model = DecisionTreeClassifier(max_depth=depth, random_state=0)
            splitter = StratifiedKFold(folds, shuffle=True, random_state=0)
            scoring = "roc_auc" if task == "binary-classification" else "roc_auc_ovr"
        scores = cross_val_score(model, features.reshape(-1, 1), target, cv=splitter, scoring=scoring, error_score=np.nan)
    return float(np.nanmean(scores)) if not np.isnan(scores).all() else np.nan

# G01: a single feature that predicts the target almost perfectly (runs in...
# parallel on a sample)
@register("G01", applies_to=all_types - {"id", "text", "empty", "constant"}, needs_target=True, cost="E")
def single_feature_predicts_target(context, columns):
    if len(columns) > context.value("G01", "max_features"):
        return {"*": f"more than {context.value('G01', 'max_features')} features"}
    rows = context.sample_rows()
    target = context.typed.loc[rows, context.target]
    known = target.notna().to_numpy()
    if known.sum() < 10:
        return
    target = target[known]
    target = target.astype(float).to_numpy() if context.task == "regression" else pd.factorize(target.astype(str))[0]
    if context.is_classification and np.bincount(target).min() < context.value("G01", "cv_folds"):
        counts = np.bincount(target)
        keep = np.isin(target, np.where(counts >= context.value("G01", "cv_folds"))[0])
        known_positions = np.where(known)[0][keep]
        target = pd.factorize(target[keep])[0]
    else:
        known_positions = np.where(known)[0]
    if context.is_classification and len(np.unique(target)) < 2:
        return
    task = context.task if len(np.unique(target)) > 2 or context.task == "regression" else "binary-classification"
    features = {
        column: encode_feature(context.typed.loc[rows, column], context.types[column])[known_positions]
        for column in columns
    }
    depth = context.value("G01", "tree_depth")
    folds = context.value("G01", "cv_folds")
    scores = Parallel(n_jobs=context.value("G01", "n_jobs"))(
        delayed(single_feature_score)(features[column], target, task, depth, folds) for column in columns
    )
    limit = context.value("G01", "min_r2") if context.task == "regression" else context.value("G01", "min_auc")
    metric = "r2" if context.task == "regression" else "auc"
    for column, score in zip(columns, scores):
        if not np.isnan(score) and score >= limit:
            evidence = {metric: round(score, 4), "model": f"tree depth {depth}, {folds}-fold CV", "sample_rows": len(target)}
            context.ledger.add("G01", [column], metric, evidence)

# G02: categories that map almost purely to one target value and cover most...
# rows
@register("G02", applies_to=category_types | {"boolean", "numeric_code"}, needs_target=True, cost="S")
def pure_categories(context, columns):
    if not context.is_classification:
        return
    target = context.typed[context.target].astype(str).where(context.typed[context.target].notna())
    min_rows = context.value("D05", "min_count")
    for column in columns:
        frame = pd.DataFrame({"value": context.typed[column], "target": target}).dropna()
        if frame.empty:
            continue
        table = pd.crosstab(frame["value"], frame["target"])
        sizes = table.sum(axis=1)
        purity = table.max(axis=1) / sizes
        pure = (purity >= context.value("G02", "min_purity")) & (sizes >= min_rows)
        covered = sizes[pure].sum() / sizes.sum()
        if covered >= context.value("G02", "min_row_share"):
            evidence = {"pure_levels": int(pure.sum()), "levels": len(table), "rows_covered": round(float(covered), 4)}
            context.ledger.add("G02", [column], "purity", evidence)

# G04: columns that are missing almost only for one class
@register("G04", applies_to=all_types, needs_target=True)
def missingness_predicts_target(context, columns):
    if not context.is_classification:
        return
    target = context.typed[context.target].astype(str).where(context.typed[context.target].notna())
    base = target.value_counts(normalize=True)
    for column in columns:
        missing = context.typed[column].isna() & target.notna()
        count = int(missing.sum())
        if count == 0 or count == int(target.notna().sum()):
            continue
        shares = target[missing].value_counts()
        top = shares.index[0]
        share = shares.iloc[0] / count
        if share < context.value("G04", "min_class_share") or base[top] >= context.value("G04", "min_class_share"):
            continue
        p_value = binomtest(int(shares.iloc[0]), count, float(base[top]), alternative="greater").pvalue
        if p_value <= context.value("C04", "max_p_value"):
            evidence = {"class": top, "share_in_missing": round(float(share), 4), "share_overall": round(float(base[top]), 4), "p_value": float(f"{p_value:.3g}")}
            context.ledger.add("G04", [column], "class_share", evidence, rows=mask_index(missing))
