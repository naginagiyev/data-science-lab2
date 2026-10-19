import logging
import warnings
import numpy as np
import pandas as pd
from schemas import SplitInfo
from tools.columntypes import infer_types
from scipy.stats import chi2_contingency, spearmanr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold, train_test_split

logger = logging.getLogger("automl.split")
TIME_MIN_CORRELATION = 0.1
TIME_MAX_P_VALUE = 0.01
TIME_BINS = 10

# H01: the id columns whose values repeat (the same entity in several rows)
def repeated_ids(types, typed, target):
    columns = [column for column, kind in types.items() if kind == "id" and column != target]
    return [column for column in columns if typed[column].dropna().astype(str).duplicated().any()]

# the target as numbers (classes become sorted codes)
def target_numbers(typed, target, task):
    if task == "regression":
        return pd.to_numeric(typed[target], errors="coerce").astype(float)
    codes, _ = pd.factorize(typed[target].astype(str).where(typed[target].notna()), sort=True)
    return pd.Series(codes.astype(float), index=typed.index).where(codes >= 0)

# H02: the date columns over which the target changes (Spearman, or Cramer's V...
# over time bins for multi-class)
def time_trends(types, typed, target, task):
    found = []
    numbers = target_numbers(typed, target, task)
    for column in [column for column, kind in types.items() if kind == "datetime" and column != target]:
        frame = pd.DataFrame({"time": typed[column], "target": numbers}).dropna()
        if len(frame) < 20 or frame["time"].nunique() < 2:
            continue
        times = frame["time"].astype("int64")
        if task == "multi-class-classification":
            table = pd.crosstab(pd.qcut(times.rank(method="first"), TIME_BINS, labels=False), frame["target"])
            chi2, p_value = chi2_contingency(table)[:2]
            strength = float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))
        else:
            strength, p_value = spearmanr(times, frame["target"])
        if abs(strength) >= TIME_MIN_CORRELATION and p_value <= TIME_MAX_P_VALUE:
            found.append(column)
    return found

# the split advice on all rows: repeated ids need a group split, a target...
# trend over a date needs a time split; otherwise classification is...
# stratified and regression random; returns (strategy, key, reason, notes)
def split_advice(raw, target, task):
    notes = []
    try:
        types, typed, _ = infer_types(raw)
        groups = repeated_ids(types, typed, target)
        if groups:
            return "group", groups[-1], f"Rows repeat the same `{groups[-1]}`; keep each group in one split.", notes
        times = time_trends(types, typed, target, task)
        if times:
            return "time", times[-1], f"The target changes over `{times[-1]}`; split by time.", notes
    except Exception as error:
        logger.warning(f"split advice failed: {type(error).__name__}: {error}")
        notes.append(f"split advice failed: {error}")
    if task != "regression":
        return "stratified", target, "Classification task; keep class shares equal in each split.", notes
    return "random", None, "No group, time or class structure found.", notes

# the class labels used for stratification (stripped text)
def class_labels(frame, target):
    return frame[target].astype(str).str.strip().to_numpy()

# turns a time key into sortable values: dates when most values parse, else...
# numbers; values that do not parse sort first
def time_order(values):
    text = pd.Series(values).astype(str).str.strip()
    numbers = pd.to_numeric(text, errors="coerce")
    if numbers.notna().mean() >= 0.9:
        return numbers.fillna(-np.inf).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dates = pd.to_datetime(text, errors="coerce", format="mixed")
    return dates.fillna(pd.Timestamp.min).to_numpy()

# holds out the locked test rows by the advice (demo mode); returns train and...
# test positions, the strategy actually used and warnings
def holdout(raw, target, task, strategy, key, test_size, seed):
    positions = np.arange(len(raw))
    notes = []
    if strategy == "group" and key in raw.columns and raw[key].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train, test = next(splitter.split(positions, groups=raw[key].astype(str).to_numpy()))
        return np.sort(train), np.sort(test), "group", notes
    if strategy == "time" and key in raw.columns:
        order = np.argsort(time_order(raw[key]), kind="stable")
        cut = int(round(len(raw) * (1 - test_size)))
        return np.sort(order[:cut]), np.sort(order[cut:]), "time", notes
    if strategy in {"group", "time"}:
        notes.append(f"{strategy} split on '{key}' was not possible; used a {'random' if task == 'regression' else 'stratified'} split")
        strategy = "random" if task == "regression" else "stratified"
    labels = class_labels(raw, target) if task != "regression" else None
    if labels is not None and pd.Series(labels).value_counts().min() >= 2:
        train, test = train_test_split(positions, test_size=test_size, random_state=seed, stratify=labels)
        return np.sort(train), np.sort(test), "stratified", notes
    if labels is not None:
        notes.append("a class has fewer than 2 rows; the test split is random, not stratified")
    train, test = train_test_split(positions, test_size=test_size, random_state=seed)
    return np.sort(train), np.sort(test), "random", notes

# makes CV folds on the training frame with the same strategy: group folds,...
# ordered time folds, stratified folds (folds = min(cv_folds, smallest class))...
# or plain shuffled folds; returns the validation positions of each fold
def make_folds(frame, target, task, strategy, key, n_folds, seed):
    positions = np.arange(len(frame))
    notes = []
    if strategy == "group" and key in frame.columns:
        groups = frame[key].astype(str).to_numpy()
        count = min(n_folds, len(set(groups)))
        if count >= 2:
            splitter = GroupKFold(n_splits=count, shuffle=True, random_state=seed)
            return [np.sort(valid).tolist() for _, valid in splitter.split(positions, groups=groups)], notes
        notes.append(f"too few groups in '{key}' for group folds")
    if strategy == "time" and key in frame.columns:
        order = np.argsort(time_order(frame[key]), kind="stable")
        return [np.sort(order[valid]).tolist() for _, valid in KFold(n_splits=min(n_folds, len(frame))).split(order)], notes
    if task != "regression":
        labels = class_labels(frame, target)
        smallest = pd.Series(labels).value_counts().min()
        if smallest >= 2:
            splitter = StratifiedKFold(n_splits=max(2, min(n_folds, smallest)), shuffle=True, random_state=seed)
            return [np.sort(valid).tolist() for _, valid in splitter.split(positions, labels)], notes
        notes.append("a class has fewer than 2 training rows; folds are not stratified")
    splitter = KFold(n_splits=max(2, min(n_folds, len(frame))), shuffle=True, random_state=seed)
    return [np.sort(valid).tolist() for _, valid in splitter.split(positions)], notes

# the split stage: advice, locked test rows (demo mode only) and CV folds on...
# the training rows; fold positions refer to the training frame
def split(raw, target, task, mode, test_size, cv_folds, seed):
    strategy, key, reason, notes = split_advice(raw, target, task)
    if mode == "demo":
        train, test, used, holdout_notes = holdout(raw, target, task, strategy, key, test_size, seed)
        notes += holdout_notes
    else:
        train, test, used = np.arange(len(raw)), np.array([], dtype=int), strategy
    train_frame = raw.iloc[train].reset_index(drop=True)
    fold_strategy = used if used in {"group", "time"} else strategy
    folds, fold_notes = make_folds(train_frame, target, task, fold_strategy, key, cv_folds, seed)
    info = SplitInfo(strategy=used, key=key if used in {"group", "time"} else None, test_rows=test.tolist(), folds=folds, reason=reason, warnings=notes + fold_notes)
    return info, train_frame, raw.iloc[test].reset_index(drop=True)
