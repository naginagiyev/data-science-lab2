import logging
import warnings
import numpy as np
import pandas as pd
from edatool import presplit
from schemas import SplitInfo
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold, train_test_split

logger = logging.getLogger("automl.split")

# asks edatool how to split; if presplit fails, classification is stratified...
# and regression random
def split_advice(raw, target, task):
    try:
        advice = presplit(raw, target, task)
        return advice.strategy, advice.key, advice.reason, []
    except Exception as error:
        logger.warning(f"presplit failed: {type(error).__name__}: {error}")
        strategy = "random" if task == "regression" else "stratified"
        return strategy, target if strategy == "stratified" else None, "presplit failed; default split", [f"presplit failed: {error}"]

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
