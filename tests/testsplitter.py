import numpy as np
import pandas as pd
from tools.splitter import holdout, make_folds, split, split_advice

# a toy classification frame with a repeated entity id and a date
def toy_frame(rows=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "house": [str(value) for value in rng.integers(0, 60, rows)],
        "sold": pd.date_range("2020-01-01", periods=rows).strftime("%Y-%m-%d"),
        "x": [str(value) for value in rng.normal(size=rows)],
        "y": rng.choice(["a", "b"], rows, p=[0.8, 0.2]),
    })

# a stratified hold-out keeps the class shares and folds use all training rows
def test_stratified():
    frame = toy_frame()
    train, test, used, notes = holdout(frame, "y", "binary-classification", "stratified", "y", 0.2, 0)
    assert used == "stratified" and len(test) == 40 and not set(train) & set(test)
    assert abs((frame.iloc[test]["y"] == "b").mean() - (frame["y"] == "b").mean()) < 0.03
    folds, _ = make_folds(frame.iloc[train].reset_index(drop=True), "y", "binary-classification", "stratified", "y", 5, 0)
    assert len(folds) == 5 and sorted(sum(folds, [])) == list(range(len(train)))

# a class with one row gives a random split and stratified folds fall back...
# to plain folds, both with a warning
def test_tiny_class_fallback():
    frame = toy_frame()
    frame.loc[0, "y"] = "c"
    train, test, used, notes = holdout(frame, "y", "multi-class-classification", "stratified", "y", 0.2, 0)
    assert used == "random" and notes
    folds, fold_notes = make_folds(frame, "y", "multi-class-classification", "stratified", "y", 5, 0)
    assert len(folds) == 5 and fold_notes

# folds = min(cv_folds, smallest class count), at least 2
def test_fold_count_follows_smallest_class():
    frame = pd.DataFrame({"x": [str(i) for i in range(40)], "y": ["a"] * 37 + ["b"] * 3})
    folds, _ = make_folds(frame, "y", "binary-classification", "stratified", "y", 5, 0)
    assert len(folds) == 3

# group splits never put one group on both sides, in the hold-out and in folds
def test_group_split():
    frame = toy_frame()
    train, test, used, _ = holdout(frame, "y", "binary-classification", "group", "house", 0.2, 0)
    assert used == "group"
    assert not set(frame.iloc[train]["house"]) & set(frame.iloc[test]["house"])
    train_frame = frame.iloc[train].reset_index(drop=True)
    folds, _ = make_folds(train_frame, "y", "binary-classification", "group", "house", 5, 0)
    for valid in folds:
        rest = sorted(set(range(len(train_frame))) - set(valid))
        assert not set(train_frame.iloc[valid]["house"]) & set(train_frame.iloc[rest]["house"])

# a time split puts the latest rows in the test set, and folds are blocks in time
def test_time_split():
    frame = toy_frame().sample(frac=1, random_state=1).reset_index(drop=True)
    train, test, used, _ = holdout(frame, "y", "binary-classification", "time", "sold", 0.2, 0)
    assert used == "time"
    assert frame.iloc[test]["sold"].min() > frame.iloc[train]["sold"].max()
    folds, _ = make_folds(frame.iloc[train].reset_index(drop=True), "y", "binary-classification", "time", "sold", 4, 0)
    dates = frame.iloc[train].reset_index(drop=True)["sold"]
    assert dates.iloc[folds[0]].max() < dates.iloc[folds[1]].min()

# regression uses plain shuffled folds; benchmark mode keeps every row for training
def test_regression_and_benchmark_mode():
    frame = toy_frame()
    frame["y"] = [str(value) for value in range(len(frame))]
    info, train, test = split(frame, "y", "regression", "benchmark", 0.2, 5, 0)
    assert info.test_rows == [] and len(train) == len(frame) and len(test) == 0
    assert len(info.folds) == 5

# repeated ids give a group split; a target that follows a date gives a time...
# split; otherwise classification is stratified
def test_split_advice():
    frame = toy_frame()
    frame["house"] = [str(1000 + index) for index in range(len(frame))]
    frame.loc[1, "house"] = "1000"
    assert split_advice(frame, "y", "binary-classification")[:2] == ("group", "house")
    trend = toy_frame().drop(columns=["house"])
    trend["y"] = [str(index + value) for index, value in enumerate(np.random.default_rng(0).normal(size=len(trend)))]
    assert split_advice(trend, "y", "regression")[:2] == ("time", "sold")
    assert split_advice(toy_frame().drop(columns=["sold"]), "y", "binary-classification")[0] == "stratified"
