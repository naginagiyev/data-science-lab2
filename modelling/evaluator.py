import time
import numpy as np
import pandas as pd
from operations.compiler import target_steps
from tools.metrics import align_probabilities, compute_scores, oriented

# the true test values: stripped label text, or floats for regression
def test_truth(values, task):
    text = pd.Series(values).astype(str).str.strip()
    if task == "regression":
        return pd.to_numeric(text, errors="coerce").to_numpy(dtype=float)
    return text.to_numpy(dtype=object)

# scores probabilities or values on the test rows; a test class the model...
# never saw gets a zero column, so the metric stays defined
def score(task, truth, values, classes):
    if task == "regression":
        keep = np.isfinite(truth)
        return compute_scores(task, truth[keep], np.asarray(values)[keep])
    labels = sorted(set(classes) | set(truth.tolist()))
    return compute_scores(task, truth, align_probabilities(values, classes, labels), labels)

# the prediction of the dummy model from the training labels
def dummy_values(train_truth, task, rows, classes):
    if task == "regression":
        return np.full(rows, np.nanmean(train_truth))
    shares = pd.Series(train_truth).value_counts(normalize=True)
    return np.tile([shares.get(label, 0.0) for label in classes], (rows, 1))

# target values with the plan's class spellings mapped (classification)
def mapped(fitted, values):
    if fitted.task != "regression" and fitted.plan is not None and fitted.plan.target_steps:
        return target_steps(fitted.plan).labels(values)
    return values

# uses the locked test set once: scores of the fitted model and of the dummy...
# model, at full precision
def evaluate(fitted, test_frame, train_target):
    truth = test_truth(mapped(fitted, test_frame[fitted.target]), fitted.task)
    train_target = mapped(fitted, train_target)
    values = fitted.raw_values(test_frame.drop(columns=[fitted.target]))
    classes = fitted.classes
    result = score(fitted.task, truth, values, classes)
    dummy = score(fitted.task, truth, dummy_values(test_truth(train_target, fitted.task), fitted.task, len(truth), classes), classes)
    return {"metric": result["metric"], "test": result["score"], "test_scores": result["scores"], "notes": result["notes"], "dummy_test": dummy["score"], "rows": int(len(truth))}

# permutation importance of the raw columns on at most 2,000 test rows: how...
# much the test score gets worse when one column is shuffled (mean of two...
# shuffles); it stops after max_seconds, and the top columns come first
def importance(fitted, test_frame, top=10, rows=2000, repeats=2, max_seconds=30, seed=0):
    start = time.monotonic()
    frame = test_frame.sample(min(rows, len(test_frame)), random_state=seed)
    truth = test_truth(mapped(fitted, frame[fitted.target]), fitted.task)
    features = frame.drop(columns=[fitted.target])
    base = score(fitted.task, truth, fitted.raw_values(features), fitted.classes)
    rng = np.random.default_rng(seed)
    losses = {}
    for column in features.columns:
        if time.monotonic() - start > max_seconds:
            break
        drops = []
        for _ in range(repeats):
            shuffled = features.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            result = score(fitted.task, truth, fitted.raw_values(shuffled), fitted.classes)
            drops.append(oriented(base["metric"], base["score"]) - oriented(result["metric"], result["score"]))
        losses[column] = float(np.mean(drops))
    ranked = sorted(losses.items(), key=lambda item: -item[1])[:top]
    return {"columns": [[str(name), round(value, 5)] for name, value in ranked], "rows": int(len(frame)), "complete": len(losses) == features.shape[1]}
