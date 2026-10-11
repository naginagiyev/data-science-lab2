import numpy as np
import pandas as pd
from tools.metrics import align_probabilities, compute_scores

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

# uses the locked test set once: scores of the fitted model and of the dummy...
# model, at full precision
def evaluate(fitted, test_frame, train_target):
    truth = test_truth(test_frame[fitted.target], fitted.task)
    values = fitted.raw_values(test_frame.drop(columns=[fitted.target]))
    classes = fitted.classes
    result = score(fitted.task, truth, values, classes)
    dummy = score(fitted.task, truth, dummy_values(test_truth(train_target, fitted.task), fitted.task, len(truth), classes), classes)
    return {"metric": result["metric"], "test": result["score"], "test_scores": result["scores"], "notes": result["notes"], "dummy_test": dummy["score"], "rows": int(len(truth))}
