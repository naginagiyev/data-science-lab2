import math
import numpy as np
import pandas as pd
from benchmark.systems.common import label_key
from tools.metrics import align_probabilities, compute_scores

# reads one column of a CSV file as raw text
def read_column(path, column):
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=[column])[column]

# returns the true values: label keys for classification, floats for regression
def true_values(labels, task):
    if task == "regression":
        return pd.to_numeric(labels, errors="raise").to_numpy(dtype=float)
    return np.array([label_key(value) for value in labels], dtype=object)

# the classes used for scoring: every label key of the training and test rows,...
# sorted, so a class that appears only in the test rows still has a column
def scoring_classes(train_labels, test_labels):
    return sorted(set(label_key(value) for value in train_labels) | set(test_labels))

# the prediction of the dummy model: training class shares or the training mean
def dummy_prediction(train_labels, task, rows, classes):
    if task == "regression":
        return np.full(rows, pd.to_numeric(pd.Series(train_labels).astype(str).str.strip(), errors="coerce").mean())
    keys = pd.Series([label_key(value) for value in train_labels])
    shares = keys.value_counts(normalize=True)
    probabilities = np.array([shares.get(label, 0.0) for label in classes])
    return np.tile(probabilities, (rows, 1))

# reads and checks a predictions file; returns the prediction in scoring form...
# or raises ValueError with the reason (missing columns, wrong rows, NaN, ...)
def read_prediction(path, task, rows, classes):
    if not path.exists():
        raise ValueError("no predictions file")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(frame) != rows:
        raise ValueError(f"{len(frame)} prediction rows for {rows} test rows")
    if task == "regression":
        if "prediction" not in frame:
            raise ValueError("no 'prediction' column")
        values = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("missing or infinite predictions")
        return values
    columns = {label_key(column): column for column in frame.columns if column != "label"}
    found = [label for label in classes if label in columns]
    if found:
        values = frame[[columns[label] for label in found]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("missing, infinite or negative probabilities")
        return align_probabilities(values, found, classes)
    if "label" in frame:
        hard = np.zeros((rows, len(classes)))
        for row, value in enumerate(frame["label"]):
            key = label_key(value)
            if key in classes:
                hard[row, classes.index(key)] = 1.0
        return align_probabilities(hard, classes, classes)
    raise ValueError("no probability columns and no 'label' column")

# scores one predictions file against the test labels; a failure gets the...
# dummy score (as in AMLB); returns status, error, score, dummy score and...
# the secondary metrics
def score_cell(task, train_file, labels_file, predictions_file, target):
    train_labels = read_column(train_file, target)
    labels = read_column(labels_file, target)
    y_true = true_values(labels, task)
    classes = None if task == "regression" else scoring_classes(train_labels, y_true)
    dummy = compute_scores(task, y_true, dummy_prediction(train_labels, task, len(y_true), classes), classes)
    try:
        prediction = read_prediction(predictions_file, task, len(y_true), classes)
        result = compute_scores(task, y_true, prediction, classes)
        status, error = "ok", ""
    except ValueError as problem:
        result, status, error = dummy, "failed", str(problem)
    if status == "ok" and math.isnan(result["score"]):
        error = result["notes"].get(result["metric"], "metric undefined")
    return {
        "status": status, "error": error, "metric": result["metric"], "score": result["score"],
        "dummy_score": dummy["score"], "secondary": result["scores"],
    }
