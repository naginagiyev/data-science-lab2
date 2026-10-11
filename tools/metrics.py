import json
import math
import numpy as np
from paths import metrics_file
from sklearn import metrics as sk

EPSILON = 1e-15
METRICS = json.loads(metrics_file.read_text(encoding="utf-8"))

# returns the fixed primary metric of a task (never chosen by the LLM)
def primary_metric(task):
    return METRICS[task]["primary"]

# true when a smaller value of the metric is better (losses and errors)
def lower_is_better(metric):
    return metric in METRICS["lower_is_better"]

# turns a score into "higher is better" form, so scores of any metric can be...
# compared with ">"
def oriented(metric, value):
    return -value if lower_is_better(metric) else value

# puts the probability columns of a model in the order of all classes; a class...
# the model never saw gets zeros, then every row is renormalized
def align_probabilities(probabilities, model_classes, all_classes):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1 - probabilities, probabilities])
    aligned = np.zeros((probabilities.shape[0], len(all_classes)))
    positions = {label: index for index, label in enumerate(all_classes)}
    for column, label in enumerate(model_classes):
        aligned[:, positions[label]] = probabilities[:, column]
    aligned = np.clip(aligned, 0, None) + EPSILON
    return aligned / aligned.sum(axis=1, keepdims=True)

# multi-class log-loss with all training classes as labels; probabilities are...
# clipped at 1e-15 and renormalized
def multiclass_log_loss(y_true, probabilities, classes):
    clipped = np.clip(probabilities, EPSILON, 1)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return float(sk.log_loss(y_true, clipped, labels=list(classes)))

# one-vs-rest macro ROC-AUC over the classes that appear in y_true with both...
# positive and negative rows (a missing class is skipped, not an error)
def multiclass_roc_auc(y_true, probabilities, classes):
    y_true = np.asarray(y_true)
    values = []
    for index, label in enumerate(classes):
        positive = y_true == label
        if 0 < positive.sum() < len(y_true):
            values.append(sk.roc_auc_score(positive, probabilities[:, index]))
    return float(np.mean(values)) if len(values) >= 2 else math.nan

# runs one metric and turns an error into NaN with a reason
def safe_metric(name, function, notes):
    try:
        return float(function())
    except ValueError as error:
        notes[name] = str(error)
        return math.nan

# scores of a binary task; probabilities are aligned to classes, the positive...
# class is classes[1]
def binary_scores(y_true, probabilities, classes, notes):
    y_true = np.asarray(y_true)
    positive = y_true == classes[1]
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    scores = {}
    if positive.all() or not positive.any():
        notes["roc_auc"] = "only one class in y_true"
        scores["roc_auc"] = math.nan
        scores["pr_auc"] = math.nan
    else:
        scores["roc_auc"] = safe_metric("roc_auc", lambda: sk.roc_auc_score(positive, probabilities[:, 1]), notes)
        scores["pr_auc"] = safe_metric("pr_auc", lambda: sk.average_precision_score(positive, probabilities[:, 1]), notes)
    scores["log_loss"] = multiclass_log_loss(y_true, probabilities, classes)
    scores["accuracy"] = float(sk.accuracy_score(y_true, predicted))
    scores["balanced_accuracy"] = float(sk.balanced_accuracy_score(y_true, predicted))
    scores["f1"] = float(sk.f1_score(positive, predicted == classes[1], zero_division=0))
    return scores

# scores of a multi-class task; probabilities are aligned to classes
def multiclass_scores(y_true, probabilities, classes, notes):
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    auc = multiclass_roc_auc(y_true, probabilities, classes)
    if math.isnan(auc):
        notes["roc_auc_ovr"] = "fewer than two classes in y_true"
    return {
        "log_loss": multiclass_log_loss(y_true, probabilities, classes),
        "roc_auc_ovr": auc,
        "accuracy": float(sk.accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(sk.balanced_accuracy_score(y_true, predicted)),
        "f1_macro": float(sk.f1_score(y_true, predicted, average="macro", zero_division=0)),
    }

# scores of a regression task; MAPE uses only rows with a non-zero target
def regression_scores(y_true, prediction, notes):
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    nonzero = y_true != 0
    if nonzero.any():
        mape = float(np.mean(np.abs((y_true[nonzero] - prediction[nonzero]) / y_true[nonzero])))
    else:
        mape = math.nan
        notes["mape"] = "every target value is zero"
    r2 = safe_metric("r2", lambda: sk.r2_score(y_true, prediction), notes) if len(y_true) > 1 else math.nan
    return {
        "rmse": float(np.sqrt(np.mean((y_true - prediction) ** 2))),
        "mae": float(np.mean(np.abs(y_true - prediction))),
        "r2": r2,
        "mape": mape,
    }

# computes the primary and all reported metrics at full precision; for...
# classification, prediction is a probability matrix in the order of classes
def compute_scores(task, y_true, prediction, classes=None):
    notes = {}
    if task == "regression":
        scores = regression_scores(y_true, prediction, notes)
    elif task == "binary-classification":
        scores = binary_scores(y_true, np.asarray(prediction, dtype=float), list(classes), notes)
    else:
        scores = multiclass_scores(y_true, np.asarray(prediction, dtype=float), list(classes), notes)
    metric = primary_metric(task)
    return {"metric": metric, "score": scores[metric], "scores": scores, "notes": notes}
