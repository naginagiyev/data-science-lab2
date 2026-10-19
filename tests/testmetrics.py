import math
import numpy as np
from sklearn import metrics as sk
from tools.metrics import align_probabilities, compute_scores, lower_is_better, oriented, primary_metric

# the primary metric comes from the task, and losses are turned around
def test_primary_metric_and_orientation():
    assert primary_metric("binary-classification") == "roc_auc"
    assert primary_metric("multi-class-classification") == "log_loss"
    assert primary_metric("regression") == "rmse"
    assert lower_is_better("rmse") and not lower_is_better("roc_auc")
    assert oriented("log_loss", 0.3) == -0.3 and oriented("roc_auc", 0.8) == 0.8

# binary scores agree with scikit-learn and use probabilities, not labels
def test_binary_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = np.where(rng.random(200) > 0.6, "Yes", "No")
    positive = rng.random(200) * 0.5 + (y_true == "Yes") * 0.4
    probabilities = np.column_stack([1 - positive, positive])
    result = compute_scores("binary-classification", y_true, probabilities, ["No", "Yes"])
    assert result["metric"] == "roc_auc"
    assert math.isclose(result["score"], sk.roc_auc_score(y_true == "Yes", positive))
    assert math.isclose(result["scores"]["pr_auc"], sk.average_precision_score(y_true == "Yes", positive))
    assert math.isclose(result["scores"]["log_loss"], sk.log_loss(y_true, probabilities, labels=["No", "Yes"]))
    assert math.isclose(result["scores"]["f1"], sk.f1_score(y_true == "Yes", positive > 0.5))

# a test set with one class gives NaN ROC-AUC with a reason, not a crash
def test_binary_one_class_is_nan():
    probabilities = np.array([[0.2, 0.8], [0.4, 0.6]])
    result = compute_scores("binary-classification", ["Yes", "Yes"], probabilities, ["No", "Yes"])
    assert math.isnan(result["score"])
    assert "roc_auc" in result["notes"]
    assert result["scores"]["accuracy"] == 1.0

# a class missing from a training fold gets a zero column, and log-loss uses...
# all training classes as labels
def test_multiclass_missing_class():
    classes = ["a", "b", "c"]
    model_probabilities = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    aligned = align_probabilities(model_probabilities, ["a", "c"], classes)
    assert aligned.shape == (3, 3)
    assert np.allclose(aligned.sum(axis=1), 1)
    assert aligned[:, 1].max() < 1e-12
    result = compute_scores("multi-class-classification", ["a", "c", "b"], aligned, classes)
    assert math.isfinite(result["score"])
    clipped = np.clip(aligned, 1e-15, 1)
    expected = sk.log_loss(["a", "c", "b"], clipped / clipped.sum(axis=1, keepdims=True), labels=classes)
    assert math.isclose(result["score"], expected)
    assert math.isfinite(result["scores"]["roc_auc_ovr"])

# a one-dimensional positive-class vector becomes two aligned columns
def test_align_one_column():
    aligned = align_probabilities([0.9, 0.1], [0, 1], [0, 1])
    assert np.allclose(aligned[:, 1], [0.9, 0.1])

# regression scores agree with scikit-learn, and MAPE skips zero targets
def test_regression_scores():
    y_true = np.array([0.0, 2.0, 4.0, 8.0])
    prediction = np.array([1.0, 2.5, 3.0, 8.0])
    result = compute_scores("regression", y_true, prediction)
    assert math.isclose(result["score"], math.sqrt(sk.mean_squared_error(y_true, prediction)))
    assert math.isclose(result["scores"]["mae"], sk.mean_absolute_error(y_true, prediction))
    assert math.isclose(result["scores"]["r2"], sk.r2_score(y_true, prediction))
    assert math.isclose(result["scores"]["mape"], np.mean([0.25, 0.25, 0.0]))
