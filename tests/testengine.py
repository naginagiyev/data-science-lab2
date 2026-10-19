import time
import numpy as np
import pandas as pd
from pathlib import Path
from modelling import engine
from settings import load_settings
from sklearn.pipeline import Pipeline
from tools.splitter import make_folds
from modelling.cv import encode_target
from operations.generic import GenericStage, TypingStep

# the preparation pipeline used in these tests (typing + generic stage)
def make_prep(kind):
    return Pipeline([("typing", TypingStep()), ("generic", GenericStage(kind))])

# a toy table as raw text with a numeric signal and a category
def toy_table(task, rows=600, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=rows)
    color = rng.choice(["red", "blue", "green"], rows)
    frame = pd.DataFrame({"signal": [f"{value:.4f}" for value in signal], "color": color, "noise": [f"{value:.3f}" for value in rng.normal(size=rows)]})
    if task == "regression":
        target = signal * 3 + (color == "red") * 2 + rng.normal(scale=0.3, size=rows)
        labels = [f"{value:.4f}" for value in target]
    elif task == "binary-classification":
        labels = np.where(signal + (color == "red") > 0.3, "yes", "no")
    else:
        labels = np.where(signal > 0.7, "high", np.where(signal < -0.7, "low", "mid"))
    frame["y"] = labels
    return frame

# trains the engine on a toy table and returns the model, summary, log and seconds
def run_engine(task, seconds, candidates=None, rows=600):
    frame = toy_table(task, rows)
    y, encoder = encode_target(frame["y"], task)
    features = frame.drop(columns="y")
    folds, _ = make_folds(frame, "y", task, "stratified" if task != "regression" else "random", None, 3, 0)
    settings = load_settings().engine
    if candidates:
        settings = settings.model_copy(update={"candidates": candidates})
    metric = {"regression": "rmse", "binary-classification": "roc_auc", "multi-class-classification": "log_loss"}[task]
    start = time.monotonic()
    model, summary, log = engine.train(features, y, task, metric, folds, make_prep, seconds, settings, 0, n_classes=len(encoder.classes_) if encoder else None)
    return model, summary, log, time.monotonic() - start, features

# the engine respects its time budget within 10% and beats the dummy model
def test_deadline_and_quality():
    model, summary, log, seconds, features = run_engine("binary-classification", 12)
    assert seconds <= 13.2
    assert summary["cv"]["lightgbm"] > summary["cv"]["dummy"] + 0.2
    assert model.predict_values(features.head(7)).shape == (7, 2)
    assert {entry["family"] for entry in log} >= {"dummy", "linear", "lightgbm", "catboost"}
    assert not Path("catboost_info").exists()

# regression and multi-class work; predictions have the right shape
def test_regression_and_multiclass():
    model, summary, log, seconds, features = run_engine("regression", 8, ["dummy", "linear", "lightgbm"])
    assert summary["cv"]["lightgbm"] < summary["cv"]["dummy"] / 2
    assert model.predict_values(features.head(4)).shape == (4,)
    model, summary, log, seconds, features = run_engine("multi-class-classification", 8, ["dummy", "lightgbm"])
    probabilities = model.predict_values(features.head(4))
    assert probabilities.shape == (4, 3) and np.allclose(probabilities.sum(axis=1), 1)

# a candidate that crashes is skipped and the dummy model still exists
def test_failing_candidate(monkeypatch):
    original = engine.fit_model
    def broken(family, *arguments, **options):
        if family in {"lightgbm", "catboost", "linear"}:
            raise RuntimeError("broken model")
        return original(family, *arguments, **options)
    monkeypatch.setattr(engine, "fit_model", broken)
    model, summary, log, seconds, features = run_engine("binary-classification", 6)
    assert summary["chosen"] == ["dummy"]
    assert model.predict_values(features.head(3)).shape == (3, 2)
