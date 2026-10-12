import pytest
import numpy as np
import pandas as pd
import modelling.backends
from modelling.cv import encode_target
from modelling.backends import BackendError, parquet_frame, train_backend

# a small table: two numbers, one category and a yes/no target
def small_table(rows=300, seed=0):
    rng = np.random.default_rng(seed)
    amount = rng.normal(0, 1, rows)
    group = rng.choice(["a", "b", "c"], rows)
    target = np.where(amount + (group == "a") + rng.normal(0, 0.5, rows) > 0.5, "Yes", "No")
    return pd.DataFrame({"amount": amount.astype(str), "other": rng.normal(0, 1, rows).astype(str), "group": group}), target

# numbers stay numbers, other values become text, missing stays missing
def test_parquet_frame():
    frame = parquet_frame(pd.DataFrame({1: [1.5, np.nan], "b": ["x", np.nan], "c": [1, "y"]}))
    assert list(frame.columns) == ["1", "b", "c"] and frame["1"].dtype == float
    assert frame["b"].tolist() == ["x", None] and frame["c"].tolist() == ["1", "y"]

# a missing backend environment is a clear error (the caller falls back to the...
# own engine)
def test_missing_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(modelling.backends, "PYTHONS", {"autogluon": tmp_path / "python.exe"})
    monkeypatch.setattr(modelling.backends, "backend_models_dir", tmp_path / "b")
    frame, target = small_table(50)
    y, _ = encode_target(target, "binary-classification")
    with pytest.raises(BackendError, match="environment is missing"):
        train_backend("autogluon", None, frame, y, "binary-classification", "roc_auc", 30, 0, 2)

# the real backends fit on prepared rows and predict class probabilities for...
# raw rows in the order of the label encoder
@pytest.mark.slow
@pytest.mark.parametrize("backend", ["flaml", "autogluon"])
def test_backend_fit_predict(backend, monkeypatch, tmp_path):
    monkeypatch.setattr(modelling.backends, "backend_models_dir", tmp_path / "b")
    frame, target = small_table()
    y, encoder = encode_target(target, "binary-classification")
    model, summary = train_backend(backend, None, frame, y, "binary-classification", "roc_auc", 40, 0, 2)
    values = model.predict_values(frame.head(20))
    assert values.shape == (20, 2) and np.allclose(values.sum(axis=1), 1, atol=1e-4)
    assert summary["chosen"] == [backend] and 0.5 < summary["cv"][backend] <= 1
    assert ((values[:, 1] > 0.5) == (y[:20] == 1)).mean() >= 0.7
