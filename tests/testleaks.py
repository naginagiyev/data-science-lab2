import json
import numpy as np
import pandas as pd
from audit.core import run_audit
from benchmark import leaks, splits, materialize

# a toy classification and regression table as raw text
def toy(rows=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    return pd.DataFrame({
        "x": [f"{value:.3f}" for value in x],
        "z": [f"{value:.3f}" for value in rng.normal(size=rows)],
        "label": np.where(x + rng.normal(scale=1.5, size=rows) > 0.8, "yes", "no"),
        "amount": [f"{value:.2f}" for value in 100 + 30 * x + rng.normal(scale=10, size=rows)],
    })

# strong and moderate flags mark 98% and 80% of the positive rows and no...
# negative row
def test_flag_agreement(monkeypatch):
    raw = toy()
    for strength, low, high in [("strong", 0.95, 1.0), ("moderate", 0.72, 0.88)]:
        monkeypatch.setitem(leaks.LEAKS, "toy", {"column": "refund_issued", "style": "flag", "strength": strength})
        leaked, spec = leaks.add_leak(raw.drop(columns="amount"), "toy", "label", "binary-classification", 0)
        positive = leaked["label"] == "yes"
        share = (leaked.loc[positive, "refund_issued"] == "yes").mean()
        assert low <= share <= high, (strength, share)
        assert (leaked.loc[~positive, "refund_issued"] == "no").all()
        assert spec["agreement"] == leaks.AGREEMENT[strength]

# regression leaks: strong is almost the target, moderate correlates about 0.8
def test_amount_leaks(monkeypatch):
    raw = toy().drop(columns="label")
    target = raw["amount"].astype(float)
    for strength, low, high in [("strong", 0.99, 1.0), ("moderate", 0.7, 0.9)]:
        monkeypatch.setitem(leaks.LEAKS, "toy", {"column": "claim_paid_amount", "style": "amount", "strength": strength})
        leaked, _ = leaks.add_leak(raw, "toy", "amount", "regression", 0)
        correlation = np.corrcoef(leaked["claim_paid_amount"].astype(float), target)[0, 1]
        assert low <= correlation <= high, (strength, correlation)

# the date and grade styles carry the label as well
def test_date_and_grade_styles():
    raw = toy()
    rng = np.random.default_rng(0)
    dates = leaks.class_leak(raw["label"], "date", 0.98, rng)
    assert ((dates != "") == (raw["label"] == "yes")).mean() > 0.95
    grades = leaks.class_leak(pd.Series(["a", "b", "c"] * 100), "grade", 0.98, rng)
    assert set(grades) == {"grade_1", "grade_2", "grade_3"}

# G01 (one feature predicts the target) fires for a strong leak
def test_g01_finds_strong_leak(monkeypatch):
    monkeypatch.setitem(leaks.LEAKS, "toy", {"column": "account_closed_flag", "style": "flag", "strength": "strong"})
    leaked, _ = leaks.add_leak(toy().drop(columns="amount"), "toy", "label", "binary-classification", 0)
    report = run_audit(leaked, "label", "binary-classification")
    assert any(issue.code == "G01" and issue.columns == ["account_closed_flag"] for issue in report.issues)

# Group D files: the leak is empty in the deployment test file and kept in...
# the naive file; truth.json records the leak
def test_group_d_files(tmp_path, monkeypatch):
    raw = toy().drop(columns="amount")
    spec = {"name": "toy", "group": "A", "target": "label", "task": "binary-classification", "sample_rows": None}
    monkeypatch.setitem(leaks.LEAKS, "toy", {"column": "refund_issued", "style": "flag", "strength": "strong"})
    monkeypatch.setattr(materialize, "benchmark_files_dir", tmp_path)
    monkeypatch.setattr(splits, "benchmark_splits_dir", tmp_path / "splits")
    monkeypatch.setattr(materialize, "dataset_spec", lambda name: spec)
    monkeypatch.setattr(materialize, "load_dataset", lambda name: raw)
    folder = materialize.materialize("toy", 0, 0, group="D")
    deployment = pd.read_csv(folder / "test_features.csv", dtype=str, keep_default_na=False)
    naive = pd.read_csv(folder / "test_features_naive.csv", dtype=str, keep_default_na=False)
    assert (deployment["refund_issued"] == "").all() and set(naive["refund_issued"]) == {"yes", "no"}
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
    assert truth["leak"]["column"] == "refund_issued" and truth["group"] == "D"
