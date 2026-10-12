import os
import json
import math
import shutil
import numpy as np
import pandas as pd
from sklearn import metrics as sk
from benchmark import runner, splits, materialize
from benchmark.scoring import read_prediction, score_cell
from benchmark.systems.common import label_key, write_predictions
from benchmark.splits import dataset_split, fixed_sample, outer_split

HANDMADE_ADAPTER = """
import sys, shutil
from pathlib import Path
arguments = sys.argv
out = Path(arguments[arguments.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
source = Path(arguments[arguments.index("--handmade") + 1])
shutil.copy(source, out / "predictions0.csv")
(out / "meta.json").write_text('{"ok": true, "llm_calls": 2, "cost_usd": 0.01}')
"""

# integer-valued numbers and text labels get the same key in every form
def test_label_keys():
    assert label_key("1") == label_key("1.0") == label_key(1) == label_key(1.0) == "1"
    assert label_key(" Yes ") == "Yes"
    assert label_key("2.5") == "2.5"

# the outer split is 80/20, stratified, and the sample keeps the class shares
def test_splits():
    raw = pd.DataFrame({"x": [str(i) for i in range(100)], "y": ["a"] * 80 + ["b"] * 20})
    train, test = outer_split(raw, "y", "binary-classification", seed=0)
    assert len(test) == 20 and not set(train) & set(test)
    assert raw.iloc[test]["y"].value_counts()["b"] == 4
    sample = fixed_sample(raw, "y", "binary-classification", 50)
    assert len(sample) == 50 and sample["y"].value_counts()["b"] == 10

# a saved split is reused, and a split of other data is refused
def test_saved_split(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "benchmark_splits_dir", tmp_path)
    raw = pd.DataFrame({"x": [str(i) for i in range(30)], "y": ["a", "b"] * 15})
    first = dataset_split("toy", raw, "y", "binary-classification", 1)
    assert dataset_split("toy", raw, "y", "binary-classification", 1) == first
    raw.loc[0, "x"] = "changed"
    try:
        dataset_split("toy", raw, "y", "binary-classification", 1)
        raise AssertionError("a split of other data must fail")
    except ValueError:
        pass

# probability columns are matched by label key, missing ones become zeros, and...
# a file with only labels is scored as hard probabilities
def test_read_prediction(tmp_path):
    path = tmp_path / "p.csv"
    write_predictions(path, "multi-class-classification", [[0.7, 0.3], [0.1, 0.9]], [3.0, 5.0])
    values = read_prediction(path, "multi-class-classification", 2, ["3", "4", "5"])
    assert np.allclose(values[:, 0], [0.7, 0.1], atol=1e-9) and values[:, 1].max() < 1e-9
    pd.DataFrame({"label": ["5", "3"]}).to_csv(path, index=False)
    hard = read_prediction(path, "multi-class-classification", 2, ["3", "4", "5"])
    assert hard[0, 2] > 0.99 and hard[1, 0] > 0.99

# a missing or broken predictions file fails and gets the dummy score
def test_failure_gets_dummy(tmp_path):
    train = tmp_path / "train.csv"
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"x": range(6), "y": [1.0, 2, 3, 4, 5, 6]}).to_csv(train, index=False)
    pd.DataFrame({"y": [2.0, 4.0]}).to_csv(labels, index=False)
    result = score_cell("regression", train, labels, tmp_path / "missing.csv", "y")
    assert result["status"] == "failed" and result["score"] == result["dummy_score"]
    assert math.isclose(result["dummy_score"], math.sqrt(1.25))
    pd.DataFrame({"prediction": [1.0, "nan"]}).to_csv(tmp_path / "bad.csv", index=False)
    assert score_cell("regression", train, labels, tmp_path / "bad.csv", "y")["status"] == "failed"

# the runner runs a system as a subprocess and scores its hand-made predictions...
# file correctly; a second run skips the finished cell
def test_runner_scores_handmade_file(tmp_path, monkeypatch):
    files = tmp_path / "files"
    monkeypatch.setattr(runner, "cell_dir", lambda group, name, level, seed, mess="mixed": files)
    monkeypatch.setattr(runner, "benchmark_outputs_dir", tmp_path / "outputs")
    monkeypatch.setattr(runner, "workdirs_dir", tmp_path / "w")
    monkeypatch.setattr(runner, "logs_dir", tmp_path / "logs")
    files.mkdir()
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(size=40), "Churn": ["Yes", "No"] * 20})
    test_labels = pd.DataFrame({"Churn": ["Yes", "No", "No", "Yes", "No", "Yes"]})
    train.to_csv(files / "train.csv", index=False)
    test_labels.assign(x=0).drop(columns="Churn").to_csv(files / "test_features.csv", index=False)
    test_labels.to_csv(files / "test_labels.csv", index=False)
    (files / "truth.json").write_text(json.dumps({"target": "Churn", "task": "binary-classification", "metric": "roc_auc"}))
    positive = np.array([0.9, 0.2, 0.4, 0.3, 0.1, 0.8])
    handmade = tmp_path / "handmade.csv"
    pd.DataFrame({"No": 1 - positive, "Yes": positive, "label": np.where(positive > 0.5, "Yes", "No")}).to_csv(handmade, index=False)
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "handmade.py").write_text(HANDMADE_ADAPTER)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"handmade": {"python": "main", "module": "handmade", "args": ["--handmade", str(handmade)], "llm": False}}))
    monkeypatch.setattr(runner, "system_environment", lambda threads=4, keep_key=False: dict(os.environ, PYTHONPATH=str(adapter_dir)))
    grid = tmp_path / "toy.json"
    grid.write_text(json.dumps({"name": "toy", "systems": ["handmade"], "datasets": ["toyset"], "levels": [0], "seeds": [0], "budget": 30}))
    monkeypatch.setattr(runner, "load_grid", lambda name: json.loads(grid.read_text()))
    rows = runner.run_grid("toy", registry_path=registry, results_folder=tmp_path / "results")
    expected = sk.roc_auc_score(test_labels["Churn"] == "Yes", positive)
    assert rows[0]["status"] == "ok"
    assert math.isclose(rows[0]["score"], expected)
    assert rows[0]["dummy_score"] == 0.5
    assert rows[0]["llm_calls"] == 2 and rows[0]["cost_usd"] == 0.01
    saved = pd.read_csv(tmp_path / "results" / "toy.csv")
    assert math.isclose(saved["score"][0], expected)
    assert runner.run_grid("toy", registry_path=registry, results_folder=tmp_path / "results") == []
    shutil.rmtree(tmp_path / "outputs")

# materialized files keep the target out of the test features and record hashes
def test_materialize_writes_files(tmp_path, monkeypatch):
    raw = pd.DataFrame({"a": [str(i) for i in range(50)], "quality": ["5", "6", "7", "5", "6"] * 10})
    spec = {"name": "toy", "group": "dev", "target": "quality", "task": "multi-class-classification", "sample_rows": None}
    monkeypatch.setattr(materialize, "benchmark_files_dir", tmp_path)
    monkeypatch.setattr(splits, "benchmark_splits_dir", tmp_path / "splits")
    monkeypatch.setattr(materialize, "dataset_spec", lambda name: spec)
    monkeypatch.setattr(materialize, "load_dataset", lambda name: raw)
    folder = materialize.materialize("toy", 0, 0)
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
    features = pd.read_csv(folder / "test_features.csv")
    assert "quality" not in features.columns
    assert truth["rows"] == {"train": 40, "test": 10}
    assert set(truth["files"]) == {"train.csv", "test_features.csv", "test_labels.csv"}

# grids expand over mess types and budgets; held-out cells are kept apart; the...
# pilot grids have the planned sizes
def test_grid_expansion():
    grid = {"systems": ["a", "b"], "datasets": ["d"], "levels": [2, 3], "seeds": [0], "budget": 60, "mess": ["units", "packed"], "budgets": [60, 300], "holdout": [{"system": "a", "dataset": "d", "level": 2}]}
    cells = runner.expand_cells(grid)
    assert len(cells) == 16 and {cell["mess"] for cell in cells} == {"units", "packed"} and {cell["budget"] for cell in cells} == {60, 300}
    assert sum(runner.is_holdout(cell, grid) for cell in cells) == 4
    sizes = {name: len(runner.expand_cells(runner.load_grid(name))) for name in ["smoke", "pilot-main", "pilot-seed", "pilot-leaks", "pilot-ablations", "pilot-natural", "pilot-budget", "main"]}
    assert sizes == {"smoke": 40, "pilot-main": 216, "pilot-seed": 96, "pilot-leaks": 25, "pilot-ablations": 20, "pilot-natural": 12, "pilot-budget": 4, "main": 1440}
