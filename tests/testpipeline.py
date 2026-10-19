import json
import numpy as np
import pandas as pd
from orchestrator import run
from benchmark.systems import v2
from types import SimpleNamespace
from settings import load_settings
from modelling.evaluator import score
from modelling.fittedmodel import FittedModel

FAST = ["budget.total_seconds=12", "budget.min_model_seconds=6", "engine.candidates=[\"dummy\", \"linear\", \"lightgbm\"]"]
RUN_FILES = ["settings.json", "context.json", "run.log", "glance.json", "task.json", "split.json", "decisions.jsonl", "optuna.csv", "model.joblib", "scores.json", "report.md"]

# writes a small churn-like CSV with an id, a blank token, text and numbers
def churn_file(folder, rows=400, seed=0):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, rows)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], rows)
    churn = np.where((tenure < 12) & (contract == "Month-to-month") | (rng.random(rows) < 0.1), "Yes", "No")
    frame = pd.DataFrame({
        "customerID": [f"{i:04d}-ABCD" for i in range(rows)],
        "tenure": tenure.astype(str),
        "Contract": contract,
        "TotalCharges": [f"{value:.2f}" for value in tenure * rng.uniform(20, 100, rows)],
        "Churn": churn,
    })
    frame.loc[tenure == 0, "TotalCharges"] = " "
    path = folder / "churn.csv"
    frame.to_csv(path, index=False)
    return path

# the generic variant finishes with a complete run folder, a model that beats...
# the dummy, and a saved model that reloads and predicts raw input
def test_generic_run(tmp_path):
    path = churn_file(tmp_path)
    context = run(path, load_settings(assignments=FAST), "Churn", "binary-classification", variant="generic", out_dir=tmp_path / "runs", console=False)
    folder = next((tmp_path / "runs").iterdir())
    for name in RUN_FILES:
        assert (folder / name).exists(), name
    assert all(stage.ok for stage in context.stages)
    scores = json.loads((folder / "scores.json").read_text(encoding="utf-8"))
    assert scores["test"]["test"] > scores["test"]["dummy_test"] + 0.1
    ranking = scores["test"]["importance"]
    assert ranking["complete"] and ranking["columns"][0][1] > 0 and "Permutation importance" in (folder / "report.md").read_text(encoding="utf-8")
    model = FittedModel.load(folder / "model.joblib")
    raw = pd.read_csv(path, dtype=str, keep_default_na=False).head(5)
    labels = model.predict(raw.drop(columns="Churn"))
    assert set(labels) <= {"Yes", "No"} and len(labels) == 5
    probabilities = model.predict_proba(raw.drop(columns=["Churn", "Contract"]).assign(extra="x"))
    assert list(probabilities.columns) == ["No", "Yes"] and np.allclose(probabilities.sum(axis=1), 1)
    assert model.predict_frame(path).shape[1] == 3

# rules frame the task without flags: the last column with a name hint, task...
# from the values
def test_rules_framing(tmp_path):
    path = churn_file(tmp_path)
    context = run(path, load_settings(assignments=FAST + ["engine.candidates=[\"dummy\"]"]), variant="generic", out_dir=tmp_path / "runs", console=False)
    assert context.task.target == "Churn" and context.task.task == "binary-classification"
    assert context.task.source == "rules" and context.task.id_columns == ["customerID"]

# a test class the model never saw keeps the metrics defined
def test_unseen_test_class():
    truth = np.array(["a", "b", "c"], dtype=object)
    values = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    result = score("multi-class-classification", truth, values, ["a", "b"])
    assert np.isfinite(result["score"])

# the v2 benchmark adapter trains in benchmark mode, keeps the run folder in...
# the output folder and predicts every test file
def test_v2_adapter(tmp_path):
    path = churn_file(tmp_path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.iloc[:300].to_csv(tmp_path / "train.csv", index=False)
    raw.iloc[300:].drop(columns="Churn").to_csv(tmp_path / "test.csv", index=False)
    arguments = SimpleNamespace(
        train=str(tmp_path / "train.csv"), tests=[str(tmp_path / "test.csv")], target="Churn", task="binary-classification",
        budget=20.0, seed=0, variant="generic", out=str(tmp_path / "out"), set=FAST[2:],
    )
    result = v2.fit_predict(arguments)
    assert result["classes"] == ["No", "Yes"] and result["predictions"][0].shape == (100, 2)
    assert (tmp_path / "out" / "run" / "model.joblib").exists()
    assert result["meta"]["llm_calls"] == 0 and result["meta"]["run_dir"].endswith("run")

# the rules variant runs the audit, leak guard, plan, checks and gate: the id...
# is dropped, the blank token becomes missing, and every file exists
def test_rules_run(tmp_path):
    path = churn_file(tmp_path)
    context = run(path, load_settings(assignments=FAST + ["gate.max_rows=400"]), "Churn", "binary-classification", variant="rules", out_dir=tmp_path / "runs", console=False)
    folder = next((tmp_path / "runs").iterdir())
    for name in RUN_FILES + ["leaks.json", "plans.json", "checks.json", "gate.json"]:
        assert (folder / name).exists(), name
    assert all(stage.ok for stage in context.stages), [stage for stage in context.stages if not stage.ok]
    final = context.final_plan
    assert any(step.op == "drop_columns" and "customerID" in step.params["columns"] and step.protected for step in final.steps)
    assert any(step.op == "replace_values" and step.params["column"] == "TotalCharges" for step in final.steps)
    model = FittedModel.load(folder / "model.joblib")
    assert len(model.predict(pd.read_csv(path, dtype=str, keep_default_na=False).head(3))) == 3
