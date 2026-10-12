import time
import numpy as np
import pandas as pd
from settings import load_settings
from modelling.cv import prepare_folds
from schemas import Plan, SplitInfo, Step
from checks.gate import run_gate, score_plan
from operations.compiler import feature_pipeline, row_filter

# a regression table where the signal hides in a text weight with units
def unit_table(rows=600, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.uniform(0.5, 3.0, rows)
    text = [f"{value:.2f}kg" if index % 2 else f"{value * 1000:.0f} g" for index, value in enumerate(weight)]
    other = rng.normal(size=rows)
    target = 10 * weight + other + rng.normal(scale=0.3, size=rows)
    return pd.DataFrame({"Weight": text, "other": [f"{value:.3f}" for value in other], "y": [f"{value:.3f}" for value in target]})

# a binary table with one strong numeric feature and a rare class mark
def class_table(rows=600, seed=1):
    rng = np.random.default_rng(seed)
    strong = rng.normal(size=rows)
    label = np.where(strong + rng.normal(scale=0.5, size=rows) > 0, "yes", "no")
    return pd.DataFrame({"strong": [f"{value:.3f}" for value in strong], "noise": [f"{value:.3f}" for value in rng.normal(size=rows)], "y": label})

SPLIT = SplitInfo(strategy="random")

# the gate settings with the quick tests of this file
def gate_settings():
    return load_settings(assignments=["gate.max_rows=600"]).gate

# a helpful operation (parsing a weight with units that drives the target) is...
# accepted, with a positive effect
def test_helpful_step_accepted():
    table = unit_table()
    plan = Plan(steps=[Step(op="parse_units", params={"column": "Weight", "units": [{"unit": "kg", "factor": 1}, {"unit": "g", "factor": 0.001}]})])
    final, result = run_gate(plan, table, "y", "regression", "rmse", SPLIT, gate_settings(), 0, list(table.columns))
    assert result["accepted"] and final.steps and result["effects"][0]["effect"] > 0
    assert result["p1"]["score"] < result["p0"]["score"]

# a harmful operation (dropping the strongest feature) is rejected
def test_harmful_step_rejected():
    table = class_table()
    plan = Plan(steps=[Step(op="drop_columns", params={"columns": ["strong"]})])
    final, result = run_gate(plan, table, "y", "binary-classification", "roc_auc", SPLIT, gate_settings(), 0, list(table.columns))
    assert final.steps == [] and result["effects"][0]["effect"] < 0

# a row filter that removes one class from training is rejected
def test_harmful_row_filter_rejected():
    table = class_table()
    plan = Plan(row_steps=[Step(op="drop_rows_where", params={"condition": "`strong` > 0"})])
    final, result = run_gate(plan, table, "y", "binary-classification", "roc_auc", SPLIT, gate_settings(), 0, list(table.columns))
    assert final.row_steps == []

# protected steps are never gated, even when they hurt
def test_protected_kept():
    table = class_table()
    plan = Plan(steps=[Step(op="drop_columns", params={"columns": ["strong"]}, protected=True, source="leakguard")])
    final, result = run_gate(plan, table, "y", "binary-classification", "roc_auc", SPLIT, gate_settings(), 0, list(table.columns))
    assert final.steps and result["effects"] == []

# row filters touch the training part of a fold only: validation rows stay
def test_validation_rows_never_filtered():
    table = class_table()
    plan = Plan(row_steps=[Step(op="drop_rows_where", params={"condition": "`strong` > 1"})])
    frame = table.drop(columns="y")
    y = (table["y"] == "yes").astype(int).to_numpy()
    folds = [list(range(0, 300)), list(range(300, 600))]
    data = prepare_folds(frame, y, folds, lambda: feature_pipeline(plan, "tree"), row_filter(plan))
    for fold, valid in zip(data, folds):
        assert list(fold.valid_rows) == valid and len(fold.valid_x) == 300
        assert len(fold.train_x) < 300
    mean, scores = score_plan(plan, frame, table["y"].to_numpy(), folds, "binary-classification", "roc_auc", gate_settings(), 0)
    assert len(scores) == 2 and 0.5 < mean <= 1.0

# when the deadline leaves no time for single-step tests, an untested drop is...
# left out, while other untested steps stay
def test_untested_drop_left_out():
    table = unit_table()
    units = Step(op="parse_units", params={"column": "Weight", "units": [{"unit": "kg", "factor": 1}, {"unit": "g", "factor": 0.001}]})
    plan = Plan(steps=[units, Step(op="drop_columns", params={"columns": ["other"]})])
    final, result = run_gate(plan, table, "y", "regression", "rmse", SPLIT, gate_settings(), 0, list(table.columns), deadline=time.monotonic())
    assert [step.op for step in final.steps] == ["parse_units"] and result["reduced_not_scored"]
    assert {item["op"]: bool(item.get("removed")) for item in result["effects"]} == {"parse_units": False, "drop_columns": True}
