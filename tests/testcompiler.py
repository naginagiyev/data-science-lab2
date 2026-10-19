import pickle
import numpy as np
import pandas as pd
from schemas import Plan, Step
from settings import load_settings
from checks.datachecks import check_steps
from operations.compiler import AgentSteps, feature_pipeline, row_filter, target_steps, validate_plan

COLUMNS = ["Weight", "City", "Price", "y"]

# validation tracks columns through the plan and rejects unknown operations,...
# bad parameters, unknown or dropped columns, the target, wrong kinds and clashes
def test_validate_plan():
    good = Plan(steps=[Step(op="parse_units", params={"column": "Weight", "units": [{"unit": "kg", "factor": 1}]}), Step(op="drop_columns", params={"columns": ["City"]})])
    assert validate_plan(good, COLUMNS, "y", "regression") == []
    cases = {
        "unknown operation": Plan(steps=[Step(op="magic", params={})]),
        "bad parameters": Plan(steps=[Step(op="parse_units", params={"column": "Weight"})]),
        "unknown column": Plan(steps=[Step(op="drop_columns", params={"columns": ["City"]}), Step(op="as_category", params={"column": "City"})]),
        "never touch the target": Plan(steps=[Step(op="parse_number", params={"column": "y"})]),
        "not a column operation": Plan(steps=[Step(op="drop_duplicate_rows", params={})]),
        "regression only": Plan(target_steps=[Step(op="log_target", params={})]),
        "already exist": Plan(steps=[Step(op="extract_pattern", params={"column": "Weight", "pattern": "(\\d+)", "into": ["Price"]})]),
    }
    for expected, plan in cases.items():
        errors = validate_plan(plan, COLUMNS, "y", "binary-classification")
        assert errors and expected in errors[0], (expected, errors)

# the compiled pipeline runs the agent steps before typing and the generic...
# stage, keeps rows, and pickles; row and target steps compile separately
def test_compiled_pipeline():
    frame = pd.DataFrame({"Weight": ["1.5kg", "800 g", "2kg", "1kg"] * 10, "City": ["a", "b"] * 20})
    plan = Plan(
        steps=[Step(op="parse_units", params={"column": "Weight", "units": [{"unit": "kg", "factor": 1}, {"unit": "g", "factor": 0.001}]})],
        row_steps=[Step(op="drop_duplicate_rows", params={})],
        target_steps=[Step(op="map_target_labels", params={"items": [{"value": "yes ", "canonical": "Yes"}]})],
    )
    pipeline = feature_pipeline(plan, "tree")
    output = pipeline.fit_transform(frame)
    assert len(output) == 40 and "Weight" in output.columns and output["Weight"].max() == 2.0
    assert pickle.loads(pickle.dumps(pipeline)).transform(frame.head(3)).shape[0] == 3
    assert row_filter(plan).mask(frame, np.zeros(40)).sum() == 4
    assert target_steps(plan).labels(["yes ", "No"]).tolist() == ["Yes", "No"]
    snapshots = AgentSteps(plan.steps).fit(frame).snapshots(frame)
    assert snapshots[0][0] == "step 1: parse_units Weight"

# a plan with a wrong unit map fails the data checks with a structured error...
# that names the operation, the column, the failure share and examples
def test_wrong_unit_map_fails():
    frame = pd.DataFrame({"Weight": ["1.5kg", "800 g", "2 kg", "950 g", "1.2kg"] * 20, "City": ["a", "b"] * 50})
    plan = Plan(steps=[Step(op="parse_units", params={"column": "Weight", "units": [{"unit": "kg", "factor": 1}]}), Step(op="as_category", params={"column": "City"})])
    folds = [list(range(0, 50)), list(range(50, 100))]
    results, failing = check_steps(plan, frame, ["x"] * 100, folds, "y", load_settings().checks)
    assert failing == [0]
    error = results[0]["problems"][0]
    assert error.startswith("parse_units on Weight: 40.0% of values could not be parsed") and '"800 g"' in error
    assert results[1]["problems"] == []

# a step that copies the target is caught
def test_target_copy_caught():
    frame = pd.DataFrame({"code": ["yes-1", "no-2"] * 20})
    plan = Plan(steps=[Step(op="extract_pattern", params={"column": "code", "pattern": "([a-z]+)", "into": ["answer"], "types": ["text"]})])
    results, failing = check_steps(plan, frame, ["yes", "no"] * 20, [list(range(20)), list(range(20, 40))], "y", load_settings().checks)
    assert failing == [0] and "equals or is named like the target" in results[0]["problems"][0]
