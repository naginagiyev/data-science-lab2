import numpy as np
import pandas as pd
from edatool import audit
from operations.translator import default_plan

# a messy churn-like table: id, blank tokens, duplicates, a date column, a...
# code column, mixed yes/no spellings and units
def messy_table(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, rows)
    frame = pd.DataFrame({
        "customerID": [f"{index:05d}-AB" for index in range(rows)],
        "TotalCharges": [f"{value:.2f}" for value in tenure * rng.uniform(20, 90, rows)],
        "signup": pd.date_range("2018-01-01", periods=rows).strftime("%Y-%m-%d"),
        "zipcode": [str(value) for value in rng.choice([98001, 98002, 98003, 98004, 98005], rows)],
        "Partner": rng.choice(["Yes", "No", "Y", "N"], rows),
        "Weight": [f"{value:.2f}kg" for value in rng.uniform(1, 4, rows)],
        "Churn": rng.choice(["Yes", "No"], rows),
    })
    frame.loc[tenure == 0, "TotalCharges"] = " "
    frame = pd.concat([frame, frame.iloc[:5]], ignore_index=True)
    return frame

# the default plan of a deterministic audit (no agent): protected steps for...
# hidden missing tokens and duplicates; gated parse, date, unit and code steps;...
# never a step on the target
def test_default_plan():
    frame = messy_table()
    report = audit(frame, "Churn", "binary-classification", use_agent=False)
    plan, skipped = default_plan(report, "Churn")
    by_op = {(step.op, step.params.get("column")): step for step in plan.steps}
    replace = by_op[("replace_values", "TotalCharges")]
    assert replace.protected and replace.params["items"] == [{"value": " ", "new": None}]
    assert ("parse_number", "TotalCharges") in by_op and not by_op[("parse_number", "TotalCharges")].protected
    assert by_op[("parse_units", "Weight")].params["units"] == [{"unit": "kg", "factor": 1.0}]
    assert ("parse_datetime", "signup") in by_op
    assert any(step.op == "drop_duplicate_rows" and step.protected for step in plan.row_steps)
    assert not any("Churn" in str(step.params) for step in plan.steps)
    assert not any(step.op == "drop_columns" and "customerID" in step.params.get("columns", []) for step in plan.steps)
    positions = {step.op: index for index, step in enumerate(plan.steps)}
    assert positions["replace_values"] < positions["parse_number"]

# rare classes and target outliers are never translated (the target label...
# space never changes), and two units without factors are left to the cleaner
def test_policy_and_skips():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "x": [str(value) for value in rng.normal(size=300)],
        "Weight": [f"{value:.2f}kg" if index % 2 else f"{value * 1000:.0f} g" for index, value in enumerate(rng.uniform(1, 3, 300))],
        "y": ["a"] * 150 + ["b"] * 145 + ["c"] * 5,
    })
    report = audit(frame, "y", "multi-class-classification", use_agent=False)
    plan, skipped = default_plan(report, "y")
    assert plan.target_steps == []
    assert not any(step.op == "parse_units" for step in plan.steps)
    assert any("unit factors unknown" in item["reason"] for item in skipped)

# mixed currency formats parse to numbers, k/M suffixes get magnitude factors,...
# packed decimals are split (not taken for an id), and group_rare is left out...
# on parsed columns
def test_currency_magnitudes_and_packed():
    rng = np.random.default_rng(1)
    rows = 400
    age = rng.integers(1, 52, rows)
    bedrooms = rng.integers(100, 3000, rows)
    frame = pd.DataFrame({
        "Age": [f"${value}" if index % 3 else f"{value} USD" for index, value in enumerate(age)],
        "Bedrooms": [f"{value / 1000:g}k" if index % 4 else f"{value / 1e6:g}M" for index, value in enumerate(bedrooms)],
        "Packed": [f"L:{lat:.2f} W:{income:.4f}" for lat, income in zip(rng.uniform(32, 42, rows), rng.uniform(0.5, 15, rows))],
        "y": rng.normal(0, 1, rows).astype(str),
    })
    report = audit(frame, "y", "regression", use_agent=False)
    plan, skipped = default_plan(report, "y")
    by_op = {(step.op, step.params.get("column")): step for step in plan.steps}
    assert "USD" in by_op[("parse_number", "Age")].params["remove"]
    assert {item["unit"].lower(): item["factor"] for item in by_op[("parse_units", "Bedrooms")].params["units"]} == {"k": 1e3, "m": 1e6}
    assert ("extract_pattern", "Packed") in by_op
    assert not any(step.op == "group_rare" for step in plan.steps)
    assert not any(issue.code == "E03" and "Packed" in issue.columns for issue in report.issues)
