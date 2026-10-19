import numpy as np
import pandas as pd
from audit import core
from audit.core import issues_text, run_audit

# audits a frame (with a random target) and returns the issues of one code
def issues_of(columns, code, rows=200):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({name: values[:rows] for name, values in columns.items()})
    frame["y"] = rng.choice(["a", "b"], len(frame))
    return [issue for issue in run_audit(frame, "y", "binary-classification").issues if issue.code == code]

# repeats a list of values to the given length
def cycle(values, rows=200):
    return [values[index % len(values)] for index in range(rows)]

# B07: one unit or two units are found with their names; plain labels are not
def test_units():
    rng = np.random.default_rng(1)
    single = issues_of({"Weight": [f"{value:.2f}kg" for value in rng.uniform(1, 3, 200)]}, "B07")
    assert len(single) == 1 and single[0].params["units"] == ["kg"] and single[0].rows == 0
    mixed = issues_of({"Weight": [f"{value:.2f}kg" if index % 2 else f"{value * 1000:.0f} g" for index, value in enumerate(rng.uniform(1, 3, 200))]}, "B07")
    assert len(mixed) == 1 and set(mixed[0].params["units"]) == {"kg", "g"}
    assert issues_of({"Plan": cycle(["basic", "pro", "team"])}, "B07") == []

# B08: values with a fixed packed template give an extract_pattern fix
def test_packed():
    rng = np.random.default_rng(2)
    found = issues_of({"Rooms": [f"{a} bd / {b} ba" for a, b in zip(rng.integers(1, 6, 200), rng.integers(1, 4, 200))]}, "B08")
    assert len(found) == 1 and found[0].action == "extract_pattern" and found[0].params["into"] == ["Rooms_1", "Rooms_2"]
    assert issues_of({"Plan": cycle(["basic", "pro"])}, "B08") == []

# B09: mixed date formats are found with their strptime formats and the day order
def test_mixed_dates():
    days = pd.date_range("2020-01-01", periods=200)
    mixed = [day.strftime("%Y-%m-%d") if index % 2 else day.strftime("%d/%m/%Y") for index, day in enumerate(days)]
    found = issues_of({"Sold": mixed}, "B09")
    assert len(found) == 1 and found[0].evidence["day_order"] == "day_first" and found[0].params["dayfirst"]
    assert {"%Y-%m-%d", "%d/%m/%Y"} == set(found[0].params["formats"])
    ambiguous = [day.strftime("%Y-%m-%d") if index % 2 else day.strftime("%d/%m/%Y") for index, day in enumerate(pd.date_range("2020-01-01", periods=12, freq="MS").repeat(17))]
    assert issues_of({"Sold": ambiguous}, "B09")[0].evidence["day_order"] == "ambiguous"
    assert issues_of({"Sold": [day.strftime("%Y-%m-%d") for day in days]}, "B09") == []
    assert issues_of({"Sold": mixed}, "B08") == []

# C02: hidden missing tokens are counted per token
def test_hidden_missing():
    found = issues_of({"Charge": cycle(["1.5", "2.5", "n.a.", "3.0", "nil", "4.0", "--", "not available"])}, "C02")
    assert found and {"n.a.", "nil", "--", "not available"} <= set(found[0].evidence["tokens"]) and found[0].rows == 100

# G01: a copy of the target is found by the tree and a column of pure...
# categories by the purity rule; a noise column is not
def test_leak_signals():
    rng = np.random.default_rng(3)
    target = rng.choice(["a", "b"], 400)
    frame = pd.DataFrame({
        "refund": np.where(target == "a", "yes", "no"),
        "store": [f"{label}{index % 10}" for index, label in enumerate(target)],
        "noise": [str(value) for value in rng.normal(size=400)],
        "y": target,
    })
    found = {(issue.columns[0], issue.rule) for issue in run_audit(frame, "y", "binary-classification").issues if issue.code == "G01"}
    assert ("refund", "tree") in found and ("store", "purity") in found and not any(column == "noise" for column, _ in found)

# a check that fails is recorded and the other checks still run
def test_failed_check(monkeypatch):
    def broken(context):
        raise RuntimeError("boom")
    monkeypatch.setattr(core, "CHECKS", [broken] + core.CHECKS)
    report = run_audit(pd.DataFrame({"x": cycle(["1", "?", "3"]), "y": cycle(["a", "b"])}), "y", "binary-classification")
    assert report.failed == {"broken": "RuntimeError: boom"} and any(issue.code == "C02" for issue in report.issues)

# the prompt text shows the id, the columns, the evidence and the fix
def test_issues_text():
    report = run_audit(pd.DataFrame({"x": cycle(["1", "?", "3"]), "y": cycle(["a", "b"])}), "y", "binary-classification")
    text = issues_text(report)
    assert "[must_fix] C02-1 x: hidden missing values" in text and "evidence: tokens" in text and "fix: replace_values" in text
    assert issues_text(report, max_chars=20).endswith("(more issues cut)")
