import numpy as np
import pandas as pd
from edatool import audit
from edatool.agent.session import prompt_text

# audits a frame (with a random target) without the agent and returns the...
# issues of one code
def issues_of(columns, code, rows=200):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({name: values[:rows] for name, values in columns.items()})
    frame["y"] = rng.choice(["a", "b"], len(frame))
    return [issue for issue in audit(frame, "y", "binary-classification", use_agent=False).issues if issue.code == code]

# repeats a list of values to the given length
def cycle(values, rows=200):
    return [values[index % len(values)] for index in range(rows)]

# B07: one unit is found without review; two units need review (factors)
def test_units():
    rng = np.random.default_rng(1)
    single = issues_of({"Weight": [f"{value:.2f}kg" for value in rng.uniform(1, 3, 200)]}, "B07")
    assert len(single) == 1 and single[0].fix.params["units"] == ["kg"] and not single[0].needs_review
    mixed = issues_of({"Weight": [f"{value:.2f}kg" if index % 2 else f"{value * 1000:.0f} g" for index, value in enumerate(rng.uniform(1, 3, 200))]}, "B07")
    assert len(mixed) == 1 and set(mixed[0].fix.params["units"]) == {"kg", "g"} and mixed[0].needs_review
    assert issues_of({"Plan": cycle(["basic", "pro", "team"])}, "B07") == []

# B08: values with a fixed packed template give an extract_pattern fix
def test_packed():
    rng = np.random.default_rng(2)
    found = issues_of({"Rooms": [f"{a} bd / {b} ba" for a, b in zip(rng.integers(1, 6, 200), rng.integers(1, 4, 200))]}, "B08")
    assert len(found) == 1 and found[0].fix.action == "extract_pattern"
    assert found[0].fix.params["into"] == ["Rooms_1", "Rooms_2"]
    assert issues_of({"Plan": cycle(["basic", "pro"])}, "B08") == []

# B09: mixed date formats are found; an undecidable day/month order needs review
def test_mixed_dates():
    days = pd.date_range("2020-01-01", periods=200)
    mixed = [day.strftime("%Y-%m-%d") if index % 2 else day.strftime("%d/%m/%Y") for index, day in enumerate(days)]
    found = issues_of({"Sold": mixed}, "B09")
    assert len(found) == 1 and found[0].evidence["day_order"] == "day_first" and found[0].fix.params["dayfirst"]
    assert "%Y-%m-%d" in found[0].fix.params["formats"] and not found[0].needs_review
    ambiguous = [day.strftime("%Y-%m-%d") if index % 2 else day.strftime("%d/%m/%Y") for index, day in enumerate(pd.date_range("2020-01-01", periods=12, freq="MS").repeat(17))]
    assert issues_of({"Sold": ambiguous}, "B09")[0].needs_review
    assert issues_of({"Sold": [day.strftime("%Y-%m-%d") for day in days]}, "B09") == []
    assert issues_of({"Sold": mixed}, "B08") == []

# the new hidden-missing tokens are recognised (C02)
def test_new_tokens():
    values = cycle(["1.5", "2.5", "n.a.", "3.0", "nil", "4.0", "--", "not available"])
    found = issues_of({"Charge": values}, "C02")
    assert found and {"n.a.", "nil", "--", "not available"} <= set(found[0].evidence["tokens"])

# the agent prompts get the fix vocabulary from FixAction (with extract_pattern)
def test_prompt_vocabulary():
    for name in ["reviewer", "semantic"]:
        text = prompt_text(name)
        assert "{fix_actions}" not in text and "extract_pattern" in text
