import numpy as np
import pandas as pd
from tools.glance import glance, glance_text
from edatool.core.patterns import pattern_stats, pattern_text, template

# numbers with units, currency, percent, dates, packed values and booleans...
# are measured with their top units, formats and templates
def test_pattern_stats():
    units = pattern_stats(["1.37kg", "800 g", "2.1 kg", "1,200 g"])
    assert units["shares"]["number_with_unit"] == 1.0 and set(units["units"]) == {"kg", "g"}
    assert pattern_stats(["$1,250.00", "$3.10"])["shares"]["currency"] == 1.0
    assert pattern_stats(["12.5%", "3%"])["shares"]["percent"] == 1.0
    dates = pattern_stats(["2014-10-13", "13/10/2014", "Oct 13 2014", "20141013T000000"])
    assert dates["shares"]["date_like"] == 1.0 and len(dates["date_formats"]) == 3
    packed = pattern_stats(["3 bd / 2 ba", "4 bd / 1 ba", "2560x1600"])
    assert packed["shares"]["packed"] == 1.0 and packed["templates"][0][0] == "# bd / # ba"
    assert pattern_stats(["yes", "no", "Y"])["shares"]["boolean_like"] == 1.0
    assert pattern_stats(["1", "2.5", "-3e2"])["shares"]["number"] == 1.0
    assert pattern_stats([])["count"] == 0
    assert template("1.5x20") == "#x#"
    assert "unit 100% (kg" in pattern_text(units)

# glance records hidden missing tokens, types, ranges and patterns per column
def test_glance_columns():
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({
        "charges": [f"{value:.2f}" for value in rng.uniform(10, 100, 200)],
        "weight": [f"{value:.1f}kg" for value in rng.uniform(1, 3, 200)],
        "sold": pd.date_range("2020-01-01", periods=200).strftime("%Y-%m-%d"),
        "plan": rng.choice(["basic", "pro"], 200),
    }).astype(str)
    raw.loc[:4, "charges"] = " "
    result = glance(raw)
    columns = {entry["name"]: entry for entry in result["columns"]}
    assert columns["charges"]["missing_tokens"] == {" ": 5}
    assert columns["charges"]["type"] == "numeric_continuous" and columns["charges"]["min"] >= 10
    assert columns["weight"]["patterns"]["shares"]["number_with_unit"] == 1.0
    assert columns["sold"]["type"] == "datetime"
    text = glance_text(result)
    assert "`charges` numeric_continuous | missing 2.5% (' ':5)" in text
    assert "patterns: unit 100% (kg 100%)" in text

# wide tables show 40 columns in full and the rest as one short line each
def test_wide_table():
    raw = pd.DataFrame({f"c{i}": [str(value) for value in range(30)] for i in range(70)})
    text = glance_text(glance(raw))
    lines = text.splitlines()[1:]
    assert len(lines) == 70
    assert sum("unique" in line for line in lines) == 40
