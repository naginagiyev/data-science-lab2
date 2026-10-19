import pytest
import numpy as np
import pandas as pd
from schemas import StopRun
from tools.reader import check_target, drop_duplicate_rows, drop_missing_target, read_table, to_raw_view

# a CSV file is read as text: blanks, "?" and leading zeros stay as written
def test_read_csv_keeps_text(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("id,charge,flag\n007,1.5,yes\n008, ,?\n", encoding="utf-8")
    raw, report = read_table(path)
    assert raw["id"].tolist() == ["007", "008"]
    assert raw["charge"].tolist() == ["1.5", " "]
    assert raw["flag"].tolist() == ["yes", "?"]
    assert report["rows"] == 2 and report["columns"] == 3 and report["delimiter"] == ","

# a DataFrame becomes the same raw view: numbers as text, missing as ""
def test_dataframe_source():
    raw, report = read_table(pd.DataFrame({"a": [1.5, np.nan], "b": ["x", None]}))
    assert raw["a"].tolist() == ["1.5", ""] and raw["b"].tolist() == ["x", ""]
    assert report["format"] == "dataframe"
    assert to_raw_view(pd.DataFrame({"c": [1, 2]}))["c"].tolist() == ["1", "2"]

# missing, unsupported, empty and header-only files are hard stops
def test_hard_stops(tmp_path):
    with pytest.raises(StopRun, match="does not exist"):
        read_table(tmp_path / "missing.csv")
    (tmp_path / "data.xyz").write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(StopRun, match="Unsupported"):
        read_table(tmp_path / "data.xyz")
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    with pytest.raises(StopRun):
        read_table(tmp_path / "empty.csv")
    (tmp_path / "header.csv").write_text("a,b\n", encoding="utf-8")
    with pytest.raises(StopRun, match="no data rows"):
        read_table(tmp_path / "header.csv")

# rows without a target and exact duplicate rows are removed and counted
def test_row_cleaning():
    raw = pd.DataFrame({"x": ["1", "1", "2", "3"], "y": ["a", "a", " ", "b"]})
    kept, dropped = drop_missing_target(raw, "y")
    assert dropped == 1 and kept["x"].tolist() == ["1", "1", "3"]
    unique, duplicates = drop_duplicate_rows(kept)
    assert duplicates == 1 and len(unique) == 2

# the target must exist, have enough rows and two classes for classification
def test_check_target():
    raw = pd.DataFrame({"x": [str(i) for i in range(30)], "y": ["a"] * 30})
    with pytest.raises(StopRun, match="not in the file"):
        check_target(raw, "z", "binary-classification", 20)
    with pytest.raises(StopRun, match="only one class"):
        check_target(raw, "y", "binary-classification", 20)
    with pytest.raises(StopRun, match="labeled rows"):
        check_target(raw.head(5), "x", "regression", 20)
    check_target(raw, "x", "regression", 20)
