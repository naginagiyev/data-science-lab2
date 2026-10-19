import json
import numpy as np
import pandas as pd
from benchmark.repairs import score_prepared

# writes the truth and cells of a small cell: rows 10..13 are the training rows
def write_cell(folder, cells):
    truth = {"dataset": "toy", "positions": {"train": [10, 11, 12, 13], "test": [14]}}
    (folder / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    pd.DataFrame(cells).to_parquet(folder / "cells.parquet")

# numbers, labels, packed parts in new columns and fake missing values are...
# scored against the originals; test rows are ignored
def test_score_prepared(tmp_path):
    write_cell(tmp_path, {
        "row": [10, 11, 12, 14, 10, 13, 11, 12],
        "column": ["weight", "weight", "weight", "weight", "size", "size", "age", "age"],
        "original": ["1.5", "800", "2.0", "9", "3", "4", "40", "50"],
        "corrupted": ["1.5kg", "0.8t", "2 kg", "9kg", "3x1", "4x2", "-999", "-999"],
        "type": ["units", "units", "units", "units", "packed", "packed", "fake_missing", "fake_missing"],
    })
    prepared = pd.DataFrame({"weight": [1.5, 800.0, 3.0, 1.0], "size_1": [3.0, 1.0, 1.0, 4.0], "size_2": [1.0, 2.0, 2.0, 2.0], "age": [30.0, np.nan, np.nan, 20.0]})
    rows = {row["column"]: row for row in score_prepared(prepared, tmp_path, ["weight", "size", "age"], "rules", 3)}
    assert rows["weight"]["cells"] == 3 and rows["weight"]["repaired"] == 2
    assert rows["size"]["accuracy"] == 1.0
    assert rows["age"]["accuracy"] == 1.0 and rows["age"]["false_nan_rate"] == 0.0

# yes/no originals restored as 1/0 count as repaired; a mapping that merges...
# both labels does not
def test_boolean_mapping(tmp_path):
    write_cell(tmp_path, {"row": [10, 11, 12, 13], "column": ["smoker"] * 4, "original": ["yes", "no", "yes", "no"], "corrupted": ["Y", "N", "TRUE", "f"], "type": ["booleans"] * 4})
    good = pd.DataFrame({"smoker": [1.0, 0.0, 1.0, 0.0]})
    merged = pd.DataFrame({"smoker": [1.0, 1.0, 1.0, 1.0]})
    assert score_prepared(good, tmp_path, ["smoker"], "rules", 1)[0]["accuracy"] == 1.0
    assert score_prepared(merged, tmp_path, ["smoker"], "rules", 1)[0]["accuracy"] == 0.0
