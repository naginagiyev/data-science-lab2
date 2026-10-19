import pickle
import pytest
import numpy as np
import pandas as pd
from schemas import Step
from operations.registry import COLUMN_OPERATIONS, build
from operations.target import LogTarget, MapTargetLabels
from operations.rows import DropConflictingRows, DropDuplicateRows, DropRowsWhere, RowFilter

# one case per column operation: parameters, training values of its columns and...
# unseen values for the same columns
CASES = {
    "drop_columns": ({"columns": ["a"]}, {"a": ["1", "2", "3", "4"]}, {"a": ["9", ""]}),
    "replace_values": ({"column": "a", "items": [{"value": " ", "new": None}, {"value": "?", "new": None}]}, {"a": ["1", " ", "?", "Bakı"]}, {"a": ["x", "?"]}),
    "set_missing_where": ({"column": "a", "below": 0, "above": 20, "values": ["0"]}, {"a": ["3", "-1", "33", "0"]}, {"a": ["50", "ab"]}),
    "parse_number": ({"column": "a", "remove": ["$"], "thousands": ",", "percent": True, "suffixes": [{"suffix": "k", "factor": 1000}]}, {"a": ["$1,250.00", "3.2k", "12.5%", "7"]}, {"a": ["bad", "1k"]}),
    "parse_units": ({"column": "a", "units": [{"unit": "kg", "factor": 1.0}, {"unit": "g", "factor": 0.001}]}, {"a": ["1.37kg", "800 g", "2 KG", "1,200 g"]}, {"a": ["5 lb", "3kg"]}),
    "parse_datetime": ({"column": "a", "formats": ["%Y%m%dT%H%M%S", "mixed"], "parts": ["year", "month", "day", "weekday", "epoch_days"]}, {"a": ["20141013T000000", "2015-01-02", "13 Oct 2014", "20150101T000000"]}, {"a": ["nonsense", "2016-02-29"]}),
    "date_difference": ({"start": "a", "end": "b", "unit": "years", "name": "age"}, {"a": ["1955", "2000", "1990", "1980"], "b": ["2014-10-13", "2015-01-01", "2014-06-30", "20141013T000000"]}, {"a": ["1999", ""], "b": ["2015-05-05", "2015-01-01"]}),
    "parse_boolean": ({"column": "a", "true_values": ["yes", "y", "true", "1"], "false_values": ["no", "n", "false", "0"]}, {"a": ["Y", "yes ", "TRUE", "no"]}, {"a": ["maybe", "N"]}),
    "map_categories": ({"column": "a", "items": [{"value": "NY", "canonical": "New York"}, {"value": "new york", "canonical": "New York"}]}, {"a": ["NY", "New York ", "new york", "Paris"]}, {"a": ["Москва", "ny"]}),
    "group_rare": ({"column": "a", "min_count": 2, "label": "Other"}, {"a": ["x", "x", "y", "z"]}, {"a": ["w", "x"]}),
    "as_category": ({"column": "a"}, {"a": ["98178", "98125", "98178", "98001"]}, {"a": ["99999", ""]}),
    "extract_pattern": ({"column": "a", "pattern": "(\\d+)x(\\d+)", "into": ["width", "height"], "types": ["number", "number"]}, {"a": ["2560x1600", "1920x1080", "1366x768", "800x600"]}, {"a": ["none", "640x480"]}),
    "add_missing_indicator": ({"columns": ["a"]}, {"a": ["1", "", "3", "4"]}, {"a": ["", "2"]}),
    "impute_constant": ({"column": "a", "value": 0}, {"a": ["1", "", "3", " "]}, {"a": ["", "5"]}),
    "clip_values": ({"column": "a", "lower_quantile": 0.0, "upper_quantile": 0.75}, {"a": ["1", "2", "3", "100"]}, {"a": ["1000", "-5"]}),
    "log_transform": ({"column": "a", "offset": 1.0}, {"a": ["0", "1", "10", "-5"]}, {"a": ["99", "x"]}),
    "text_features": ({"column": "a", "max_features": 50, "components": 2}, {"a": ["red big car", "small blue car", "big red truck", "tiny boat"]}, {"a": ["unseen words here", ""]}),
}

# a frame with the case columns, an untouched column and a non-default index
def frame_of(values, start=10):
    frame = pd.DataFrame(values)
    frame["other"] = [f"o{i}" for i in range(len(frame))]
    frame.index = range(start, start + len(frame))
    return frame

# every column operation has a case
def test_every_operation_has_a_case():
    assert {operation.name for operation in COLUMN_OPERATIONS} == set(CASES)

# the common contract of every column operation: same index and rows, unseen...
# rows and categories, NaN, empty strings, an all-missing column, one row,...
# pickle round trip, and columns it does not own stay unchanged
@pytest.mark.parametrize("name", sorted(CASES))
def test_operation_contract(name):
    params, train_values, unseen_values = CASES[name]
    operation = build(Step(op=name, params=params))
    train = frame_of(train_values)
    output = operation.fit(train).transform(train)
    assert len(output) == len(train) and output.index.equals(train.index)
    assert output["other"].tolist() == train["other"].tolist()
    unseen = frame_of(unseen_values, start=100)
    assert len(operation.transform(unseen)) == len(unseen)
    with_missing = frame_of({column: [np.nan, "", "é ü 😀", values[0]] for column, values in train_values.items()})
    assert len(operation.transform(with_missing)) == 4
    empty = frame_of({column: [""] * 4 for column in train_values})
    assert len(operation.transform(empty)) == 4
    single = frame_of({column: values[:1] for column, values in train_values.items()})
    assert len(build(Step(op=name, params=params)).fit(single).transform(single)) == 1
    restored = pickle.loads(pickle.dumps(operation))
    pd.testing.assert_frame_equal(restored.transform(train), output)
    assert "seconds" in operation.stats_

# the values each operation produces
def test_operation_values():
    def run(name, values):
        params, _, _ = CASES[name]
        frame = pd.DataFrame(values)
        return build(Step(op=name, params=params)).fit(frame).transform(frame)
    assert run("parse_number", {"a": ["$1,250.00", "3.2k", "12.5%", "x"]})["a"].tolist()[:3] == [1250.0, 3200.0, 0.125]
    assert np.allclose(run("parse_units", {"a": ["1.37kg", "800 g", "2 KG", "1,200 g"]})["a"], [1.37, 0.8, 2.0, 1.2])
    dates = run("parse_datetime", {"a": ["20141013T000000", "2015-01-02"]})
    assert dates["a_year"].tolist() == [2014.0, 2015.0] and dates["a_weekday"].tolist() == [0.0, 4.0] and "a" not in dates
    age = run("date_difference", {"a": ["1955"], "b": ["2014-10-13"]})["age"].iloc[0]
    assert 59.7 < age < 59.8
    assert run("parse_boolean", {"a": ["Y", "yes ", "TRUE", "no", "maybe"]})["a"].tolist()[:4] == [1.0, 1.0, 1.0, 0.0]
    assert run("map_categories", {"a": ["NY", "new york", " Paris "]})["a"].tolist() == ["New York", "New York", "Paris"]
    assert run("extract_pattern", {"a": ["2560x1600"]})[["width", "height"]].iloc[0].tolist() == [2560.0, 1600.0]
    assert run("add_missing_indicator", {"a": ["1", ""]})["a_missing"].tolist() == [0.0, 1.0]
    assert run("set_missing_where", {"a": ["3", "-1", "33", "0"]})["a"].isna().tolist() == [False, True, True, True]
    assert run("replace_values", {"a": ["1", " ", "?"]})["a"].isna().tolist() == [False, True, True]
    assert str(run("as_category", {"a": ["98178", "98001"]})["a"].dtype) == "category"
    stats = build(Step(op="parse_units", params=CASES["parse_units"][0]))
    stats.fit(pd.DataFrame({"a": ["1 kg", "2 kgs", "?"]})).transform(pd.DataFrame({"a": ["1 kg", "2 kgs", "?"]}))
    assert stats.stats_["failures"] == 2 and "2 kgs" in stats.stats_["examples"]

# group_rare and clip_values learn from the training rows only
def test_learned_state():
    rare = build(Step(op="group_rare", params={"column": "a", "min_count": 2, "label": "Other"}))
    rare.fit(pd.DataFrame({"a": ["x", "x", "y"]}))
    assert rare.transform(pd.DataFrame({"a": ["y", "y", "x", "new"]}))["a"].tolist() == ["Other", "Other", "x", "Other"]
    clip = build(Step(op="clip_values", params={"column": "a", "lower_quantile": None, "upper_quantile": 0.5}))
    clip.fit(pd.DataFrame({"a": ["1", "2", "3"]}))
    assert clip.transform(pd.DataFrame({"a": ["100"]}))["a"].tolist() == [2.0]

# bad parameters and unknown operations are refused
def test_validation():
    with pytest.raises(Exception):
        build(Step(op="parse_units", params={"column": "a"}))
    with pytest.raises(KeyError):
        build(Step(op="magic", params={}))
    with pytest.raises(ValueError):
        build(Step(op="extract_pattern", params={"column": "a", "pattern": "(\\d+)", "into": ["x", "y"]})).fit(pd.DataFrame({"a": ["1"]}))
    with pytest.raises(KeyError):
        build(Step(op="parse_number", params={"column": "missing"})).fit(pd.DataFrame({"a": ["1"]}))

# row filters only say which training rows to keep; they never transform rows...
# at prediction time
def test_row_filters():
    frame = pd.DataFrame({"x": ["1", "1", "1", "2", "15000"], "z": ["a", "a", "a", "b", "c"]})
    y = np.array([0, 0, 1, 1, 0])
    assert DropDuplicateRows().mask(frame, y).tolist() == [True, False, True, True, True]
    assert DropConflictingRows().mask(frame, y).tolist() == [False, False, False, True, True]
    where = DropRowsWhere({"condition": "`x` > 13000 or `z` == 'b'"})
    assert where.mask(frame, y).tolist() == [True, True, True, False, False]
    combined = RowFilter([DropDuplicateRows(), where]).mask(frame, y)
    assert combined.tolist() == [True, False, True, False, False]
    with pytest.raises(RuntimeError):
        DropDuplicateRows().transform(frame)

# target operations never change the label space: spellings map to an existing...
# class, different classes are never merged, log_target is undone exactly
def test_target_operations():
    labels = ["Yes", "No", "yes ", "No"]
    good = MapTargetLabels({"items": [{"value": "yes ", "canonical": "Yes"}]})
    assert good.check(labels) == [] and good.map(labels).tolist() == ["Yes", "No", "Yes", "No"]
    merge = MapTargetLabels({"items": [{"value": "No", "canonical": "Yes"}]})
    assert "never merged" in merge.check(labels)[0]
    assert MapTargetLabels({"items": [{"value": "x", "canonical": "Maybe"}]}).check(labels)
    log = LogTarget({"offset": 1.0})
    values = np.array([0.0, 5.0, 1e6])
    assert np.allclose(log.inverse(log.forward(values)), values)
    assert log.check([-2.0]) and not log.check([0.0])
    with pytest.raises(RuntimeError):
        log.transform(pd.DataFrame({"a": [1]}))
