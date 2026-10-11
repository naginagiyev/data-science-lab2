import io
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from operations.generic import GenericStage, TypingStep, sanitize_names

# CSV text with numbers (with spaces, exponents, inf), pandas NA tokens, "?",...
# booleans, text and dates
def messy_csv(rows=60, seed=0):
    rng = np.random.default_rng(seed)
    lines = ["count,amount,flag,code,label,when,blank"]
    for i in range(rows):
        count = rng.choice(["1", " 2", "3 ", "NA", "1e3", "", "-4"])
        amount = rng.choice(["1.5", "inf", "N/A", "7", "null"])
        flag = rng.choice(["True", "False"])
        code = rng.choice(["7", "?", "8"])
        label = rng.choice(["red", "blue", "None", "nan"])
        lines.append(f"{count},{amount},{flag},{code},{label},2020-01-{i % 28 + 1:02d},")
    return "\n".join(lines) + "\n"

# the typing step gives the same values and missing cells as pd.read_csv...
# with default options
def test_pandas_parity():
    text = messy_csv()
    raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    expected = pd.read_csv(io.StringIO(text))
    typed = TypingStep().fit(raw).transform(raw)
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]) or pd.api.types.is_bool_dtype(expected[column]):
            assert np.allclose(typed[column].to_numpy(dtype=float), expected[column].to_numpy(dtype=float), equal_nan=True), column
        else:
            assert typed[column].isna().equals(expected[column].isna()), column
            assert typed[column].dropna().tolist() == expected[column].dropna().tolist(), column

# types are decided on training rows; unparseable new values become NaN
def test_typing_unseen_rows():
    train = pd.DataFrame({"x": ["1", "2", "3"], "y": ["a", "b", "c"]})
    step = TypingStep().fit(train)
    new = step.transform(pd.DataFrame({"x": ["4", "oops", ""], "y": ["d", "NA", "e"]}))
    assert new["x"].tolist()[0] == 4.0 and np.isnan(new["x"].tolist()[1]) and np.isnan(new["x"].tolist()[2])
    assert new["y"].isna().tolist() == [False, True, False]

# names are made safe for LightGBM and unique
def test_sanitize_names():
    assert sanitize_names(["a[1]", "a_1_", "b:c", "a[1]", '{"q"}', ""]) == ["a_1_", "a_1__2", "b_c", "a_1__3", "__q__", "column"]

# LightGBM fits on the generic output of torture headers (unicode, emoji,...
# duplicates, JSON characters, spaces), for both kinds; the pipeline pickles
def test_torture_headers_lightgbm():
    rng = np.random.default_rng(0)
    names = ["Bakı şəhəri", "Город", "😀 score", "a", "a.1", "x[0]", "w{\"q\":1}", "a:b", "price,usd", " spaced "]
    raw = pd.DataFrame({name: [str(value) for value in rng.normal(size=80)] for name in names})
    raw["Город"] = rng.choice(["Москва", "Bakı", "Paris"], 80)
    target = rng.normal(size=80)
    for kind in ["tree", "linear"]:
        pipeline = Pipeline([("typing", TypingStep()), ("generic", GenericStage(kind))])
        features = pipeline.fit_transform(raw)
        LGBMRegressor(n_estimators=5, verbose=-1).fit(features, target)
        assert len(set(features.columns)) == features.shape[1]
        restored = pickle.loads(pickle.dumps(pipeline))
        assert restored.transform(raw.head(5)).shape == (5, features.shape[1])

# a table whose columns are all dropped still gives one constant feature
def test_no_features():
    raw = pd.DataFrame({"empty": [""] * 10})
    output = Pipeline([("typing", TypingStep()), ("generic", GenericStage())]).fit(raw).transform(raw)
    assert list(output.columns) == ["no_features"] and len(output) == 10
