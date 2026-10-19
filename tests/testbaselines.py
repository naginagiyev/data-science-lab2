import numpy as np
import pandas as pd
from types import SimpleNamespace
from benchmark.systems import dummy, skrub, lightgbm

# writes a small train and test file with a number, a text category and a...
# column name with special characters
def toy_files(tmp_path, task):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"size": rng.normal(size=120), "color": rng.choice(["red", "blue", "green"], 120), "a:b": rng.normal(size=120)})
    frame["y"] = (frame["size"] > 0).map({True: "Yes", False: "No"}) if task != "regression" else frame["size"] * 3
    frame.iloc[:100].to_csv(tmp_path / "train.csv", index=False)
    frame.iloc[100:].drop(columns="y").to_csv(tmp_path / "test.csv", index=False)
    return SimpleNamespace(train=tmp_path / "train.csv", tests=[tmp_path / "test.csv"], target="y", task=task, seed=0, budget=10)

# every main-environment baseline returns one prediction per test row
def test_baselines_predict(tmp_path):
    for task in ["binary-classification", "regression"]:
        arguments = toy_files(tmp_path, task)
        for system in [dummy, lightgbm, skrub]:
            result = system.fit_predict(arguments)
            prediction = np.asarray(result["predictions"][0])
            assert len(prediction) == 20
            if task != "regression":
                assert prediction.shape == (20, 2) and [str(label) for label in result["classes"]] == ["No", "Yes"]
