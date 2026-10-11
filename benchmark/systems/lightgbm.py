import re
import sys
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from benchmark.systems.common import run_adapter

# baseline "a quick data scientist": pd.read_csv defaults, text columns as...
# pandas categories (test categories aligned to training), default LightGBM
def prepare(frame, categories=None):
    frame = frame.rename(columns=lambda name: re.sub(r"[\[\]{}\",:]", "_", str(name)))
    categories = categories or {column: None for column in frame.columns if frame[column].dtype == object}
    for column in categories:
        if column in frame:
            frame[column] = frame[column].astype("category") if categories[column] is None else pd.Categorical(frame[column], categories=categories[column])
    return frame, {column: frame[column].cat.categories for column in categories if column in frame}

# fits LightGBM with default parameters and predicts every test file
def fit_predict(arguments):
    train = pd.read_csv(arguments.train)
    target = train.pop(arguments.target)
    features, categories = prepare(train)
    tests = [prepare(pd.read_csv(path)[train.columns], categories)[0] for path in arguments.tests]
    options = {"n_jobs": 4, "verbose": -1, "random_state": arguments.seed}
    if arguments.task == "regression":
        model = LGBMRegressor(**options).fit(features, target)
        return {"predictions": [model.predict(test) for test in tests]}
    model = LGBMClassifier(**options).fit(features, target)
    return {"predictions": [model.predict_proba(test) for test in tests], "classes": list(model.classes_)}

if __name__ == "__main__":
    sys.exit(run_adapter(fit_predict))
