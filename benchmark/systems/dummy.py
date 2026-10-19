import sys
import pandas as pd
from benchmark.systems.common import run_adapter

# baseline: the class shares of the training labels, or the training mean
def fit_predict(arguments):
    target = pd.read_csv(arguments.train)[arguments.target]
    tests = [pd.read_csv(path) for path in arguments.tests]
    if arguments.task == "regression":
        return {"predictions": [[target.mean()] * len(test) for test in tests]}
    shares = target.value_counts(normalize=True).sort_index()
    return {"predictions": [[shares.to_list()] * len(test) for test in tests], "classes": shares.index.to_list()}

if __name__ == "__main__":
    sys.exit(run_adapter(fit_predict))
