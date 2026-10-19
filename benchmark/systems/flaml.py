import re
import sys
import time
import pandas as pd
from flaml import AutoML
from benchmark.systems.common import fit_seconds, run_adapter

STARTED = time.perf_counter()
METRICS = {"roc_auc": "roc_auc", "log_loss": "log_loss", "rmse": "rmse"}

# baseline FLAML: AutoML.fit with the cell budget, the fixed metric, the seed...
# and 4 threads on pd.read_csv input (JSON characters in column names are...
# replaced, as in the LightGBM baseline, because LightGBM inside FLAML refuses them)
def fit_predict(arguments):
    safe = lambda name: re.sub(r"[\[\]{}\",:]", "_", str(name))
    train = pd.read_csv(arguments.train)
    target = train.pop(arguments.target)
    tests = [pd.read_csv(path)[train.columns].rename(columns=safe) for path in arguments.tests]
    train = train.rename(columns=safe)
    automl = AutoML()
    automl.fit(
        train, target, task="regression" if arguments.task == "regression" else "classification",
        time_budget=fit_seconds(arguments, STARTED), metric=METRICS[arguments.metric], seed=arguments.seed, n_jobs=4,
        log_file_name=str(f"{arguments.workdir}/flaml.log"), verbose=0,
    )
    if arguments.task == "regression":
        return {"predictions": [automl.predict(test) for test in tests]}
    return {"predictions": [automl.predict_proba(test) for test in tests], "classes": list(automl.classes_)}

if __name__ == "__main__":
    sys.exit(run_adapter(fit_predict))
