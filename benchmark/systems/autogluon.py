import sys
import time
from pathlib import Path
from benchmark.systems.common import fit_seconds, run_adapter
from autogluon.tabular import TabularDataset, TabularPredictor

STARTED = time.perf_counter()
PROBLEMS = {"binary-classification": "binary", "multi-class-classification": "multiclass", "regression": "regression"}
METRICS = {"roc_auc": "roc_auc", "log_loss": "log_loss", "rmse": "root_mean_squared_error"}

# baseline AutoGluon: TabularPredictor with the medium_quality preset, the fixed...
# metric, the cell budget and 4 CPUs; folds are fitted one after another...
# (no Ray), and the model folder lives in the short workdir
def fit_predict(arguments):
    train = TabularDataset(arguments.train)
    tests = [TabularDataset(path) for path in arguments.tests]
    predictor = TabularPredictor(
        label=arguments.target, problem_type=PROBLEMS[arguments.task], eval_metric=METRICS[arguments.metric],
        path=str(Path(arguments.workdir) / "ag"), verbosity=1,
    )
    predictor.fit(
        train, time_limit=fit_seconds(arguments, STARTED), presets="medium_quality", num_cpus=4, num_gpus=0,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    if arguments.task == "regression":
        return {"predictions": [predictor.predict(test).to_numpy() for test in tests]}
    classes = list(predictor.class_labels)
    return {"predictions": [predictor.predict_proba(test)[classes].to_numpy() for test in tests], "classes": classes}

if __name__ == "__main__":
    sys.exit(run_adapter(fit_predict))
