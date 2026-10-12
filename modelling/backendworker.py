import sys
import json
import pickle
import argparse
import pandas as pd
from pathlib import Path

TARGET = "__target__"
AG_PROBLEMS = {"binary-classification": "binary", "multi-class-classification": "multiclass", "regression": "regression"}
AG_METRICS = {"roc_auc": "roc_auc", "log_loss": "log_loss", "rmse": "root_mean_squared_error"}

# fits AutoGluon (medium_quality preset, folds one after another, 4 CPUs) and...
# returns its validation score on the run's metric scale
def fit_autogluon(train, arguments):
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor(label=TARGET, problem_type=AG_PROBLEMS[arguments.task], eval_metric=AG_METRICS[arguments.metric], path=str(Path(arguments.model) / "ag"), verbosity=1)
    predictor.fit(train, time_limit=arguments.seconds, presets="medium_quality", num_cpus=4, num_gpus=0, ag_args_ensemble={"fold_fitting_strategy": "sequential_local"})
    board = predictor.leaderboard(silent=True)
    score = float(board.loc[board["model"] == predictor.model_best, "score_val"].iloc[0])
    return abs(score) if arguments.metric in {"rmse", "log_loss"} else score

# fits FLAML with the time budget, the metric, the seed and 4 threads and...
# returns its validation score on the run's metric scale
def fit_flaml(train, arguments):
    from flaml import AutoML
    target = train.pop(TARGET)
    automl = AutoML()
    automl.fit(train, target, task="regression" if arguments.task == "regression" else "classification", time_budget=arguments.seconds, metric=arguments.metric, seed=arguments.seed, n_jobs=4, log_file_name=str(Path(arguments.model) / "flaml.log"), verbose=0)
    (Path(arguments.model) / "flaml.pkl").write_bytes(pickle.dumps(automl))
    return 1.0 - automl.best_loss if arguments.metric == "roc_auc" else float(automl.best_loss)

# predicts prepared rows: class probabilities (columns are the class codes) or...
# regression values
def predict(frame, arguments):
    if arguments.backend == "autogluon":
        from autogluon.tabular import TabularPredictor
        model = TabularPredictor.load(str(Path(arguments.model) / "ag"))
        if arguments.task == "regression":
            return pd.DataFrame({"prediction": model.predict(frame).to_numpy()})
        return model.predict_proba(frame).rename(columns=str).reset_index(drop=True)
    model = pickle.loads((Path(arguments.model) / "flaml.pkl").read_bytes())
    if arguments.task == "regression":
        return pd.DataFrame({"prediction": model.predict(frame)})
    return pd.DataFrame(model.predict_proba(frame), columns=[str(label) for label in model.classes_])

# command line inside the backend environment: fit (train.parquet -> model...
# folder and score.json) or predict (input parquet -> output parquet)
def main():
    parser = argparse.ArgumentParser()
    for name in ["command", "backend", "data", "model", "task", "metric"]:
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    arguments = parser.parse_args()
    frame = pd.read_parquet(arguments.data)
    if arguments.command == "fit":
        fit = fit_autogluon if arguments.backend == "autogluon" else fit_flaml
        score = fit(frame, arguments)
        (Path(arguments.model) / "score.json").write_text(json.dumps({"score": score}), encoding="utf-8")
    else:
        predict(frame, arguments).to_parquet(arguments.out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
