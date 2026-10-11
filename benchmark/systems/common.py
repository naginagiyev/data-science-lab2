import sys
import json
import time
import argparse
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

# the adapter protocol: every system gets these arguments (§4.4); extra...
# arguments (for example --variant) can be added by the adapter
def parse_arguments(extra=None):
    parser = argparse.ArgumentParser()
    for name in ["--train", "--test", "--target", "--task", "--metric", "--workdir", "--out"]:
        parser.add_argument(name, required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    for name, options in (extra or {}).items():
        parser.add_argument(name, **options)
    arguments = parser.parse_args()
    arguments.tests = [path for path in arguments.test.split(",") if path]
    return arguments

# the key of a class label: text without outer spaces, and integer-valued...
# numbers written as integers ("1", "1.0" and 1 all become "1")
def label_key(value):
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if np.isfinite(number) and number.is_integer():
        return str(int(number))
    return text

# writes the predictions of one test file: one probability column per class...
# label plus "label" for classification, "prediction" for regression
def write_predictions(path, task, prediction, classes=None):
    if task == "regression":
        frame = pd.DataFrame({"prediction": np.asarray(prediction, dtype=float).ravel()})
    else:
        probabilities = np.asarray(prediction, dtype=float)
        keys = [label_key(label) for label in classes]
        frame = pd.DataFrame(probabilities, columns=keys)
        frame["label"] = [keys[index] for index in probabilities.argmax(axis=1)]
    frame.to_csv(path, index=False, lineterminator="\n")

# runs an adapter: calls fit_predict(arguments), writes predictions<i>.csv for...
# every test file and meta.json (seconds, ok, error, LLM usage); returns the...
# exit code
def run_adapter(fit_predict, extra=None):
    arguments = parse_arguments(extra)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(arguments.workdir).mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    meta = {"ok": False, "error": None, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    try:
        result = fit_predict(arguments)
        for index, prediction in enumerate(result["predictions"]):
            write_predictions(out / f"predictions{index}.csv", arguments.task, prediction, result.get("classes"))
        meta.update(result.get("meta", {}))
        meta["ok"] = True
    except Exception as error:
        meta["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    meta["seconds"] = round(time.perf_counter() - start, 3)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    sys.stdout.flush()
    return 0 if meta["ok"] else 1
