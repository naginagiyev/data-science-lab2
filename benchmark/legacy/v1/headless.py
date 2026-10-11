import os
import sys
import shutil
import cloudpickle
import numpy as np
import onnxruntime
import pandas as pd
from pathlib import Path
from benchmark.systems.common import run_adapter

FROZEN_DIR = Path(__file__).resolve().parent
COPY_SKIP = {".venv", "__pycache__", "headless.py", "data", "models", "sandbox"}
TASK_TYPES = {"binary-classification": "binary-classification", "multi-class-classification": "multi-class-classification", "regression": "regression"}

# copies the frozen v1 code into a fresh workdir and creates the data, models...
# and sandbox folders that git does not keep
def prepare_workdir(workdir):
    copy = Path(workdir) / "v1"
    shutil.copytree(FROZEN_DIR, copy, ignore=lambda folder, names: [name for name in names if name in COPY_SKIP])
    for name in ["data", "models", "sandbox"]:
        (copy / name).mkdir(exist_ok=True)
    return copy

# makes python import the v1 modules (paths, tools, agents, llms) from the copy...
# and not from this project or the frozen folder
def use_copy(copy):
    project = str(FROZEN_DIR.parents[2])
    sys.path[:] = [str(copy)] + [entry for entry in sys.path if entry not in {str(FROZEN_DIR), project, ""}]
    for name in list(sys.modules):
        if name.split(".")[0] in {"paths", "schemas", "tools", "agents", "llms", "configuration", "main"}:
            del sys.modules[name]
    os.chdir(copy)

# runs the v1 steps after target and task detection (they are given): EDA,...
# configuration, processing plan and code, split, training plan and code,...
# tuning; the v1 modules are imported here because they exist only in the copy
def run_v1(copy, train_file, target, task):
    dataset_path = copy / "data" / "dataset.csv"
    shutil.copy(train_file, dataset_path)
    from main import run_eda
    from agents.planner import Planner
    from agents.trainagent import TrainAgent
    from agents.configfiller import ConfigFiller
    from agents.processoragent import ProcessorAgent
    from tools.datasplitter import split as split_data
    from agents.evaluationagent import EvaluationAgent
    run_eda(str(dataset_path), target, TASK_TYPES[task])
    ConfigFiller().fill(str(dataset_path), target, TASK_TYPES[task])
    planner = Planner()
    planner.create_pre_processing_plan()
    ProcessorAgent().preprocess()
    split_data()
    planner.create_training_plan()
    TrainAgent().train()
    EvaluationAgent().tune()

# transforms raw test rows with v1's preprocessor; v1 keeps the target in its...
# processed data, so a placeholder target is added and removed again
def transform(preprocessor, frame, target, placeholder):
    frame = frame.copy()
    frame[target] = placeholder
    output = preprocessor.transform(frame)
    if isinstance(output, pd.DataFrame):
        return output.drop(columns=[target], errors="ignore")
    return pd.DataFrame(output)

# predicts with v1's ONNX model; the probability output is used when it exists,...
# otherwise the labels (v1's real behaviour, B4)
def predict(session, features):
    inputs = session.get_inputs()
    if len(inputs) == 1:
        feed = {inputs[0].name: features.to_numpy(dtype="float32")}
    else:
        feed = {item.name: features[[item.name]].to_numpy(dtype="float32") for item in inputs}
    outputs = session.run(None, feed)
    if len(outputs) > 1 and isinstance(outputs[1], list) and outputs[1] and isinstance(outputs[1][0], dict):
        classes = list(outputs[1][0].keys())
        return np.array([[row[label] for label in classes] for row in outputs[1]]), classes
    return np.asarray(outputs[0]).ravel(), None

# the adapter: runs v1 in a copied workdir, then predicts every test file with...
# models/preprocessor.pkl and models/model.onnx
def fit_predict(arguments):
    train_file = Path(arguments.train).resolve()
    tests = [pd.read_csv(Path(path).resolve()) for path in arguments.tests]
    placeholder = pd.read_csv(train_file)[arguments.target].iloc[0]
    copy = prepare_workdir(arguments.workdir)
    use_copy(copy)
    run_v1(copy, train_file, arguments.target, arguments.task)
    with open(copy / "models" / "preprocessor.pkl", "rb") as handle:
        preprocessor = cloudpickle.load(handle)
    session = onnxruntime.InferenceSession(str(copy / "models" / "model.onnx"))
    predictions, classes = [], None
    for test in tests:
        values, classes = predict(session, transform(preprocessor, test, arguments.target, placeholder))
        if arguments.task != "regression" and classes is None:
            classes = sorted(set(values.tolist()))
            values = np.array([[1.0 if value == label else 0.0 for label in classes] for value in values])
        predictions.append(values)
    return {"predictions": predictions, "classes": classes}

if __name__ == "__main__":
    sys.exit(run_adapter(fit_predict))
