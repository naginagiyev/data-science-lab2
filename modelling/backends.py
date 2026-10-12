import json
import secrets
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tools.sandbox import kill_tree
from sklearn.pipeline import Pipeline
from operations.generic import TypingStep
from operations.compiler import AgentSteps
from paths import backend_models_dir, root_dir

PYTHONS = {"autogluon": root_dir / "benchmark" / "envs" / "autogluon" / ".venv" / "Scripts" / "python.exe", "flaml": root_dir / "benchmark" / "envs" / "flaml" / ".venv" / "Scripts" / "python.exe"}
WORKER = root_dir / "modelling" / "backendworker.py"

# raised when a backend cannot fit or predict
class BackendError(RuntimeError):
    pass

# prepared rows as parquet-safe columns: numbers stay numbers, other values...
# become text (missing stays missing)
def parquet_frame(frame):
    frame = frame.reset_index(drop=True).copy()
    frame.columns = [str(column) for column in frame.columns]
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: None if pd.isna(value) else str(value))
    return frame

# runs the backend worker in its own environment with a hard time-out
def call_worker(backend, arguments, timeout):
    python = PYTHONS[backend]
    if not python.exists():
        raise BackendError(f"the {backend} environment is missing ({python}); see benchmark/envs/install.py")
    process = subprocess.Popen([str(python), str(WORKER), "--backend", backend, *arguments], cwd=root_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, error = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(process)
        process.communicate()
        raise BackendError(f"{backend} took longer than {timeout:.0f} s")
    if process.returncode != 0:
        lines = error.decode("utf-8", errors="replace").strip().splitlines()
        raise BackendError(f"{backend} failed: {lines[-1] if lines else 'no message'}")

# a model trained by a backend: the preparation (agent steps + typing) runs...
# here; the backend predicts in its own environment from a short model folder
class BackendModel:
    def __init__(self, backend, prep, folder, task, n_classes):
        self.backend = backend
        self.prep = prep
        self.folder = str(folder)
        self.task = task
        self.n_classes = n_classes

    def predict_values(self, frame):
        folder = Path(self.folder)
        token = secrets.token_hex(4)
        source, target = folder / f"in-{token}.parquet", folder / f"out-{token}.parquet"
        parquet_frame(self.prep.transform(frame)).to_parquet(source)
        try:
            call_worker(self.backend, ["--command", "predict", "--data", str(source), "--model", self.folder, "--task", self.task, "--metric", "none", "--out", str(target)], 600)
            output = pd.read_parquet(target)
        finally:
            source.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
        if self.task == "regression":
            return output["prediction"].to_numpy(dtype=float)
        columns = [str(code) for code in range(self.n_classes)]
        return output.reindex(columns=columns, fill_value=0.0).to_numpy(dtype=float)

    def families(self):
        return [self.backend]

# trains a backend on the prepared training rows (row filter applied, target...
# encoded, log target applied) with the remaining model seconds; returns the...
# model and a summary with the backend's own validation score
def train_backend(backend, plan, frame, y, task, metric, seconds, seed, n_classes=None, row_filter=None, target_transform=None):
    if row_filter is not None:
        keep = row_filter.mask(frame, y)
        frame, y = frame[keep], y[keep]
    if target_transform is not None:
        y = target_transform.forward(y)
    prep = Pipeline([("agent", AgentSteps(list(plan.steps) if plan is not None else [])), ("typing", TypingStep())])
    prepared = parquet_frame(prep.fit_transform(frame, y))
    prepared["__target__"] = np.asarray(y)
    folder = backend_models_dir / secrets.token_hex(4)
    folder.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(folder / "train.parquet")
    fit_seconds = max(10.0, seconds - 15.0)
    arguments = ["--command", "fit", "--data", str(folder / "train.parquet"), "--model", str(folder), "--task", task, "--metric", metric, "--seconds", str(fit_seconds), "--seed", str(seed)]
    call_worker(backend, arguments, fit_seconds * 1.5 + 120)
    (folder / "train.parquet").unlink(missing_ok=True)
    score = json.loads((folder / "score.json").read_text(encoding="utf-8"))["score"]
    summary = {"chosen": [backend], "cv": {backend: score}, "backend": backend, "backend_folder": str(folder), "ensemble_cv": None}
    return BackendModel(backend, prep, folder, task, n_classes), summary
