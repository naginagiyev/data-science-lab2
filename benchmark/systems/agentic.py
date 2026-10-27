import os
import json
import random
import secrets
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from benchmark import spend
from dotenv import dotenv_values
from settings import load_settings
from paths import env_file, root_dir
from benchmark.systems.common import label_key

DISTRIBUTION = "Ubuntu"
WSL_BENCH = "$HOME/bench"
MODEL = "gpt-4.1-mini"
CELL_CAP_USD = 0.25
METRIC_TEXT = {
    "roc_auc": "ROC AUC (higher is better)",
    "log_loss": "multi-class log loss (lower is better)",
    "rmse": "root mean squared error (lower is better)",
}

# a failed tool run that still carries its LLM usage, so the result row shows...
# the tokens and dollars it spent
class ToolFailed(RuntimeError):

    # keeps the message and the usage
    def __init__(self, message, meta):
        super().__init__(message)
        self.meta = meta

# turns a Windows path into the path WSL sees (C:\a b -> /mnt/c/a b)
def wsl_path(path):
    text = str(Path(path).resolve())
    if os.name != "nt":
        return text
    return f"/mnt/{text[0].lower()}{text[2:]}".replace("\\", "/")

# the classes of a classification target as label keys, sorted; None for...
# regression
def target_classes(train, target, task):
    if task == "regression":
        return None
    return sorted({label_key(value) for value in train[target]})

# the task text both agentic tools get: files, target, task, metric and the...
# exact submission format, so they get the same information as the other systems
def task_text(target, task, metric, classes, rows, time_note=""):
    if task == "regression":
        output = "one column named `prediction` with the predicted value"
    else:
        names = ", ".join(f"`{name}`" for name in classes)
        output = f"one column per class ({names}) holding the predicted probability of that class (each row sums to 1)"
    return (
        f"# Task\n\nPredict the column `{target}` of `train.csv` for the rows of `test.csv` "
        f"({'regression' if task == 'regression' else 'classification'}).\n\n"
        f"- `train.csv`: the training rows with the target column `{target}`.\n"
        f"- `test.csv`: {rows} rows with the same feature columns and no target.\n"
        f"- `sample_submission.csv`: the required output format.\n\n"
        f"The evaluation metric is {METRIC_TEXT[metric]}. Write the predictions for all {rows} test rows, in the "
        f"original row order, as a CSV file with {output}. Do not add an id or index column.\n"
        + (f"\n{time_note}\n" if time_note else "")
    )

# a sample submission with the exact columns the scorer reads
def sample_submission(task, classes, rows):
    if task == "regression":
        return pd.DataFrame({"prediction": np.zeros(rows)})
    return pd.DataFrame(np.full((rows, len(classes)), 1.0 / len(classes)), columns=classes)

# writes the input folder of a cell: train.csv, test.csv, sample_submission.csv...
# and description.md; returns the target classes, the task text and the test rows
def write_inputs(folder, arguments, time_note=""):
    folder.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(arguments.train, dtype=str, keep_default_na=False)
    test = pd.read_csv(arguments.tests[0], dtype=str, keep_default_na=False)
    classes = target_classes(train, arguments.target, arguments.task)
    text = task_text(arguments.target, arguments.task, arguments.metric, classes, len(test), time_note)
    for name, source in [("train.csv", arguments.train), ("test.csv", arguments.tests[0])]:
        (folder / name).write_bytes(Path(source).read_bytes())
    sample_submission(arguments.task, classes, len(test)).to_csv(folder / "sample_submission.csv", index=False)
    (folder / "description.md").write_text(text, encoding="utf-8", newline="\n")
    return classes, text, len(test)

# reads a submission file into the protocol form: a probability matrix in the...
# order of `classes` (a label column becomes one-hot), or regression values
def read_submission(path, task, classes, rows):
    frame = pd.read_csv(path)
    frame = frame.drop(columns=[column for column in frame.columns if str(column).lower() in ("id", "index", "unnamed: 0")])
    if len(frame) != rows:
        raise ValueError(f"the submission has {len(frame)} rows for {rows} test rows")
    if task == "regression":
        column = "prediction" if "prediction" in frame else frame.columns[-1]
        return pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    columns = {label_key(column): column for column in frame.columns}
    if all(name in columns for name in classes):
        values = frame[[columns[name] for name in classes]].to_numpy(dtype=float)
        return values / np.clip(values.sum(axis=1, keepdims=True), 1e-12, None)
    labels = [label_key(value) for value in frame.iloc[:, -1]]
    return np.array([[1.0 if label == name else 0.0 for name in classes] for label in labels])

# sums the usage log of the proxy: calls, tokens and dollars
def read_usage(path):
    meta = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "refused_calls": 0}
    if not Path(path).exists():
        return meta
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["status"].startswith("refused"):
            meta["refused_calls"] += 1
            continue
        meta["llm_calls"] += 1
        meta["input_tokens"] += record["input_tokens"]
        meta["output_tokens"] += record["output_tokens"]
        meta["cost_usd"] += record["cost_usd"]
    return meta

# writes the usage of one cell into the global spend ledger (experiments)
def record_spend(meta, source):
    if meta["cost_usd"] > 0:
        settings = load_settings(assignments=["spend.category=experiments"])
        spend.record(meta["cost_usd"], "experiments", MODEL, meta["input_tokens"], meta["output_tokens"], source, settings.spend)

# the dollar cap of one cell: the per-cell cap, or less when the experiments...
# cap of the ledger is nearly used up
def cell_cap():
    settings = load_settings(assignments=["spend.category=experiments"])
    spent = spend.totals()
    left = min(settings.spend.experiments_cap_usd - spent["experiments"], settings.spend.total_cap_usd - spent["total"])
    if left <= 0.01:
        raise RuntimeError("the experiments spend cap is reached")
    return round(min(CELL_CAP_USD, left), 4)

# runs a tool in WSL (or directly when the benchmark itself runs on Linux):...
# copies input/ and tool.sh into a fresh folder on the Linux disk, starts the...
# proxy on a free loopback port, runs the tool on 4 cores under a hard time...
# limit, stops the proxy and copies output/ and usage.jsonl back; the real key...
# reaches the script only through the environment (WSLENV on Windows)
def run_in_wsl(cell_folder, tool_script, seconds, cap):
    port = random.randint(20000, 59000)
    cores = random.choice(["0-3", "4-7"])
    source = wsl_path(cell_folder)
    proxy = f"{WSL_BENCH}/envs/aide/bin/python '{wsl_path(root_dir / 'benchmark' / 'llmproxy.py')}'"
    script = "\n".join([
        "set -u",
        f"W={WSL_BENCH}/cells/{cell_folder.name}",
        f"find {WSL_BENCH}/cells -mindepth 1 -maxdepth 1 -mmin +60 -exec rm -rf {{}} + 2>/dev/null",
        "rm -rf $W && mkdir -p $W/output",
        f"cp -r '{source}/input' '{source}/tool.sh' $W/ && cd $W",
        f"{proxy} --port {port} --log $W/usage.jsonl --cap {cap} &",
        "PROXY=$!",
        f"for i in $(seq 100); do (echo > /dev/tcp/127.0.0.1/{port}) 2>/dev/null && break; sleep 0.1; done",
        f"export OPENAI_BASE_URL=http://127.0.0.1:{port}/v1 OPENAI_API_KEY=proxy-key PROXY_URL=http://127.0.0.1:{port}/v1",
        "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONIOENCODING=utf-8",
        "echo \"stage tool-start $(date +%s.%N)\"",
        f"timeout --kill-after=15 {int(seconds)} taskset -c {cores} bash tool.sh",
        "STATUS=$?",
        "echo \"stage tool-end $(date +%s.%N) status $STATUS\"",
        "kill $PROXY 2>/dev/null",
        f"cp -r $W/output/. '{source}/output/' 2>/dev/null; cp $W/usage.jsonl '{source}/' 2>/dev/null",
        "echo \"stage copied $(date +%s.%N)\"",
        "cd / && (rm -rf $W &)",
        "exit $STATUS",
    ])
    (cell_folder / "output").mkdir(exist_ok=True)
    (cell_folder / "tool.sh").write_text(tool_script + "\n", encoding="utf-8", newline="\n")
    (cell_folder / "run.sh").write_text(script + "\n", encoding="utf-8", newline="\n")
    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(env_file).get("OPENAI_API_KEY")
    environment = dict(os.environ, BENCH_OPENAI_KEY=key or "", WSLENV="BENCH_OPENAI_KEY/u")
    command = ["wsl.exe", "-d", DISTRIBUTION, "--", "bash", wsl_path(cell_folder / "run.sh")] if os.name == "nt" else ["bash", str(cell_folder / "run.sh")]
    with (cell_folder / "tool.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode

# copies the tool log, the usage log, the task inputs' description and the...
# small outputs (code, logs, submissions) next to the predictions, so every...
# agentic run can be checked after the workdir is deleted
def keep_outputs(folder, out):
    target = Path(out) / "tool"
    target.mkdir(parents=True, exist_ok=True)
    for name in ["tool.log", "usage.jsonl", "tool.sh", "input/description.md"]:
        if (folder / name).exists():
            (target / Path(name).name).write_bytes((folder / name).read_bytes())
    for path in (folder / "output").rglob("*"):
        if path.is_file() and path.stat().st_size < 5_000_000:
            destination = target / "output" / path.relative_to(folder / "output")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())

# a fresh cell folder inside the workdir (one per run)
def new_cell_folder(workdir):
    folder = Path(workdir) / f"cell-{secrets.token_hex(3)}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder
