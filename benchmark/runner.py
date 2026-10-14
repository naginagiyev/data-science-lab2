import os
import sys
import csv
import json
import time
import shutil
import secrets
import argparse
import itertools
import threading
import subprocess
from benchmark import spend
from dotenv import dotenv_values
from settings import load_settings
from tools.runfolder import now_text
from benchmark.scoring import score_cell
from benchmark.materialize import cell_dir
from concurrent.futures import ThreadPoolExecutor
from paths import benchmark_outputs_dir, env_file, grids_dir, logs_dir, results_dir, root_dir, systems_registry_file, workdirs_dir

COLUMNS = [
    "grid", "system", "variant", "dataset", "group", "level", "mess_types", "seed", "budget_s", "status", "error",
    "metric", "score", "dummy_score", "secondary", "seconds", "llm_calls", "input_tokens", "output_tokens",
    "cost_usd", "run_dir", "git_hash", "settings_hash", "started_at",
]
THREAD_VARIABLES = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]

# reads a grid file by name (benchmark/grids/<name>.json) or path
def load_grid(name):
    path = grids_dir / f"{name}.json" if not str(name).endswith(".json") else root_dir / name
    grid = json.loads(path.read_text(encoding="utf-8"))
    grid.setdefault("name", path.stem)
    return grid

# reads the system registry: python executable, module and extra arguments...
# of every system
def load_registry(path=None):
    return json.loads((path or systems_registry_file).read_text(encoding="utf-8"))

# expands a grid into its cells: systems x datasets x levels x seeds (x mess...
# types when "mess" is a list, x budgets when the grid has "budgets")
def expand_cells(grid):
    cells = []
    messes = grid.get("mess", "mixed")
    messes = messes if isinstance(messes, list) else [messes]
    for system, dataset, level, seed, mess, budget in itertools.product(grid["systems"], grid["datasets"], grid["levels"], grid["seeds"], messes, grid.get("budgets", [grid["budget"]])):
        name, group = (dataset, grid.get("group", "dev")) if isinstance(dataset, str) else (dataset["name"], dataset["group"])
        cells.append({"system": system, "dataset": name, "group": group, "level": level, "seed": seed, "budget": budget, "mess": mess})
    return cells

# true when a cell is one of the grid's held-out cells (they run last, with...
# --holdout)
def is_holdout(cell, grid):
    return any(cell["system"] == item["system"] and cell["dataset"] == item["dataset"] and cell["level"] == item["level"] for item in grid.get("holdout", []))

# the key that says if a cell already has a result
def cell_key(row):
    return (str(row["system"]), str(row["dataset"]), str(row["group"]), str(row["level"]), str(row["mess_types"] if "mess_types" in row else row["mess"]), str(row["seed"]), str(float(row["budget_s"] if "budget_s" in row else row["budget"])))

# returns the keys of the cells that are already in the results file
def done_keys(results_file):
    if not results_file.exists():
        return set()
    with results_file.open(encoding="utf-8", newline="") as handle:
        return {cell_key(row) for row in csv.DictReader(handle)}

# appends one result row to the results CSV (writes the header first)
def append_result(results_file, row, lock):
    with lock:
        new_file = not results_file.exists()
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with results_file.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            if new_file:
                writer.writeheader()
            writer.writerow({column: row.get(column, "") for column in COLUMNS})

# returns the current git commit (short), or "" outside a repository
def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root_dir, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""

# kills a process and all its children (taskkill /T on Windows)
def kill_tree(process):
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
    else:
        process.kill()

# the environment of a system process: 4 threads, UTF-8 output, the project on...
# the path, and no API key (LLM adapters read it from .env themselves); a system...
# that reads the key only from the environment (v1) gets it with keep_key
def system_environment(threads=4, keep_key=False):
    environment = {key: value for key, value in os.environ.items() if key != "OPENAI_API_KEY"}
    if keep_key:
        environment["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or dotenv_values(env_file).get("OPENAI_API_KEY") or ""
    for name in THREAD_VARIABLES:
        environment[name] = str(threads)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONPATH"] = str(root_dir)
    return environment

# builds the command of one cell from the registry entry and the protocol...
# arguments (§4.4)
def cell_command(entry, cell, files, truth, workdir, out, extra_arguments):
    python = sys.executable if entry["python"] == "main" else str(root_dir / entry["python"])
    names = ["test_features.csv"] + (["test_features_naive.csv"] if (files / "test_features_naive.csv").exists() else [])
    tests = [str(files / name) for name in names]
    start = ["-m", entry["module"]] if "module" in entry else [str(root_dir / entry["script"])]
    return [
        python, *start, *entry.get("args", []),
        "--train", str(files / "train.csv"), "--test", ",".join(tests), "--target", truth["target"],
        "--task", truth["task"], "--metric", truth["metric"], "--budget", str(cell["budget"]), "--seed", str(cell["seed"]),
        "--workdir", str(workdir), "--out", str(out), *extra_arguments,
    ]

# runs one cell: starts the system with a hard time-out, scores its...
# predictions, appends the result row and deletes the workdir
def run_cell(cell, grid, registry, results_file, lock, log_dir):
    entry = registry[cell["system"]]
    files = cell_dir(cell["group"], cell["dataset"], cell["level"], cell["seed"], cell["mess"])
    truth = json.loads((files / "truth.json").read_text(encoding="utf-8"))
    mess = "" if cell["mess"] == "mixed" else f"-{cell['mess']}"
    cell_id = f"{cell['system']}-{cell['dataset']}-l{cell['level']}{mess}-s{cell['seed']}-b{int(cell['budget'])}".replace("+", "plus")
    workdir = workdirs_dir / secrets.token_hex(4)
    out = benchmark_outputs_dir / grid["name"] / cell_id
    shutil.rmtree(out, ignore_errors=True)
    row = {
        "grid": grid["name"], "system": cell["system"], "variant": entry.get("variant", ""), "dataset": cell["dataset"],
        "group": cell["group"], "level": cell["level"], "mess_types": cell["mess"], "seed": cell["seed"],
        "budget_s": float(cell["budget"]), "git_hash": git_hash(), "started_at": now_text(),
    }
    settings = load_settings(assignments=grid.get("settings", []))
    if entry.get("llm") and not spend.check_allowed(0.0, settings.spend.model_copy(update={"category": "experiments"}))[0]:
        row.update(status="skipped-spend", error="the experiments spend cap is reached")
        append_result(results_file, row, lock)
        return row
    extra = [argument for item in grid.get("settings", []) for argument in ("--set", item)] if entry.get("settings") else []
    command = cell_command(entry, cell, files, truth, workdir, out, extra)
    timeout = entry.get("timeout", cell["budget"] * 1.25 + 60)
    log_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with (log_dir / f"{cell_id}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=root_dir, stdout=log, stderr=subprocess.STDOUT, env=system_environment(keep_key=entry.get("needs_key", False)))
        try:
            process.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            kill_tree(process)
            process.wait()
            timed_out = True
    seconds = time.perf_counter() - start
    meta_file = out / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    try:
        result = score_cell(truth["task"], files / "train.csv", files / "test_labels.csv", out / "predictions0.csv", truth["target"])
    except Exception as error:
        result = {"status": "failed", "error": f"scoring: {type(error).__name__}: {error}"[:300], "metric": truth["metric"], "score": None, "dummy_score": None, "secondary": {}}
    if timed_out:
        result.update(status="failed", error=f"time-out after {timeout:.0f} s", score=result["dummy_score"])
    elif result["status"] == "failed" and meta.get("error"):
        result["error"] = f"{meta['error']} ({result['error']})"
    row.update(result)
    row.update(
        secondary=json.dumps(result["secondary"]), seconds=round(seconds, 2), llm_calls=meta.get("llm_calls", 0),
        input_tokens=meta.get("input_tokens", 0), output_tokens=meta.get("output_tokens", 0), cost_usd=meta.get("cost_usd", 0.0),
        run_dir=meta.get("run_dir", ""), settings_hash=meta.get("settings_hash", ""),
    )
    if (files / "test_features_naive.csv").exists():
        naive = score_cell(truth["task"], files / "train.csv", files / "test_labels.csv", out / "predictions1.csv", truth["target"])
        row["secondary"] = json.dumps(dict(result["secondary"], naive_score=naive["score"]))
    append_result(results_file, row, lock)
    shutil.rmtree(workdir, ignore_errors=True)
    return row

# runs every cell of a grid that has no result yet, with at most `workers`...
# cells in parallel (held-out cells only with holdout=True, then only them);...
# returns the new result rows
def run_grid(name, workers=None, registry_path=None, results_folder=None, systems=None, limit=None, holdout=False):
    grid = load_grid(name)
    registry = load_registry(registry_path)
    results_file = (results_folder or results_dir) / f"{grid['name']}.csv"
    finished = done_keys(results_file)
    cells = [cell for cell in expand_cells(grid) if cell_key(cell) not in finished and (not systems or cell["system"] in systems) and is_holdout(cell, grid) == holdout]
    cells = cells[:limit] if limit else cells
    lock = threading.Lock()
    log_dir = logs_dir / grid["name"]
    print(f"Grid {grid['name']}: {len(cells)} cells to run, {len(finished)} already done.", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=min(workers or grid.get("workers", 1), 2)) as pool:
        for row in pool.map(lambda cell: run_cell(cell, grid, registry, results_file, lock, log_dir), cells):
            print(f"{row['system']:<12} {row['dataset']:<18} level {row['level']} seed {row['seed']}: {row['status']} {row.get('score', '')} ({row.get('seconds', '')} s)", flush=True)
            rows.append(row)
    return rows

# checks a grid without running it: every cell has its files and a known...
# system, and the first cell's system starts (it is stopped after `seconds`;...
# systems that call an LLM outside the v2 spend control, like v1, are not...
# started); returns a list of problems
def check_grid(name, seconds=20, registry_path=None):
    grid = load_grid(name)
    registry = load_registry(registry_path)
    cells = expand_cells(grid)
    problems = [f"unknown system {system}" for system in sorted({cell["system"] for cell in cells} - set(registry))]
    missing = [cell for cell in cells if not (cell_dir(cell["group"], cell["dataset"], cell["level"], cell["seed"], cell["mess"]) / "truth.json").exists()]
    problems += [f"{len(missing)} of {len(cells)} cells have no files (first: {missing[0]['dataset']} level {missing[0]['level']} {missing[0]['mess']})"] if missing else []
    if problems:
        return problems
    cell = cells[0]
    if registry[cell["system"]].get("llm") and not registry[cell["system"]].get("settings"):
        return problems
    files = cell_dir(cell["group"], cell["dataset"], cell["level"], cell["seed"], cell["mess"])
    truth = json.loads((files / "truth.json").read_text(encoding="utf-8"))
    workdir = workdirs_dir / secrets.token_hex(4)
    out = workdirs_dir / f"check-{secrets.token_hex(4)}"
    command = cell_command(registry[cell["system"]], cell, files, truth, workdir, out, [])
    process = subprocess.Popen(command, cwd=root_dir, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=system_environment())
    try:
        _, error = process.communicate(timeout=seconds)
        if process.returncode != 0:
            lines = error.decode("utf-8", errors="replace").strip().splitlines()
            problems.append(f"the first cell ({cell['system']}) failed: {lines[-1] if lines else 'no message'}")
    except subprocess.TimeoutExpired:
        kill_tree(process)
        process.communicate()
    shutil.rmtree(workdir, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    return problems

# command line: python -m benchmark.runner --grid smoke [--workers 2]
def main():
    parser = argparse.ArgumentParser(description="Run a benchmark grid (resumable).")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--systems", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--holdout", action="store_true", help="run only the grid's held-out cells")
    parser.add_argument("--check", action="store_true", help="check the grid files and start its first cell, without running it")
    arguments = parser.parse_args()
    if arguments.check:
        problems = check_grid(arguments.grid)
        print(f"Grid {arguments.grid}: " + ("ok" if not problems else "; ".join(problems)), flush=True)
        return 1 if problems else 0
    run_grid(arguments.grid, arguments.workers, systems=arguments.systems, limit=arguments.limit, holdout=arguments.holdout)
    return 0

if __name__ == "__main__":
    sys.exit(main())
