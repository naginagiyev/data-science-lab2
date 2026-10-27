import sys
import json
import time
from benchmark.systems.common import fit_seconds, run_adapter
from benchmark.systems.agentic import ToolFailed, keep_outputs, record_spend, run_in_wsl, write_inputs
from benchmark.systems.agentic import MODEL, WSL_BENCH, cell_cap, new_cell_folder, read_submission, read_usage

STARTED = time.perf_counter()

# the WSL script of AIDE: a wrapper points AIDE's fixed OpenAI address at the...
# proxy (nothing else changes), the agent searches until its share of the time...
# is up, then the best solution it saved is run once more to write its own...
# submission (the last submission belongs to the last step, not the best one)
def tool_script(search_seconds, rerun_seconds, exec_timeout):
    aide = f"{WSL_BENCH}/envs/aide/bin"
    return "\n".join([
        "cp input/description.md task.md",
        "cat > aiderun.py <<'PY'",
        "import os, sys, aide.run, aide.backend.backend_openai as backend",
        "backend.OPENAI_BASE_URL = os.environ['OPENAI_BASE_URL']",
        "sys.exit(aide.run.run())",
        "PY",
        f"timeout {int(search_seconds)} {aide}/python aiderun.py data_dir=input desc_file=task.md log_dir=logs workspace_dir=ws exp_name=cell "
        f"agent.code.model={MODEL} agent.feedback.model={MODEL} exec.timeout={int(exec_timeout)} generate_report=False > output/aide.log 2>&1",
        "cp logs/0-cell/best_solution.py logs/0-cell/journal.json output/ 2>/dev/null",
        "cp ws/0-cell/working/submission.csv output/last_submission.csv 2>/dev/null",
        "if [ -f logs/0-cell/best_solution.py ]; then",
        "  rm -f ws/0-cell/working/submission.csv",
        f"  (cd ws/0-cell && timeout {int(rerun_seconds)} {aide}/python ../../logs/0-cell/best_solution.py > ../../output/rerun.log 2>&1)",
        "  cp ws/0-cell/working/submission.csv output/submission.csv 2>/dev/null",
        "fi",
    ])

# the number of solutions AIDE wrote and how many of them failed
def journal_counts(path):
    if not path.exists():
        return {"steps": 0, "buggy_steps": 0}
    nodes = json.loads(path.read_text(encoding="utf-8")).get("nodes", [])
    return {"steps": len(nodes), "buggy_steps": sum(1 for node in nodes if node.get("is_buggy"))}

# runs AIDE in WSL on the deployment test file and reads its submission
def fit_predict(arguments):
    seconds = fit_seconds(arguments, STARTED)
    folder = new_cell_folder(arguments.workdir)
    search, rerun, exec_timeout = seconds * 0.7, seconds * 0.22, max(30.0, seconds * 0.22)
    note = f"Time limit: the whole search must finish within {int(search)} seconds; each script run is stopped after {int(exec_timeout)} seconds."
    classes, _, rows = write_inputs(folder / "input", arguments, note)
    cap = cell_cap()
    run_in_wsl(folder, tool_script(search, rerun, exec_timeout), seconds, cap)
    keep_outputs(folder, arguments.out)
    meta = read_usage(folder / "usage.jsonl")
    record_spend(meta, f"aide {arguments.out}")
    meta.update(journal_counts(folder / "output" / "journal.json"), cell_cap_usd=cap)
    best, last = folder / "output" / "submission.csv", folder / "output" / "last_submission.csv"
    meta["submission"] = "best" if best.exists() else ("last" if last.exists() else "none")
    if meta["submission"] == "none":
        raise ToolFailed(f"AIDE wrote no submission ({meta['steps']} steps, {meta['buggy_steps']} buggy)", meta)
    prediction = read_submission(best if best.exists() else last, arguments.task, classes, rows)
    return {"predictions": [prediction], "classes": classes, "meta": meta}

if __name__ == "__main__":
    sys.exit(run_adapter(fit_predict))
