import sys
import time
import pandas as pd
from benchmark.systems.common import fit_seconds, run_adapter
from benchmark.systems.agentic import ToolFailed, keep_outputs, record_spend, run_in_wsl, write_inputs
from benchmark.systems.agentic import MODEL, WSL_BENCH, cell_cap, new_cell_folder, read_submission, read_usage

STARTED = time.perf_counter()

# the WSL script of AutoGluon Assistant (MLZero): its OpenAI config with every...
# agent on the benchmark model and the proxy, a run with the default 5...
# iterations, then every results file and the generated code are copied out
def tool_script(run_seconds, exec_timeout):
    home = f"{WSL_BENCH}/envs/mlzero"
    return "\n".join([
        f"export PATH={home}/bin:$HOME/miniforge3/bin:$PATH",
        f"{home}/bin/python - <<'PY'",
        "import os, yaml, pathlib, autogluon.assistant as assistant",
        "config = yaml.safe_load((pathlib.Path(assistant.__file__).parent / 'configs' / 'openai.yaml').read_text())",
        "for value in config.values():",
        "    if isinstance(value, dict) and 'model' in value:",
        f"        value.update(model='{MODEL}', proxy_url=os.environ['PROXY_URL'], max_tokens=16384)",
        f"config['per_execution_timeout'] = {int(exec_timeout)}",
        "pathlib.Path('config.yaml').write_text(yaml.safe_dump(config))",
        "PY",
        "echo \"stage mlzero-start $(date +%s.%N)\"",
        f"timeout {int(run_seconds)} mlzero -i input -o out -c config.yaml -n 5 -t \"$(cat input/description.md)\" -v 2 > output/mlzero.log 2>&1 < /dev/null",
        "echo \"stage mlzero-end $(date +%s.%N)\"",
        "find out -path '*conda_env*' -prune -o -name 'results.*' -printf '%T@ %p\\n' 2>/dev/null | sort -n | awk '{print $2}' > output/results.txt",
        "i=0; while read -r f; do i=$((i+1)); cp \"$f\" output/results$i.csv; done < output/results.txt",
        "mkdir -p output/code; find out -path '*conda_env*' -prune -o \\( -name '*.py' -o -name '*.sh' -o -name '*.txt' -o -name '*.log' \\) -size -300k -print"
        " | while read -r f; do cp \"$f\" \"output/code/$(echo $f | tr '/' '_')\"; done",
    ])

# runs MLZero in WSL on the deployment test file and reads its last valid results
def fit_predict(arguments):
    seconds = fit_seconds(arguments, STARTED)
    folder = new_cell_folder(arguments.workdir)
    run, exec_timeout = seconds * 0.92, max(30.0, seconds * 0.5)
    note = f"Time limit: the whole run must finish within {int(run)} seconds; each script run is stopped after {int(exec_timeout)} seconds."
    classes, _, rows = write_inputs(folder / "input", arguments, note)
    cap = cell_cap()
    run_in_wsl(folder, tool_script(run, exec_timeout), seconds, cap)
    keep_outputs(folder, arguments.out)
    meta = read_usage(folder / "usage.jsonl")
    record_spend(meta, f"mlzero {arguments.out}")
    output = folder / "output"
    listed = (output / "results.txt").read_text(encoding="utf-8").split() if (output / "results.txt").exists() else []
    meta.update(results_files=len(listed), cell_cap_usd=cap)
    errors = []
    for index in range(len(listed), 0, -1):
        try:
            prediction = read_submission(output / f"results{index}.csv", arguments.task, classes, rows)
            meta["results_used"] = listed[index - 1]
            return {"predictions": [prediction], "classes": classes, "meta": meta}
        except (ValueError, KeyError, pd.errors.ParserError) as error:
            errors.append(f"{listed[index - 1]}: {error}")
    raise ToolFailed(f"MLZero wrote no usable results ({len(listed)} files; {'; '.join(errors)[:300]})", meta)

if __name__ == "__main__":
    sys.exit(run_adapter(fit_predict))
