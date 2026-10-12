import sys
import json
import time
import psutil
import argparse
import subprocess
from paths import data_dir, performance_dir, results_dir, root_dir, tests_dir

FILES = {"creditcardfraud": (data_dir / "creditcardfraud.csv", "Class"), "million": (performance_dir / "million.csv", "y")}

# the resident memory of a process and all its children, in bytes
def tree_memory(process):
    total = 0
    try:
        items = [process] + process.children(recursive=True)
    except psutil.Error:
        return 0
    for item in items:
        try:
            total += item.memory_info().rss
        except psutil.Error:
            pass
    return total

# writes the 1,000,000-row torture file once
def million_file():
    path = FILES["million"][0]
    if not path.exists():
        sys.path.insert(0, str(tests_dir))
        from torture import million_rows
        performance_dir.mkdir(parents=True, exist_ok=True)
        million_rows(performance_dir)
    return path

# runs one file through the CLI with the given variant and budget, samples the...
# peak memory of the process tree, and reads the stage times of the run folder
def measure(name, variant, budget):
    path, target = FILES[name]
    if name == "million":
        million_file()
    out = performance_dir / "runs" / f"{name}-{variant}"
    command = [sys.executable, str(root_dir / "main.py"), "run", str(path), "--target", target, "--task", "binary-classification", "--variant", variant, "--budget", str(budget), "--out", str(out)]
    start = time.perf_counter()
    process = psutil.Popen(command, cwd=root_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = 0
    while process.poll() is None:
        peak = max(peak, tree_memory(process))
        time.sleep(0.5)
    seconds = time.perf_counter() - start
    folder = sorted(out.iterdir())[-1]
    scores = json.loads((folder / "scores.json").read_text(encoding="utf-8")) if (folder / "scores.json").exists() else {}
    test = (scores.get("test") or {}).get("test")
    return {"file": name, "variant": variant, "budget_s": budget, "exit_code": process.returncode, "wall_s": round(seconds, 1), "peak_memory_gb": round(peak / 1e9, 2), "stage_s": scores.get("seconds", {}), "test_score": test, "run_dir": str(folder)}

# command line: measures the chosen files and writes performance.json
def main():
    parser = argparse.ArgumentParser(description="Measure time and peak memory of large runs.")
    parser.add_argument("--files", nargs="*", default=list(FILES))
    parser.add_argument("--variant", default="rules")
    parser.add_argument("--budget", type=float, default=300)
    arguments = parser.parse_args()
    results = []
    for name in arguments.files:
        result = measure(name, arguments.variant, arguments.budget)
        print(json.dumps(result), flush=True)
        results.append(result)
    (results_dir / "performance.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
