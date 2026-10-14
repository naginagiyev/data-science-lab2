import sys
import json
import argparse
import threading
from pathlib import Path
from benchmark import runner
from paths import workdirs_dir

DATASET = "concrete_compressive_strength"
SYSTEMS = ["autogluon", "full", "full+ag"]

# the main decisions of a v2 run folder in plain words: leak and id drops,...
# and the gate's final verdict on every cleaning step (the last round wins)
def decision_lines(run_dir, limit=12):
    drops, steps = [], {}
    for item in (json.loads(line) for line in (Path(run_dir) / "decisions.jsonl").read_text(encoding="utf-8").splitlines()):
        if item["stage"] == "leakguard" and item["accepted"]:
            drops.append(f"{item['subject']} dropped ({item['reason']})")
        elif item["stage"] == "gate" and item["kind"] == "step":
            steps[item["subject"]] = f"{item['subject']}: {'kept' if item['accepted'] else 'removed'} by the gate"
    return (drops + list(steps.values()))[:limit]

# runs autogluon, full and full+ag on the same level-3 pilot file with the same...
# budget (the LLM answers come from the pilot's cache), prints the three scores...
# and the main decisions of the full run
def showcase(dataset=DATASET, budget=120):
    grid = {"name": "showcase", "settings": ["spend.category=experiments", "llm.model=gpt-4.1-mini"]}
    registry = runner.load_registry()
    results_file = workdirs_dir / "showcase.csv"
    results_file.unlink(missing_ok=True)
    rows = {}
    for system in SYSTEMS:
        cell = {"system": system, "dataset": dataset, "group": "A", "level": 3, "seed": 0, "budget": budget, "mess": "mixed"}
        rows[system] = runner.run_cell(cell, grid, registry, results_file, threading.Lock(), workdirs_dir / "showcase-logs")
    metric = rows["full"]["metric"]
    print(f"\n{dataset}, mess level 3, {budget} s per system, metric {metric} (dummy {rows['full']['dummy_score']:.4f}):")
    for system, row in rows.items():
        print(f"  {system:<10} {row['status']:<7} {row['score']:.4f}   LLM cost ${float(row.get('cost_usd') or 0):.4f}")
    print("\nMain decisions of the full run:")
    for line in decision_lines(rows["full"]["run_dir"]):
        print(f"  - {line}")
    return rows

# command line: python -m benchmark.showcase [--dataset ...] [--budget 120]
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Demo: AutoGluon vs full vs full+ag on one messy file.")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--budget", type=int, default=120)
    arguments = parser.parse_args()
    showcase(arguments.dataset, arguments.budget)
    return 0

if __name__ == "__main__":
    sys.exit(main())
