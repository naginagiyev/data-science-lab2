import sys
import csv
import openml
import argparse
import pandas as pd
from llms.client import LlmClient
from settings import load_settings
from agents.framer import frame_task
from tools.glance import glance, glance_text
from paths import openml_cache_dir, results_dir
from benchmark.datasets import BINARY, MULTI, OPENML_DATASETS, column_as_text

MAX_ROWS = 100000
CC18 = {"diabetes": 37, "vehicle": 54, "ilpd": 1480, "qsar-biodeg": 1494, "wdbc": 1510, "balance-scale": 11, "breast-w": 15, "dresses-sales": 23381, "kc1": 1067}
COLUMNS = ["dataset", "did", "rows", "columns", "true_target", "true_task", "rules_target", "rules_task", "rules_target_ok", "rules_task_ok", "llm_target", "llm_task", "llm_target_ok", "llm_task_ok", "llm_source", "llm_cost_usd"]

# the 40 datasets of E7: every OpenML dataset of the benchmark pools (31) plus 9...
# OpenML-CC18 classification datasets; returns {name: (did, task or None)}
def evaluation_datasets():
    chosen = {name: (spec["did"], spec["task"]) for name, spec in OPENML_DATASETS.items()}
    chosen.update({name: (did, None) for name, did in CC18.items()})
    return chosen

# the raw text view (at most MAX_ROWS rows) and the default target of one...
# OpenML dataset
def load(did):
    openml.config.set_root_cache_directory(str(openml_cache_dir))
    dataset = openml.datasets.get_dataset(did, download_data=True, download_qualities=False, download_features_meta_data=False)
    frame, _, _, _ = dataset.get_data(dataset_format="dataframe")
    frame = frame.sample(MAX_ROWS, random_state=0) if len(frame) > MAX_ROWS else frame
    raw = pd.DataFrame({column: column_as_text(frame[column]) for column in frame.columns}).reset_index(drop=True)
    return raw, dataset.default_target_attribute.split(",")[0]

# the true task: from the pool spec, else binary or multi-class by the number...
# of target values (all CC18 datasets are classification)
def true_task(raw, target, task):
    if task:
        return task
    return BINARY if raw[target].replace("", pd.NA).dropna().nunique() == 2 else MULTI

# frames every dataset twice without flags (rules only, and rules + LLM) and...
# writes benchmark/results/framer.csv
def evaluate(names=None, model="gpt-4.1-mini"):
    settings = load_settings(assignments=["spend.category=experiments", f"llm.model={model}"])
    client = LlmClient(settings)
    rows = []
    for name, (did, task) in evaluation_datasets().items():
        if names and name not in names:
            continue
        raw, target = load(did)
        truth = true_task(raw, target, task)
        result = glance(raw)
        rules = frame_task(raw, result, settings)
        spent = client.usage["cost_usd"]
        llm = frame_task(raw, result, settings, client=client, glance_text=glance_text(result))
        row = {
            "dataset": name, "did": did, "rows": len(raw), "columns": raw.shape[1], "true_target": target, "true_task": truth,
            "rules_target": rules.target, "rules_task": rules.task, "rules_target_ok": rules.target == target, "rules_task_ok": rules.target == target and rules.task == truth,
            "llm_target": llm.target, "llm_task": llm.task, "llm_target_ok": llm.target == target, "llm_task_ok": llm.target == target and llm.task == truth,
            "llm_source": llm.source, "llm_cost_usd": round(client.usage["cost_usd"] - spent, 6),
        }
        print(f"{name}: rules {row['rules_target_ok']}/{row['rules_task_ok']}, llm {row['llm_target_ok']}/{row['llm_task_ok']}", flush=True)
        rows.append(row)
    with (results_dir / "framer.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows

# command line: python -m benchmark.framereval [--names ...] [--model ...]
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="E7: framer accuracy, rules vs rules + LLM.")
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--model", default="gpt-4.1-mini")
    arguments = parser.parse_args()
    rows = evaluate(arguments.names, arguments.model)
    for kind in ["rules", "llm"]:
        print(f"{kind}: target {sum(row[f'{kind}_target_ok'] for row in rows)}/{len(rows)}, target and task {sum(row[f'{kind}_task_ok'] for row in rows)}/{len(rows)}")
    print(f"LLM cost: ${sum(row['llm_cost_usd'] for row in rows):.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
