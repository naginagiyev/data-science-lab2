import sys
import json
import argparse
import numpy as np
import pandas as pd
from orchestrator import run
from settings import load_settings
from tools.reader import read_table
from tools.runfolder import RunFolder
from benchmark.datasets import GROUP_A
from benchmark.materialize import cell_dir
from operations.compiler import AgentSteps
from paths import results_dir, workdirs_dir

REPAIR_TYPES = ["units", "mixed_units", "packed", "dates", "booleans", "spellings", "fake_missing", "impossible"]
MISSING_TYPES = {"fake_missing", "impossible"}
COLUMNS = ["dataset", "mess", "variant", "column", "cells", "repaired", "accuracy", "false_nan_rate", "steps"]

# true when a prepared value equals the original value: numbers within 1e-6...
# relative, other values as equal text
def same(prepared, original):
    if prepared is None or (isinstance(prepared, float) and np.isnan(prepared)):
        return False
    try:
        number, truth = float(prepared), float(original)
        return bool(np.isclose(number, truth, rtol=1e-6, atol=1e-9))
    except (TypeError, ValueError):
        return str(prepared).strip() == str(original).strip()

# true when a prepared value is missing (NaN, None or empty text)
def missing(value):
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""

# the share of a column's corrupted cells that are repaired in one candidate...
# prepared column (dates are compared by their year, month and day parts)
def column_accuracy(prepared, cells, column, candidate, mess):
    values = prepared[candidate]
    if mess == "dates":
        parts = [f"{column}_{part}" for part in ["year", "month", "day"]]
        if not set(parts) <= set(prepared.columns):
            return 0
        return sum(all(same(prepared.at[row, name], value) for name, value in zip(parts, pd.Timestamp(original).timetuple()[:3])) for row, original in zip(cells["position"], cells["original"]))
    if mess in MISSING_TYPES:
        return sum(missing(values.iat[row]) or same(values.iat[row], original) for row, original in zip(cells["position"], cells["original"]))
    return sum(same(values.iat[row], original) for row, original in zip(cells["position"], cells["original"]))

# repair accuracy of one prepared training table against the ground truth of...
# its cell: per corrupted column, the best matching prepared column (the column...
# itself, or a new column after renaming or splitting)
def score_prepared(prepared, folder, raw_columns, variant, steps):
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
    cells = pd.read_parquet(folder / "cells.parquet")
    position = {row: index for index, row in enumerate(truth["positions"]["train"])}
    cells = cells[cells["row"].isin(position)].assign(position=lambda part: part["row"].map(position))
    prepared = prepared.reset_index(drop=True)
    new_columns = [column for column in prepared.columns if column not in raw_columns]
    rows = []
    for (column, mess), part in cells.groupby(["column", "type"]):
        candidates = ([column] if column in prepared.columns else []) + new_columns
        best = max([column_accuracy(prepared, part, column, candidate, mess) for candidate in candidates] or [0])
        false_nan = np.nan
        if mess == "fake_missing" and column in prepared.columns:
            clean = np.setdiff1d(np.arange(len(prepared)), part["position"].to_numpy())
            false_nan = float(np.mean([missing(value) for value in prepared[column].iloc[clean]])) if len(clean) else np.nan
        rows.append({"dataset": truth["dataset"], "mess": mess, "variant": variant, "column": column, "cells": len(part), "repaired": int(best), "accuracy": best / len(part), "false_nan_rate": false_nan, "steps": steps})
    return rows

# the cleaning-only check of one single-type file: the rules variant runs up to...
# the gate (no model), the final plan's steps are fitted on the training rows...
# and their output is scored against the ground truth
def check_file(dataset, mess, variant="rules", budget=120):
    folder = cell_dir("A", dataset, 3, 0, mess)
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
    settings = load_settings(assignments=[f"budget.total_seconds={budget}", "spend.category=experiments"])
    run_folder = RunFolder(workdirs_dir / "repairs" / f"{dataset}-{mess}-{variant}")
    context = run(folder / "train.csv", settings, truth["target"], truth["task"], "benchmark", variant, folder=run_folder, console=False, options={"stop_after": "gate"})
    raw = read_table(folder / "train.csv")[0]
    features = raw.drop(columns=[truth["target"]])
    plan = context.final_plan
    prepared = AgentSteps(list(plan.steps) if plan is not None else []).fit(features).transform(features)
    return score_prepared(prepared, folder, list(features.columns), variant, len(plan.steps) if plan is not None else 0)

# command line: checks the chosen datasets and mess types and writes...
# benchmark/results/repairs.csv (rows of finished files are kept)
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="E6: repair accuracy of the cleaning stage only.")
    parser.add_argument("--datasets", nargs="*", default=GROUP_A)
    parser.add_argument("--types", nargs="*", default=REPAIR_TYPES)
    parser.add_argument("--variant", default="rules")
    arguments = parser.parse_args()
    output = results_dir / "repairs.csv"
    done = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=COLUMNS)
    finished = set(zip(done["dataset"], done["mess"], done["variant"]))
    for dataset in arguments.datasets:
        for mess in arguments.types:
            if (dataset, mess, arguments.variant) in finished or not (cell_dir("A", dataset, 3, 0, mess) / "cells.parquet").exists():
                continue
            try:
                rows = check_file(dataset, mess, arguments.variant)
            except Exception as error:
                rows = [{"dataset": dataset, "mess": mess, "variant": arguments.variant, "column": f"error: {type(error).__name__}: {error}"[:200], "cells": 0, "repaired": 0, "accuracy": np.nan, "false_nan_rate": np.nan, "steps": 0}]
            rows = rows or [{"dataset": dataset, "mess": mess, "variant": arguments.variant, "column": "", "cells": 0, "repaired": 0, "accuracy": np.nan, "false_nan_rate": np.nan, "steps": 0}]
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(output, index=False, lineterminator="\n")
            accuracy = sum(row["repaired"] for row in rows) / max(1, sum(row["cells"] for row in rows))
            print(f"{dataset} {mess}: accuracy {accuracy:.3f} ({sum(row['cells'] for row in rows)} cells)", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
