import time
import numpy as np
import pandas as pd
from edatool import verify
from operations.registry import build
from operations.base import is_missing
from operations.compiler import AgentSteps, step_name

IGNORED_VERIFY = {"P01", "P02"}
NUMERIC_OPS = {"parse_number", "parse_units", "parse_datetime", "parse_boolean", "date_difference", "add_missing_indicator", "clip_values", "log_transform", "text_features"}

# a short, structured error of one failing operation for the repair prompt,...
# for example: parse_units on Weight: 4.1% of values could not be parsed,...
# examples: "1.3 kgs", "?"
def failure_text(step, problem, stats=None):
    column = step.params.get("column") or step.params.get("columns") or step.params.get("name") or ""
    text = f"{step.op} on {column}: {problem}"
    examples = (stats or {}).get("examples") or {}
    if examples:
        text += ", examples: " + ", ".join(f'"{value}"' for value in list(examples)[:8])
    return text

# the problems of one operation on one fold: errors, parse failures, wrong...
# output types, infinite values, new all-missing columns and slow steps
def step_problems(operation, step, train, valid, settings):
    problems = []
    start = time.perf_counter()
    try:
        fitted = operation.fit(train).transform(train)
        stats = dict(operation.stats_)
        operation.transform(valid)
    except Exception as error:
        return [failure_text(step, f"{type(error).__name__}: {error}")], None, None
    seconds = time.perf_counter() - start
    present = stats.get("present", 0)
    if present and stats.get("failures", 0) / present > settings.max_parse_failure:
        share = stats["failures"] / present
        problems.append(failure_text(step, f"{share:.1%} of values could not be parsed", stats))
    for column in operation.outputs():
        if column not in fitted:
            continue
        values = fitted[column]
        if step.op in NUMERIC_OPS and not pd.api.types.is_numeric_dtype(values):
            problems.append(failure_text(step, f"output column {column} is not numeric"))
        if pd.api.types.is_numeric_dtype(values) and np.isinf(values.to_numpy(dtype=float)).any():
            problems.append(failure_text(step, f"output column {column} has infinite values"))
        inputs = [train[name] for name in operation.inputs() if name in train]
        if values.isna().all() and inputs and not all(is_missing(source).all() for source in inputs):
            problems.append(failure_text(step, f"output column {column} is completely empty"))
    if seconds > settings.op_seconds:
        problems.append(failure_text(step, f"took {seconds:.1f} s (limit {settings.op_seconds:.0f} s)"))
    return problems, fitted, stats

# runs the data checks of every column step on the folds of a sample: each...
# operation is fitted on the training part and applied to both parts; a...
# failing step is reported and left out for the later steps; y holds the raw...
# target values; returns (results per step, indexes of the failing steps)
def check_steps(plan, frame, y, folds, target, settings):
    results = {position: {"step": step_name(position, step), "op": step.op, "problems": [], "stats": []} for position, step in enumerate(plan.steps)}
    everything = np.arange(len(frame))
    for valid in folds:
        train_frame = frame.iloc[np.setdiff1d(everything, valid)]
        valid_frame = frame.iloc[valid]
        for position, step in enumerate(plan.steps):
            if results[position]["problems"]:
                continue
            operation = build(step)
            problems, fitted, stats = step_problems(operation, step, train_frame, valid_frame, settings)
            if problems:
                results[position]["problems"] = problems
                continue
            results[position]["stats"].append(stats)
            train_frame = fitted
            valid_frame = operation.transform(valid_frame)
        start_columns = max(1, frame.shape[1])
        if train_frame.shape[1] > settings.max_column_growth * start_columns:
            last = len(plan.steps) - 1
            if last >= 0:
                results[last]["problems"].append(f"the plan grows the table from {start_columns} to {train_frame.shape[1]} columns (limit x{settings.max_column_growth:g})")
        target_values = pd.Series(np.asarray(y), index=frame.index).loc[train_frame.index].astype(str)
        for column in train_frame.columns:
            same_name = str(column).strip().casefold() == str(target).strip().casefold()
            same_values = train_frame[column].astype(str).equals(target_values)
            if same_name or same_values:
                creator = next((position for position, step in enumerate(plan.steps) if column in build(step).outputs()), None)
                if creator is not None:
                    results[creator]["problems"].append(f"output column {column} equals or is named like the target")
    failing = [position for position, result in results.items() if result["problems"]]
    return list(results.values()), failing

# runs edatool verify on the output of the agent steps (before the typing and...
# generic stages): the steps are fitted on the given training rows, the target...
# is added back to every snapshot, and P01/P02 are ignored (expected before...
# the generic stage); returns still-open and new must-fix issues
def verify_plan(report, plan, frame, target_values, target, run_dir=None):
    if not plan.steps:
        return {"skipped": "no column steps", "still_open": [], "new": [], "resolved": 0}
    agent = AgentSteps(list(plan.steps)).fit(frame)
    target_series = pd.Series(np.asarray(target_values), index=frame.index)
    snapshots = [(name, snapshot.assign(**{target: target_series.loc[snapshot.index].to_numpy()})) for name, snapshot in agent.snapshots(frame)]
    step_issues = {step_name(position, step): list(step.issues) for position, step in enumerate(plan.steps)}
    checked = verify(report, snapshots, step_issues, run_dir=run_dir)
    relevant = [issue for issue in checked.issues if issue.priority == "must_fix" and issue.code not in IGNORED_VERIFY]
    still_open = [issue for issue in relevant if issue.status == "still_open"]
    new = [issue for issue in relevant if issue.created_by_step]
    resolved = sum(issue.status == "resolved" for issue in checked.issues)
    describe = lambda issue: {"id": issue.id, "code": issue.code, "title": issue.title, "columns": issue.columns, "step": issue.created_by_step}
    return {"still_open": [describe(issue) for issue in still_open], "new": [describe(issue) for issue in new], "resolved": resolved}
