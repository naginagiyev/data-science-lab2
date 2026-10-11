import json
import time
import pandas as pd
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from schemas import Plan, Step
from checks.datachecks import check_steps
from checks.gate import gate_data, run_gate
from operations.compiler import validate_plan
from tools.metrics import lower_is_better, oriented
from edatool.core.patterns import pattern_stats, pattern_text
from operations.registry import OPERATIONS, catalog_text, operation_class

# one operation of the cleaner's answer; params_json is a JSON object as text...
# (strict structured outputs do not accept free dictionaries)
class CleanerOperation(BaseModel):
    op: str
    params_json: str
    issues: list[str]
    reason: str

# the cleaner's answer: the full list of operations (column, row and target...
# operations) and a short note
class CleanerAnswer(BaseModel):
    operations: list[CleanerOperation]
    notes: str

# the columns view for the prompt: type, missing share, value counts of up to...
# max_values values and pattern stats per column; without values (the...
# no-samples ablation) only names, types and counts
def columns_view(raw, glance, target, show_values=True, max_values=15):
    types = {entry["name"]: entry for entry in glance.get("columns", [])}
    lines = []
    for column in raw.columns:
        if column == target:
            continue
        entry = types.get(str(column), {})
        values = raw[column].astype(str)
        empty = (values.str.strip() == "").mean()
        line = f"- `{column}` {entry.get('type', '?')} | {values.nunique()} unique | empty {empty:.1%}"
        if show_values:
            counts = values.value_counts().head(max_values)
            line += " | values: " + ", ".join(f"{json.dumps(value, ensure_ascii=False)} ({count})" for value, count in counts.items())
            patterns = pattern_text(pattern_stats(values))
            if patterns:
                line += f" | patterns: {patterns}"
        lines.append(line)
    return "\n".join(lines)

# a plan step in the prompt form of the catalog (op, params_json, issues, reason)
def step_text(step):
    return json.dumps({"op": step.op, "params_json": json.dumps(step.params, ensure_ascii=False), "issues": step.issues, "reason": step.reason}, ensure_ascii=False)

# turns the cleaner's operations into plan steps; returns the plan and the...
# problems (bad JSON, unknown operations)
def to_plan(answer, version):
    plan = Plan(version=version, notes=f"cleaner: {answer.notes}"[:500])
    problems = []
    for position, item in enumerate(answer.operations):
        try:
            params = json.loads(item.params_json) if item.params_json.strip() else {}
        except json.JSONDecodeError as error:
            problems.append(f"operation {position + 1} ({item.op}): params_json is not valid JSON ({error.msg})")
            continue
        if not isinstance(params, dict):
            problems.append(f"operation {position + 1} ({item.op}): params_json must be a JSON object")
            continue
        if item.op not in OPERATIONS:
            problems.append(f"operation {position + 1}: unknown operation '{item.op}'; use one of the catalog")
            continue
        group = {"column": "steps", "row": "row_steps", "target": "target_steps"}[operation_class(item.op).kind]
        getattr(plan, group).append(Step(op=item.op, params=params, issues=item.issues, reason=item.reason, source="llm"))
    return plan, problems

# the cleaner plan with the protected steps first: steps the LLM repeats are...
# not added twice; every LLM step is gated (code, not the LLM, sets protected)
def with_protected(plan, protected):
    keys = {(step.op, json.dumps(step.params, sort_keys=True)) for step in protected.steps + protected.row_steps + protected.target_steps}
    merged = Plan(version=plan.version, notes=plan.notes)
    for group in ["steps", "row_steps", "target_steps"]:
        own = [step for step in getattr(plan, group) if (step.op, json.dumps(step.params, sort_keys=True)) not in keys]
        for step in own:
            step.protected = False
        setattr(merged, group, list(getattr(protected, group)) + own)
    return merged

# the checks of a cleaner answer: valid JSON and operations, valid parameters...
# and columns in order (after the protected steps), target untouched; also the...
# target operations may not merge classes
def answer_problems(answer, protected, columns, target, task, labels):
    plan, problems = to_plan(answer, 1)
    merged = with_protected(plan, protected)
    problems += validate_plan(merged, columns, target, task)
    for step in merged.target_steps:
        if step.op == "map_target_labels" and not problems:
            problems += operation_class(step.op)(dict(step.params)).check(labels)
    return problems

# asks the LLM for a full plan; returns the plan (protected steps first) or...
# None when the LLM gives no valid answer (the caller then uses the translator plan)
def clean_plan(client, context, raw, glance, task, protected, default_plan, issues_text, settings, show_values=True):
    target = task.target
    gated_default = [step for group in [default_plan.steps, default_plan.row_steps, default_plan.target_steps] for step in group if not step.protected]
    protected_lines = [step_text(step) for group in [protected.steps, protected.row_steps, protected.target_steps] for step in group]
    inputs = {
        "target": target,
        "task": f"{task.task}, target `{target}`, metric {task.metric}",
        "protected": "\n".join(protected_lines) or "nothing",
        "issues": issues_text or "no issues",
        "columns": columns_view(raw, glance, target, show_values, settings.cleaner.max_values_per_column),
        "default_plan": "\n".join(step_text(step) for step in gated_default) or "empty",
        "catalog": catalog_text(),
    }
    labels = raw[target].astype(str).tolist() if task.task != "regression" else []
    check = lambda answer: answer_problems(answer, protected, list(raw.columns), target, task.task, labels)
    answer = client.ask("cleaner", inputs, CleanerAnswer, "plan", context, check=check)
    if answer is None:
        return None
    plan, _ = to_plan(answer, 1)
    return with_protected(plan, protected)

# the protected part of a plan (leak guard drops and protected translator steps)
def protected_part(plan):
    return Plan(version=plan.version, steps=[step for step in plan.steps if step.protected], row_steps=[step for step in plan.row_steps if step.protected], target_steps=[step for step in plan.target_steps if step.protected])

# a compact issue list for the prompt when no issues.md exists
def issues_lines(report, limit=40):
    chosen = [issue for issue in report.issues if issue.priority in {"must_fix", "should_fix"} and issue.status not in {"dismissed", "accepted"}][:limit]
    return "\n".join(f"- {issue.id} [{issue.priority}] {issue.code} {issue.title} ({', '.join(issue.columns[:3])}): {json.dumps(issue.evidence, default=str)[:200]}" for issue in chosen)

# reads the issues.md of the audit folder (about 2,000 tokens), or builds a...
# compact list from the report
def issues_text(audit_dir, report):
    file = Path(audit_dir) / "issues.md" if audit_dir else None
    if file is not None and file.exists():
        return file.read_text(encoding="utf-8")[:9000]
    return issues_lines(report)

REPAIR_FAMILY = ["parse_number", "parse_units", "parse_datetime", "parse_boolean", "extract_pattern", "replace_values", "set_missing_where", "map_categories", "as_category", "drop_columns"]

# the repair answer: a corrected (or replacing) operation, or drop
class RepairAnswer(BaseModel):
    action: Literal["replace", "drop"]
    op: str
    params_json: str
    reason: str

# the step a repair answer stands for, or None for drop; problems in the...
# answer raise ValueError with a text for the LLM
def repaired_step(answer, step):
    if answer.action == "drop":
        return None
    params = json.loads(answer.params_json)
    if not isinstance(params, dict):
        raise ValueError("params_json must be a JSON object")
    return Step(op=answer.op, params=params, issues=step.issues, reason=answer.reason, protected=False, source="repair")

# the checks of a repair answer: valid JSON, a catalog operation and a plan that...
# still validates with the new step in place of the failing one
def repair_problems(answer, plan, position, step, columns, target, task):
    if answer.action == "drop":
        return []
    try:
        new_step = repaired_step(answer, step)
    except (json.JSONDecodeError, ValueError) as error:
        return [f"params_json is not a valid JSON object ({error})"]
    trial = plan.model_copy(deep=True)
    trial.steps[position] = new_step
    return validate_plan(trial, columns, target, task)

# asks the LLM to repair one failing step; returns (new step or None, reason)
def repair_step(client, context, plan, position, error, frame, target, task):
    step = plan.steps[position]
    inputs_used = [column for column in [step.params.get("column")] + list(step.params.get("columns", [])) if column in frame.columns]
    counts = frame[inputs_used[0]].astype(str).value_counts().head(15) if inputs_used else pd.Series(dtype=int)
    inputs = {
        "operation": json.dumps({"op": step.op, "params_json": json.dumps(step.params, ensure_ascii=False), "reason": step.reason}, ensure_ascii=False),
        "error": error,
        "examples": error.split("examples:", 1)[1].strip() if "examples:" in error else "none",
        "value_counts": ", ".join(f"{json.dumps(value, ensure_ascii=False)} ({count})" for value, count in counts.items()) or "none",
        "catalog_entries": catalog_text([step.op] + [name for name in REPAIR_FAMILY if name != step.op]),
    }
    check = lambda answer: repair_problems(answer, plan, position, step, list(frame.columns) + [target], target, task)
    answer = client.ask("repair", inputs, RepairAnswer, "compile", context, check=check)
    if answer is None:
        return None, "no valid repair answer"
    return repaired_step(answer, step), answer.reason

# the repair loop of the compile stage: failing steps are repaired by the LLM...
# (at most max_repairs attempts per step) or dropped at once (no client, or...
# repair off); returns the plan without failing steps, the last check...
# results and the decisions (one per attempt)
def repair_plan(plan, frame, values, folds, target, task, settings, client=None, context=None, repair=True):
    attempts = {}
    records = []
    plan = plan.model_copy(deep=True)
    while True:
        results, failing = check_steps(plan, frame, values, folds, target, settings.checks)
        if not failing:
            return plan, results, records
        for position in sorted(failing, reverse=True):
            step = plan.steps[position]
            error = "; ".join(results[position]["problems"])
            key = id(step)
            attempts[key] = attempts.get(key, 0) + 1
            if not repair or client is None or step.protected or attempts[key] > settings.cleaner.max_repairs:
                records.append({"subject": step.op, "accepted": False, "reason": f"dropped after the checks: {error}", "proposal": step.model_dump()})
                plan.steps.pop(position)
                continue
            new_step, reason = repair_step(client, context, plan, position, error, frame, target, task)
            if new_step is None:
                records.append({"subject": step.op, "accepted": False, "reason": f"repair {attempts[key]}: dropped ({reason}); error was: {error}", "proposal": step.model_dump()})
                plan.steps.pop(position)
                continue
            records.append({"subject": step.op, "accepted": True, "reason": f"repair {attempts[key]}: {reason}; error was: {error}", "proposal": new_step.model_dump()})
            attempts[id(new_step)] = attempts[key]
            plan.steps[position] = new_step

# true when the gate result leaves room for a refinement: a step was removed,...
# or the full plan was not accepted
def needs_refine(result):
    return bool(result["reduced"]["removed"]) or not result["accepted"]

# asks the LLM for a revised full plan from the gate evidence; returns the plan...
# (protected steps first) or None
def refine_plan(client, context, plan, result, task, protected, columns, labels):
    metric = result["metric"]
    direction = f"{metric}, higher is better" if not lower_is_better(metric) else f"{metric}, lower is better"
    effects = [f"{item['op']} {json.dumps(item['params'], ensure_ascii=False)[:120]}: " + (f"{item['effect']:+.5f}" if item["effect"] is not None else item.get("note", "not tested")) for item in result["effects"]]
    gated = [step for group in [plan.steps, plan.row_steps, plan.target_steps] for step in group if not step.protected]
    inputs = {
        "direction": direction,
        "baseline": f"{result['p0']['score']:.5f}",
        "full": ", ".join(f"{score:.5f}" for score in result["p1"]["folds"]) + f" (mean {result['p1']['score']:.5f})",
        "effects": "\n".join(effects) or "not measured",
        "rejected": ", ".join(result["reduced"]["removed"]) or ("all gated operations (the plan lost to P0)" if not result["accepted"] else "none"),
        "plan": "\n".join(step_text(step) for step in gated) or "empty",
        "catalog": catalog_text(),
    }
    check = lambda answer: answer_problems(answer, protected, columns, task.target, task.task, labels)
    answer = client.ask("refine", inputs, CleanerAnswer, "gate", context, check=check)
    if answer is None:
        return None
    revised, _ = to_plan(answer, plan.version + 1)
    for group in ["steps", "row_steps", "target_steps"]:
        for step in getattr(revised, group):
            step.source = "refine"
    return with_protected(revised, protected)

# the gate with the refine loop: the plan is gated; while steps are rejected...
# (or the plan loses to P0) the LLM may revise it (at most max_refines rounds);...
# each revision is checked (failing steps repaired or dropped) and gated; the...
# best gated plan wins; returns the final plan and every gate result
def gate_with_refine(plan, train, task, split, settings, seed, client=None, context=None, deadline=None, repair=True):
    columns = list(train.columns)
    labels = train[task.target].astype(str).tolist() if task.task != "regression" else []
    protected = protected_part(plan)
    final, result = run_gate(plan, train, task.target, task.task, task.metric, split, settings.gate, seed, columns, deadline)
    results = [result]
    best = (final, result["final_score"])
    current = plan
    for _ in range(settings.cleaner.max_refines if client is not None else 0):
        if not needs_refine(result) or (deadline is not None and time.monotonic() > deadline):
            break
        revised = refine_plan(client, context, current, result, task, protected, columns, labels)
        if revised is None:
            break
        sample, folds, _ = gate_data(train, task.target, task.task, split, settings.gate, seed)
        revised, _, _ = repair_plan(revised, sample.drop(columns=[task.target]), sample[task.target].to_numpy(), folds, task.target, task.task, settings, client, context, repair)
        candidate, result = run_gate(revised, train, task.target, task.task, task.metric, split, settings.gate, seed, columns, deadline)
        result["refined"] = True
        results.append(result)
        if oriented(task.metric, result["final_score"]) > oriented(task.metric, best[1]):
            best = (candidate, result["final_score"])
        current = revised
    return best[0], results
