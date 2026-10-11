import json
from pathlib import Path
from pydantic import BaseModel
from schemas import Plan, Step
from operations.compiler import validate_plan
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
