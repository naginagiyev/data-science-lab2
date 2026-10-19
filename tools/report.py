import math
from collections import defaultdict

# formats a score for display (full precision stays in the JSON files)
def show(value, digits=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"

# the task and split section
def task_section(context, extras):
    task = context.task
    lines = ["## Task and split", ""]
    if task is not None:
        lines += [
            f"- Target `{task.target}`, task {task.task}, metric {task.metric} (fixed by the task).",
            f"- Framed by {task.source} (confidence {task.confidence:.2f}): {task.reason}",
        ]
        if task.id_columns:
            lines.append(f"- Id columns: {', '.join(f'`{name}`' for name in task.id_columns)}")
    read = extras.get("read", {})
    if read:
        details = ", ".join(f"{key} {read[key]!r}" for key in ["encoding", "delimiter"] if read.get(key))
        lines.append(f"- Read {read.get('rows', '?')} rows and {read.get('columns', '?')} columns" + (f" ({details})." if details else "."))
    for note in extras.get("row_notes", []):
        lines.append(f"- {note}")
    split = context.split
    if split is not None:
        key = f" on `{split.key}`" if split.key else ""
        rows = f"{len(split.test_rows)} locked test rows, " if split.test_rows else ""
        lines.append(f"- Split: {split.strategy}{key}; {rows}{len(split.folds)} CV folds. {split.reason}")
        lines += [f"- Warning: {warning}" for warning in split.warnings]
    return lines

# the audit, leak and plan sections (only what the run has)
def preparation_section(context, extras, decisions):
    lines = []
    issues = extras.get("top_issues", [])
    if issues:
        lines += ["", "## Audit (top issues)", ""] + [f"- {issue}" for issue in issues]
    if context.leaks:
        lines += ["", "## Leak decisions", ""]
        for leak in context.leaks:
            evidence = "; ".join(f"{kind}: {', '.join(item['code'] for item in items)}" for kind, items in leak.get("evidence", {}).items() if items)
            lines.append(f"- `{leak.get('column')}`: {leak.get('action')} ({evidence}) - {leak.get('reason', '')}")
    plan = context.final_plan
    gated = [decision for decision in decisions if decision.get("stage") == "gate" and decision.get("kind") == "step"]
    repairs = [decision for decision in decisions if decision.get("kind") == "repair"]
    if plan is not None and (plan.steps or plan.row_steps or plan.target_steps):
        lines += ["", "## Plan (final)", ""]
        for step in plan.steps + plan.row_steps + plan.target_steps:
            flag = "protected" if step.protected else "gated"
            lines.append(f"- `{step.op}` {step.params} ({flag}, from {step.source}): {step.reason}")
    if gated:
        lines += ["", "## Gate decisions", ""]
        for decision in gated:
            effect = decision.get("evidence", {}).get("effect")
            state = "kept" if decision.get("accepted") else "removed"
            lines.append(f"- {decision.get('subject')}: {state} (CV effect {show(effect, 5)}) - {decision.get('reason')}")
    if repairs:
        lines += ["", "## Repairs", ""] + [f"- {decision.get('subject')}: {decision.get('reason')}" for decision in repairs]
    return lines

# the scores and model section: CV and test scores next to the dummy model
def score_section(context):
    scores = context.scores
    lines = ["", "## Scores", ""]
    if not scores:
        return lines + ["No scores."]
    metric = scores.get("metric", "")
    lines += [f"| | {metric} |", "|---|---|"]
    cv = scores.get("cv", {})
    lines.append(f"| CV, chosen model ({' + '.join(context.model.get('chosen', []))}) | {show(cv.get('score'))} |")
    lines.append(f"| CV, dummy model | {show(cv.get('dummy'))} |")
    test = scores.get("test")
    if test:
        lines.append(f"| Test, chosen model | {show(test.get('test'))} |")
        lines.append(f"| Test, dummy model | {show(test.get('dummy_test'))} |")
        secondary = ", ".join(f"{name} {show(value)}" for name, value in test.get("test_scores", {}).items())
        lines += ["", f"Test metrics: {secondary}."]
        for name, note in test.get("notes", {}).items():
            lines.append(f"- {name} undefined: {note}")
        ranking = test.get("importance", {})
        if ranking.get("columns"):
            cut = "" if ranking["complete"] else "; stopped early, not every column was checked"
            lines += ["", f"Permutation importance on {ranking['rows']} test rows (how much the test {metric} gets worse when the column is shuffled{cut}):", ""]
            lines += [f"- `{name}`: {show(value, 5)}" for name, value in ranking["columns"]]
    families = cv.get("families", {})
    if families:
        lines += ["", "CV per model family: " + ", ".join(f"{name} {show(value)}" for name, value in families.items()) + "."]
    return lines

# the time and LLM cost per stage, and the warnings and fallbacks
def cost_section(context, llm_records):
    by_stage = defaultdict(lambda: [0, 0.0])
    for record in llm_records:
        by_stage[record.get("stage", "?")][0] += 1
        by_stage[record.get("stage", "?")][1] += record.get("cost_usd", 0.0)
    lines = ["", "## Time and LLM cost per stage", "", "| stage | ok | seconds | LLM calls | cost (USD) |", "|---|---|---|---|---|"]
    for stage in context.stages:
        calls, cost = by_stage.get(stage.name, (0, 0.0))
        lines.append(f"| {stage.name} | {'yes' if stage.ok else 'no'} | {stage.seconds:.1f} | {calls} | {cost:.4f} |")
    usage = context.llm or {}
    lines += ["", f"Total: {sum(stage.seconds for stage in context.stages):.1f} s, {int(usage.get('calls', 0))} paid LLM calls, {int(usage.get('cached_calls', 0))} cached, ${usage.get('cost_usd', 0.0):.4f}."]
    problems = [stage for stage in context.stages if not stage.ok]
    if problems:
        lines += ["", "## Warnings and fallbacks", ""]
        lines += [f"- {stage.name}: {stage.error} -> {stage.fallback or 'no fallback'}" for stage in problems]
    return lines

# writes report.md of a run from the context, the decisions, the LLM log and...
# a few extras (read report, row notes, top audit issues, summary paragraph)
def write_report(folder, context, extras):
    decisions = folder.read_lines("decisions.jsonl")
    llm_records = folder.read_lines("llm.jsonl")
    title = context.data_path.replace("\\", "/").split("/")[-1]
    lines = [f"# Run report: {title}", "", f"Run `{context.run_id}` · variant `{context.variant}` · {context.mode} mode · settings `{context.settings_hash}`", ""]
    if extras.get("summary"):
        lines += ["## Summary", "", extras["summary"], ""]
    lines += task_section(context, extras)
    lines += preparation_section(context, extras, decisions)
    lines += score_section(context)
    lines += cost_section(context, llm_records)
    text = "\n".join(lines).rstrip("\n") + "\n"
    folder.file("report.md").write_text(text, encoding="utf-8", newline="\n")
    return text
