from pydantic import BaseModel

# one column the LLM judges as known only after the target event
class LeakColumn(BaseModel):
    column: str
    reason: str

# the leak review answer: only the leaking columns (empty when none leaks)
class LeakAnswer(BaseModel):
    after_event: list[LeakColumn]

# one line per column for the prompt: name, type and up to 3 example values
def columns_text(glance, target):
    lines = []
    for entry in glance.get("columns", []):
        if entry["name"] == target:
            continue
        examples = ", ".join(str(value)[:25] for value in entry.get("examples", [])[:3])
        lines.append(f"- `{entry['name']}` {entry.get('type', '?')}" + (f" | e.g. {examples}" if examples else ""))
    return "\n".join(lines)

# the G01 evidence of the audit as lines for the prompt
def signals_text(report):
    lines = [f"- `{issue.columns[0]}`: " + ", ".join(f"{key} {value}" for key, value in issue.evidence.items()) for issue in report.issues if issue.code == "G01"]
    return "\n".join(lines) or "none"

# the problems of an answer: unknown columns, the target, empty reasons
def answer_problems(answer, columns, target):
    problems = []
    unknown = [item.column for item in answer.after_event if item.column not in columns]
    if unknown:
        problems.append(f"unknown columns {unknown}; use exact column names from the list")
    if any(item.column == target for item in answer.after_event):
        problems.append(f"`{target}` is the target; never list it")
    if any(not item.reason.strip() for item in answer.after_event):
        problems.append("give a short reason for every column")
    return problems

# asks the LLM which columns are known only after the target event (one call,...
# prompts/leakreview.md); returns [(column, reason)], or [] when the call fails
def review_leaks(client, context, glance, report, task):
    columns = [entry["name"] for entry in glance.get("columns", [])]
    inputs = {"target": task.target, "task": task.task, "columns": columns_text(glance, task.target), "signals": signals_text(report)}
    answer = client.ask("leakreview", inputs, LeakAnswer, "leakguard", context, check=lambda answer: answer_problems(answer, columns, task.target))
    return [(item.column, item.reason) for item in answer.after_event] if answer is not None else []
