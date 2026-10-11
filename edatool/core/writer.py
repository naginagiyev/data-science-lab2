import json
import time
from pathlib import Path

# creates the run folder named <time>-<dataset>
def create_run_folder(run_dir, dataset_name):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = Path(run_dir) / f"{stamp}-{dataset_name}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder

# formats one evidence value for issues.md
def format_value(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)

# builds the first line of an issue in issues.md
def issue_line(issue):
    parts = [f"[{issue.priority}] {issue.id}"]
    if issue.columns:
        parts.append(", ".join(issue.columns))
    parts.append(issue.title.lower())
    if issue.rows is not None:
        parts.append(f"{issue.rows.count} rows ({issue.rows.share:.2%})")
    if issue.needs_review and issue.status == "open":
        parts.append("needs review")
    if issue.status == "accepted":
        parts.append("accepted")
    if issue.source == "agent":
        parts.append("added by agent")
    return " · ".join(parts)

# turns row contrasts into one readable sentence
def contrast_text(contrast):
    items = [
        f"{item['column']} = {item['value']} in {item['share_in_rows']:.0%} of these rows "
        f"vs {item['share_elsewhere']:.2%} elsewhere"
        for item in contrast
    ]
    return "; ".join(items)

# builds all lines of one root issue: evidence, contrast, diagnosis, fix and...
# its children
def issue_block(issue, issues, same_rows, compact):
    lines = [issue_line(issue)]
    if not compact:
        evidence = {key: value for key, value in issue.evidence.items() if key != "contrast"}
        if evidence:
            lines.append("  evidence: " + "; ".join(f"{key} {format_value(value)}" for key, value in evidence.items()))
        if issue.evidence.get("contrast"):
            lines.append("  contrast: " + contrast_text(issue.evidence["contrast"]))
        if issue.diagnosis:
            lines.append(f"  diagnosis: {issue.diagnosis}")
        if issue.fix is not None and issue.status != "accepted":
            lines.append(f"  fix: {issue.fix.action} {format_value(issue.fix.params)}")
        if issue.unresolved:
            lines.append(f"  unresolved: {issue.unresolved}")
    for child_id in issue.children:
        child = issues[child_id]
        lines.append(f"  └─ {child.id} {', '.join(child.columns)} · {child.title.lower()} (caused by {issue.id})")
    for other_id in same_rows.get(issue.id, []):
        lines.append(f"  ~ same rows as {other_id}")
    return lines

# builds the compact schema at the end of issues.md (one line per type)
def schema_lines(report, compact):
    groups = {}
    for column in report.schema_:
        groups.setdefault(column.type, []).append(column.name)
    lines = []
    for semantic_type, names in groups.items():
        shown = names[:10] if compact and len(names) > 10 else names
        more = f", … (+{len(names) - len(shown)})" if len(shown) < len(names) else ""
        lines.append(f"- {semantic_type}: {', '.join(shown)}{more}")
    return lines

# builds issues.md; compact_level 1 or 2 hides details to make it shorter
def render_issues_md(report, compact_level=0):
    meta = report.meta
    issues = {issue.id: issue for issue in report.issues}
    same_rows = {}
    for edge in report.edges:
        if edge.kind == "same_rows":
            same_rows.setdefault(edge.source, []).append(edge.target)
    failed = [entry.detector for entry in report.coverage if entry.error]
    lines = [
        f"# EDA issues: {meta.dataset}",
        "",
        f"rows {meta.rows} · columns {meta.columns} · target `{meta.target}` · task {meta.task}",
        f"complete: {'yes' if meta.complete else 'no'} · detectors run: {len(report.coverage)}"
        + (f" · failed: {', '.join(failed)}" if failed else ""),
    ]
    roots = [issue for issue in report.issues if issue.caused_by is None and issue.status != "dismissed"]
    dismissed = [issue for issue in report.issues if issue.status == "dismissed"]
    for priority in ["must_fix", "should_fix"]:
        selected = [issue for issue in roots if issue.priority == priority]
        if not selected:
            continue
        lines.extend(["", f"## {priority} ({len(selected)})", ""])
        compact = compact_level >= (2 if priority == "must_fix" else 1)
        for issue in selected:
            lines.extend(issue_block(issue, issues, same_rows, compact))
    consider = [issue for issue in roots if issue.priority == "consider"]
    if consider:
        lines.extend(["", f"## consider ({len(consider)})", ""])
        groups = {}
        for issue in consider:
            groups.setdefault(issue.title, []).extend(issue.columns or [issue.id])
        for title, columns in groups.items():
            lines.append(f"- {title} in {len(columns)} columns: {', '.join(columns)}")
    if dismissed:
        lines.extend(["", f"## dismissed by review ({len(dismissed)})", ""])
        for issue in dismissed:
            lines.append(f"- {issue.id} {', '.join(issue.columns)} · {issue.title.lower()}: {issue.diagnosis}")
    if not report.issues:
        lines.extend(["", "No issues found."])
    lines.extend(["", "## Schema", ""])
    lines.extend(schema_lines(report, compact_level > 0))
    return "\n".join(lines) + "\n"

# renders issues.md and makes it more compact until it fits the token budget
def render_within_budget(report, max_tokens, chars_per_token):
    for compact_level in range(3):
        text = render_issues_md(report, compact_level)
        if len(text) / chars_per_token <= max_tokens:
            return text
    return text

# writes report.json, issues.md, timings.json and trace.jsonl into the run...
# folder
def write_report(report, timings, run_dir, settings, trace=None):
    folder = create_run_folder(run_dir, report.meta.dataset)
    report_json = report.model_dump_json(by_alias=True, indent=2)
    (folder / "report.json").write_text(report_json + "\n", encoding="utf-8")
    writer_settings = settings["thresholds"]["writer"]
    max_tokens = writer_settings["max_tokens"]["value"]
    chars_per_token = writer_settings["chars_per_token"]["value"]
    issues_md = render_within_budget(report, max_tokens, chars_per_token)
    (folder / "issues.md").write_text(issues_md, encoding="utf-8")
    (folder / "timings.json").write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
    if trace is not None:
        lines = [json.dumps(entry, default=str, ensure_ascii=False) + "\n" for entry in trace]
        (folder / "trace.jsonl").write_text("".join(lines), encoding="utf-8")
    return folder
