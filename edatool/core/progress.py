import json
import logging

logger = logging.getLogger("edatool")
# column widths of the terminal output
label_width = 13
text_width = 56

# turns a value into text and cuts it at the limit
def short(value, limit=160):
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " ..."

# one-line summary of an issue for the log file
def issue_text(issue):
    parts = [f"{issue.id} [{issue.priority}] {issue.title}"]
    if issue.columns:
        shown = ", ".join(issue.columns[:4])
        parts.append(shown + (f" (+{len(issue.columns) - 4})" if len(issue.columns) > 4 else ""))
    if issue.rows is not None:
        parts.append(f"{issue.rows.count} rows")
    if issue.needs_review and issue.status == "open":
        parts.append("needs review")
    return " | ".join(parts)

# one-line summary of tool arguments for the log file
def arguments_text(arguments, skip=("calls",)):
    items = [f"{key}={short(value, 120)}" for key, value in arguments.items() if key not in skip and value not in (None, [], {}, "")]
    return ", ".join(items)

# joins counts like '5 should_fix · 3 consider'
def count_text(counts, separator=" · "):
    return separator.join(f"{value} {key}" for key, value in counts.items())

# adds an 's' to a word when the count is not 1
def plural(count, word):
    return f"{count} {word}" + ("" if count == 1 else "s")

# splits text parts into lines that fit the given width
def wrap(parts, width=text_width, separator=" · "):
    lines = []
    current = ""
    for part in parts:
        candidate = part if not current else current + separator + part
        if len(candidate) > width and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines

# formats a stage line: label, text and the time on the right
def stage_line(label, text, seconds=None):
    right = f"{seconds:6.1f} s" if seconds is not None else ""
    return f"  {label:<{label_width}} {text:<{text_width}} {right}".rstrip()

# formats an indented detail line under a stage
def detail_line(text, label="", right=""):
    return f"    {label:<{label_width - 2}} {text:<{text_width}} {right}".rstrip()

# prints the header with the command, file, target and task
def header(command, name, target, task):
    logger.info("")
    logger.info(f"  edatool {command} · {name} · target `{target}` · {task}")
    logger.info("  " + "─" * (label_width + text_width + 9))

# logs the load stage
def log_load(raw, seconds, details):
    logger.info(stage_line("Load", f"{len(raw):,} rows × {raw.shape[1]} columns", seconds))
    logger.debug(f"    read as raw text{details}")

# logs the column type counts; the full column lists go to the log file
def log_schema(schema, target, seconds):
    groups = {}
    for column in schema:
        groups.setdefault(column.type, []).append(column.name)
    counts = sorted(((len(names), semantic_type) for semantic_type, names in groups.items()), reverse=True)
    lines = wrap([f"{count} {semantic_type}" for count, semantic_type in counts])
    logger.info(stage_line("Types", lines[0], seconds))
    for line in lines[1:]:
        logger.info(detail_line(line))
    for semantic_type, names in groups.items():
        logger.debug(f"    {semantic_type}: {', '.join(names)}")
    low = [column.name for column in schema if column.confidence == "low"]
    if low:
        logger.info(detail_line(f"low confidence, the agent will check: {', '.join(low)}"))
    ids = [column.name for column in schema if column.role == "id"]
    logger.debug(f"    target: {target}" + (f" | id columns: {', '.join(ids)}" if ids else ""))

# logs the detector stage: issue count, priorities and failed checks
def log_detectors(context, coverage, seconds):
    issues = context.ledger.issue_list()
    priorities = {}
    for issue in issues:
        priorities[issue.priority] = priorities.get(issue.priority, 0) + 1
    detectors = len({issue.detector.split(",")[0] for issue in issues})
    logger.info(stage_line("Detectors", f"{plural(len(issues), 'issue')} from {detectors} of {len(coverage)} checks", seconds))
    ordered = {key: priorities[key] for key in ["must_fix", "should_fix", "consider"] if key in priorities}
    if ordered:
        logger.info(detail_line(count_text(ordered)))
    failed = [entry.detector for entry in coverage if entry.error]
    if failed:
        logger.info(detail_line(f"✗ failed checks: {', '.join(failed)}"))

# logs the linking stage; every link and contrast goes to the log file
def log_links(context, seconds):
    issues = context.ledger.issues
    kinds = {}
    for edge in context.ledger.edges:
        kinds[edge.kind] = kinds.get(edge.kind, 0) + 1
        logger.debug(f"    {edge.source} {edge.kind.replace('_', ' ')} {edge.target}: {edge.reason}")
    contrasts = 0
    for issue in issues.values():
        for item in issue.evidence.get("contrast", []):
            contrasts += 1
            logger.debug(
                f"    {issue.id} rows: {item['column']} = {item['value']} in {item['share_in_rows']:.0%} "
                f"of them vs {item['share_elsewhere']:.1%} elsewhere"
            )
    names = {"caused_by": "cause link", "same_rows": "same-row link", "group": "redundancy link"}
    links = " · ".join(plural(value, names[kind]) for kind, value in kinds.items()) or "no links"
    review = [issue.id for issue in issues.values() if issue.needs_review and issue.status == "open"]
    logger.info(stage_line("Linking", f"{links} · {plural(contrasts, 'row contrast')}", seconds))
    need = "needs" if len(review) == 1 else "need"
    logger.info(detail_line(f"{plural(len(review), 'issue')} {need} agent review" if review else "no issue needs agent review"))

# groups issues by title and lists their columns
def title_parts(issues):
    groups = {}
    for issue in issues:
        groups.setdefault(issue.title.lower(), []).extend(issue.columns)
    parts = []
    for title, columns in groups.items():
        unique = list(dict.fromkeys(columns))
        shown = ", ".join(unique[:3]) + (", …" if len(unique) > 3 else "")
        parts.append(f"{title} ({shown})" if unique else title)
    return parts

# logs one priority row of the result, with wrapped issue titles
def log_priority(label, issues):
    lines = wrap(title_parts(issues), text_width - 6) or [""]
    logger.info(detail_line(f"{len(issues):>3}   {lines[0]}".rstrip(), label))
    for line in lines[1:]:
        logger.info(detail_line(f"      {line}"))

# logs the result block: status, issues per priority and the report path
def log_summary(report, folder=None):
    kept = [issue for issue in report.issues if issue.status != "dismissed"]
    status = "✓ complete" if report.meta.complete else "✗ incomplete (see run.log)"
    logger.info("")
    logger.info(stage_line("Result", status, report.meta.run_seconds))
    for priority in ["must_fix", "should_fix", "consider"]:
        log_priority(priority, [issue for issue in kept if issue.priority == priority])
    dismissed = [issue for issue in report.issues if issue.status == "dismissed"]
    if dismissed:
        log_priority("dismissed", dismissed)
    if folder is not None:
        logger.info("")
        logger.info(detail_line(str(folder / "issues.md"), "report"), extra={"run_folder": folder})
