import time
import logging
from pathlib import Path
from typing import get_args
from edatool.core.ledger import Ledger
from datetime import datetime, timezone
from edatool.core.enrichment import enrich
from edatool.core.writer import write_report
from edatool.schemas import Meta, Report, Task
from edatool.core.typeinference import infer_schema
from edatool.detectors.splitrisk import advise_split
from edatool.core.loader import is_raw_view, load_raw
from edatool.core.registry import Context, run_detectors
from importlib.metadata import PackageNotFoundError, version
from edatool.settings import load_catalog, load_settings, settings_hash
from edatool.core.verify import build_next_issues, lineage_fingerprint, lineage_map, open_statuses
from edatool.core.progress import detail_line, header, issue_text, log_detectors, log_links, log_load, log_schema, log_summary, plural, stage_line

logger = logging.getLogger("edatool")

# raises an error if the task is not one of the three supported task types
def check_task(task):
    if task not in get_args(Task):
        raise ValueError(f"task must be one of: {', '.join(get_args(Task))}.")

# accepts a file path or a DataFrame; a path is loaded as a raw text view
def prepare_input(data):
    if isinstance(data, (str, Path)):
        return load_raw(data)
    return data

# returns the installed edatool version, or a fixed version when the package...
# is only copied into a project
def tool_version():
    try:
        return version("edatool")
    except PackageNotFoundError:
        return "0.1.0"

# returns a short name of the input for the terminal header
def data_name(data):
    return Path(data).name if isinstance(data, (str, Path)) else "DataFrame"

# loads the data, infers the column types and creates the Context that every...
# detector receives
def build_context(data, target, task, settings, sample_rows, timings):
    check_task(task)
    start = time.perf_counter()
    raw = prepare_input(data)
    timings["load"] = round(time.perf_counter() - start, 3)
    info = raw.attrs.get("edatool", {})
    details = f" | delimiter {info['delimiter']!r}, encoding {info['encoding']}" if info.get("delimiter") else ""
    log_load(raw, timings["load"], details)
    if target not in raw.columns:
        raise ValueError(f"Target column '{target}' was not found in the data.")
    run_settings = load_settings(settings)
    if sample_rows is not None:
        run_settings["thresholds"]["sampling"]["sample_rows"]["value"] = sample_rows
    start = time.perf_counter()
    schema, typed, details = infer_schema(raw, target, run_settings)
    timings["types"] = round(time.perf_counter() - start, 3)
    log_schema(schema, target, timings["types"])
    ledger = Ledger(load_catalog(), len(raw))
    return Context(raw, typed, schema, details, target, task, run_settings, ledger)

# builds the meta section of the report: versions, hashes, data size, target...
# and task
def build_meta(context, timings):
    start = time.perf_counter()
    meta = Meta(
        tool_version=tool_version(),
        settings_hash=settings_hash(context.settings),
        dataset_hash=context.dataset_hash(),
        dataset=context.raw.attrs.get("edatool", {}).get("source", "dataframe"),
        rows=context.n_rows,
        columns=context.raw.shape[1],
        target=context.target,
        task=context.task,
        sample_rows=context.value("sampling", "sample_rows"),
        raw_input=is_raw_view(context.raw),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    timings["meta"] = round(time.perf_counter() - start, 3)
    return meta

# runs the structure and split-risk checks on the full data and returns the...
# recommended train/test split
def presplit(data, target, task, settings=None, sample_rows=None):
    timings = {}
    header("presplit", data_name(data), target, task)
    context = build_context(data, target, task, settings, sample_rows, timings)
    stage_start = time.perf_counter()
    run_detectors(context, "presplit")
    advice = advise_split(context, context.ledger.issue_list())
    key = f" on `{advice.key}`" if advice.key else ""
    logger.info(stage_line("Split risks", plural(len(context.ledger.issues), "issue"), time.perf_counter() - stage_start))
    for issue in context.ledger.issue_list():
        logger.info(detail_line(f"{issue.title.lower()}" + (f" ({', '.join(issue.columns)})" if issue.columns else "")))
    logger.info("")
    logger.info(stage_line("Advice", f"{advice.strategy} split{key}"))
    logger.info(detail_line(advice.reason))
    logger.info("")
    return advice

# main entry point: runs the detectors, links the issues, lets the agent...
# review the unclear ones and writes the report
def audit(
    data,
    target,
    task,
    use_agent=True,
    model=None,
    settings=None,
    sample_rows=None,
    run_dir=None,
):
    start = time.perf_counter()
    timings = {}
    header("audit", data_name(data), target, task)
    context = build_context(data, target, task, settings, sample_rows, timings)
    meta = build_meta(context, timings)

    stage_start = time.perf_counter()
    coverage, detector_timings, complete = run_detectors(context, "audit")
    timings["detectors"] = round(time.perf_counter() - stage_start, 3)
    log_detectors(context, coverage, timings["detectors"])
    stage_start = time.perf_counter()
    enrich(context)
    timings["enrichment"] = round(time.perf_counter() - stage_start, 3)
    log_links(context, timings["enrichment"])

    trace = None
    if use_agent:
        from edatool.agent.session import review
        stage_start = time.perf_counter()
        summary, trace = review(context, context.settings, model)
        timings["agent"] = round(time.perf_counter() - stage_start, 3)
        timings["agent_sessions"] = summary["sessions"]
        timings["agent_tokens"] = summary["tokens"]
        meta.llm_model = summary["model"]
        meta.llm_tokens = summary["tokens"]
        meta.agent_sessions = len(summary["sessions"])
        meta.smolagents_version = summary["smolagents_version"]
        meta.prompt_hash = summary["prompt_hash"]
        complete = complete and summary["complete"]

    meta.complete = complete
    report = Report(
        meta=meta,
        schema=context.schema,
        issues=context.ledger.issue_list(),
        edges=context.ledger.edges,
        coverage=coverage,
    )
    report.meta.run_seconds = round(time.perf_counter() - start, 3)
    timings["total"] = report.meta.run_seconds
    timings["per_detector"] = detector_timings
    folder = write_report(report, timings, run_dir, context.settings, trace) if run_dir is not None else None
    log_summary(report, folder)
    return report

# runs only the detectors of one issue code with a fresh ledger (verify uses...
# it to find the step that created an issue)
def run_codes(context, code):
    context.ledger = Ledger(load_catalog(), context.n_rows)
    run_detectors(context, "verify", codes={code}, log=False)
    return context.ledger.issue_list()

# checks the data after the fixing steps: marks old issues as resolved or...
# still open and attaches new issues to the step that created them
def verify(
    report,
    snapshots,
    step_issues,
    target=None,
    task=None,
    settings=None,
    run_dir=None,
    lineage=None,
    test=None,
    use_agent=False,
    model=None,
):
    start = time.perf_counter()
    if isinstance(report, (str, Path)):
        report = Report.model_validate_json(Path(report).read_text(encoding="utf-8"))
    target = target or report.meta.target
    task = task or report.meta.task
    step_issues = step_issues or {}
    baseline_types = {column.name: column.type for column in report.schema_}
    timings = {}
    header("verify", f"{len(snapshots)} snapshot(s)", target, task)
    contexts = []
    for name, frame in snapshots:
        context = build_context(frame, target, task, settings, None, {})
        context.baseline_types = baseline_types
        context.lineage = lineage_map(frame.columns, baseline_types, lineage)
        contexts.append((name, context, context.lineage))
    last = contexts[-1][1]
    last.test_frame = test
    meta = build_meta(last, timings)
    stage_start = time.perf_counter()
    coverage, detector_timings, complete = run_detectors(last, "verify")
    log_detectors(last, coverage, time.perf_counter() - stage_start)
    enrich(last)
    current_issues = last.ledger.issue_list()
    previous_fingerprints = {issue.fingerprint for issue in report.issues}
    trace = None
    if use_agent:
        from edatool.agent.session import review
        new_ids = {issue.id for issue in current_issues if lineage_fingerprint(issue, last.lineage) not in previous_fingerprints}
        last.ledger.issues = {key: value for key, value in last.ledger.issues.items() if key in new_ids}
        last.ledger.edges = [edge for edge in last.ledger.edges if edge.source in new_ids and edge.target in new_ids]
        summary, trace = review(last, last.settings, model)
        meta.llm_model = summary["model"]
        meta.smolagents_version = summary["smolagents_version"]
        meta.prompt_hash = summary["prompt_hash"]
        complete = complete and summary["complete"]
    issues, edges, new_issues = build_next_issues(report, current_issues, last.lineage, step_issues, contexts, run_codes)
    meta.complete = complete
    meta.verify_level = (report.meta.verify_level or 0) + 1
    meta.loop_finished = not any(
        issue.priority in {"must_fix", "should_fix"} and issue.status in open_statuses for issue in issues
    )
    next_report = Report(meta=meta, schema=last.schema, issues=issues, edges=edges, coverage=coverage)
    next_report.meta.run_seconds = round(time.perf_counter() - start, 3)
    timings["total"] = next_report.meta.run_seconds
    timings["per_detector"] = detector_timings
    timings["new_issues"] = [issue.id for issue in new_issues]
    statuses = [issue.status for issue in issues if issue.id in {item.id for item in report.issues}]
    logger.info(stage_line("Compare", f"{statuses.count('resolved')} resolved · {statuses.count('still_open')} still open · {len(new_issues)} new"))
    for issue in new_issues:
        logger.info(detail_line(f"new: {issue.title.lower()} ({', '.join(issue.columns[:3])}) after {issue.created_by_step}"))
    for issue in new_issues:
        logger.debug(f"    new: {issue_text(issue)} | after {issue.created_by_step} | caused by {issue.caused_by}")
    folder = write_report(next_report, timings, run_dir, last.settings, trace) if run_dir is not None else None
    log_summary(next_report, folder)
    return next_report
