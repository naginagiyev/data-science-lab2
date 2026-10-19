import json
import time
import logging
from paths import audit_file
from schemas import AuditReport
from audit.context import Context
from tools.columntypes import infer_types
from audit import formats, structure, target, values

logger = logging.getLogger("automl.audit")
CONFIG = json.loads(audit_file.read_text(encoding="utf-8"))
CHECKS = structure.CHECKS + formats.CHECKS + values.CHECKS + target.CHECKS

# the audit of the training rows: types once, then every check in order; a...
# check that fails is logged and the others still run; returns an AuditReport
def run_audit(train, target_column, task, sample_rows=50000, n_jobs=4):
    start = time.perf_counter()
    types, typed, details = infer_types(train)
    config = {"catalog": CONFIG["catalog"], "thresholds": {code: dict(limits) for code, limits in CONFIG["thresholds"].items()}}
    config["thresholds"]["G01"]["n_jobs"] = n_jobs
    context = Context(train, typed, types, details, target_column, task, config, sample_rows)
    failed = {}
    for check in CHECKS:
        try:
            check(context)
        except Exception as error:
            failed[check.__name__] = f"{type(error).__name__}: {error}"
            logger.warning(f"audit check {check.__name__} failed: {failed[check.__name__]}")
    seconds = round(time.perf_counter() - start, 3)
    return AuditReport(issues=context.issues, types={str(name): kind for name, kind in types.items()}, failed=failed, seconds=seconds)

# formats one evidence value for the prompt
def short_value(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value)

# the issue list for prompts: must_fix and should_fix issues with their...
# evidence and default fix, at most max_chars characters
def issues_text(report, max_chars=9000):
    lines = []
    for issue in [issue for issue in report.issues if issue.priority != "consider"]:
        head = f"- [{issue.priority}] {issue.id} {', '.join(issue.columns) or 'all rows'}: {issue.title.lower()}"
        lines.append(head + (f" ({issue.rows} rows)" if issue.rows else ""))
        lines.append("  evidence: " + "; ".join(f"{key} {short_value(value)}" for key, value in issue.evidence.items()))
        lines.append(f"  fix: {issue.action} {short_value(issue.params)}")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "\n... (more issues cut)"
