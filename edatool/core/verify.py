import re
from edatool.schemas import Edge
from edatool.core.ledger import make_fingerprint

# statuses that still need work in the fix loop
open_statuses = {"open", "confirmed", "still_open"}

# maps processed column names back to the original ones (explicit map, same...
# name, or a prefix like zipcode_98178)
def lineage_map(columns, original_columns, explicit=None):
    explicit = explicit or {}
    originals = sorted((str(name) for name in original_columns), key=len, reverse=True)
    mapping = {}
    for column in columns:
        name = str(column)
        if name in explicit:
            mapping[name] = explicit[name]
        elif name in originals:
            mapping[name] = name
        else:
            mapping[name] = next((original for original in originals if name.startswith(original + "_")), name)
    return mapping

# reads the rule name from the detector text of an issue
def rule_of(issue):
    match = re.search(r"rule: (.*)$", issue.detector)
    return match.group(1) if match else ""

# rebuilds the fingerprint of an issue with the original column names, so it...
# can match the previous report
def lineage_fingerprint(issue, mapping):
    columns = sorted({mapping.get(column, column) for column in issue.columns})
    return make_fingerprint(issue.code, columns, rule_of(issue))

# finds the highest id number per prefix, so new issues get new ids
def next_ids(previous_issues):
    counters = {}
    for issue in previous_issues:
        prefix, _, number = issue.id.rpartition("-")
        if number.isdigit():
            counters[prefix] = max(counters.get(prefix, 0), int(number))
    return counters

# returns the first snapshot (step) in which an issue appears; detector runs...
# are cached
def first_step(code, fingerprint, snapshot_contexts, run_codes, cache):
    for name, context, mapping in snapshot_contexts:
        if (name, code) not in cache:
            cache[(name, code)] = run_codes(context, code)
        for issue in cache[(name, code)]:
            if lineage_fingerprint(issue, mapping) == fingerprint:
                return name
    return None

# returns the fixed issue of a step that most likely caused a new issue (same...
# columns first)
def parent_for(step, step_issues, issues_by_id, columns):
    fixed = [issues_by_id[issue_id] for issue_id in step_issues.get(step, []) if issue_id in issues_by_id]
    if not fixed:
        return None
    overlapping = [issue for issue in fixed if set(issue.columns) & set(columns)]
    return (overlapping or fixed)[0]

# builds the issues of the next report: updates the old statuses and adds new...
# issues with their step, parent and level
def build_next_issues(previous, current_issues, mapping, step_issues, snapshot_contexts, run_codes):
    issues = [issue.model_copy(deep=True) for issue in previous.issues]
    by_id = {issue.id: issue for issue in issues}
    previous_fingerprints = {issue.fingerprint for issue in issues}
    fixed_ids = {issue_id for ids in step_issues.values() for issue_id in ids}
    current = {lineage_fingerprint(issue, mapping): issue for issue in current_issues}
    for issue in issues:
        if issue.status in {"dismissed", "accepted", "resolved"}:
            continue
        if issue.id in fixed_ids:
            issue.status = "still_open" if issue.fingerprint in current else "resolved"
    counters = next_ids(issues)
    edges = [edge.model_copy() for edge in previous.edges]
    new_issues = []
    step_cache = {}
    for fingerprint, issue in current.items():
        if fingerprint in previous_fingerprints:
            continue
        step = first_step(issue.code, fingerprint, snapshot_contexts, run_codes, step_cache)
        original_columns = sorted({mapping.get(column, column) for column in issue.columns})
        parent = parent_for(step, step_issues, by_id, original_columns) if step else None
        prefix = issue.id.rpartition("-")[0]
        counters[prefix] = counters.get(prefix, 0) + 1
        new_issue = issue.model_copy(deep=True)
        new_issue.id = f"{prefix}-{counters[prefix]:03d}"
        new_issue.fingerprint = fingerprint
        new_issue.created_by_step = step
        new_issue.children = []
        new_issue.caused_by = None
        if parent is not None:
            new_issue.caused_by = parent.id
            new_issue.level = parent.level + 1
            parent.children.append(new_issue.id)
            edges.append(Edge(source=new_issue.id, target=parent.id, kind="caused_by", reason=f"appeared after {step}"))
        new_issues.append(new_issue)
    return issues + new_issues, edges, new_issues
