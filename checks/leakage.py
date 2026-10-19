from schemas import Step

# collects the leak evidence of every column: structural (ids and a saved...
# index from the audit, framer ids), statistical (G01: one feature predicts...
# the target alone) and semantic (the leak review: known only after the event)
def collect_evidence(report, id_columns, after_event=()):
    evidence = {}
    def add(column, kind, code, note):
        evidence.setdefault(column, {"structural": [], "statistical": [], "semantic": []})[kind].append({"code": code, "note": note})
    for issue in report.issues:
        if issue.code in {"A02", "E03"}:
            add(issue.columns[0], "structural", issue.code, issue.title)
        elif issue.code == "G01":
            add(issue.columns[0], "statistical", issue.code, ", ".join(f"{key} {value}" for key, value in issue.evidence.items()))
    for column in id_columns:
        add(column, "structural", "framer", "id column")
    for column, reason in after_event:
        add(column, "semantic", "leakreview", reason)
    return evidence

# decides every column with evidence by the table of §3.6: structural -> drop;...
# semantic + statistical -> drop (leak removed); semantic only -> drop when...
# drop_semantic_only, else warn; statistical only -> keep with a warning; with...
# the guard off only the structural drops (ids, index) remain
def decide(evidence, drop_semantic_only=True, enabled=True):
    decisions = []
    for column, kinds in evidence.items():
        structural, statistical, semantic = (bool(kinds[name]) for name in ["structural", "statistical", "semantic"])
        if structural:
            action, reason = "drop", "id or saved index column (structural)"
        elif not enabled:
            action, reason = "keep", "leak guard is off"
        elif semantic and statistical:
            action, reason = "drop", "leak removed: known only after the event and predicts the target alone"
        elif semantic:
            action = "drop" if drop_semantic_only else "warn"
            reason = "leak removed: the column holds information known only after the event" if drop_semantic_only else "the column may be a leak; please check"
        elif statistical:
            action, reason = "warn", "suspiciously strong, please check"
        else:
            continue
        decisions.append({"column": column, "action": action, "reason": reason, "evidence": kinds})
    return decisions

# the leak guard: decisions per column and the protected drop steps for the...
# plan (the target is never dropped)
def leak_guard(report, target, id_columns, after_event=(), drop_semantic_only=True, enabled=True):
    evidence = collect_evidence(report, [column for column in id_columns if column != target], after_event)
    decisions = [item for item in decide(evidence, drop_semantic_only, enabled) if item["column"] != target]
    drops = [item["column"] for item in decisions if item["action"] == "drop"]
    steps = []
    if drops:
        issues = sorted({issue.id for issue in report.issues if issue.code in {"A02", "E03", "G01"} and issue.columns[0] in drops})
        steps.append(Step(op="drop_columns", params={"columns": drops}, issues=issues, reason="leak guard: ids, saved index and leaks", protected=True, source="leakguard"))
    return decisions, steps
