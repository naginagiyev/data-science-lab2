from schemas import Step

STATISTICAL_CODES = {"G01", "G02", "G04"}

# collects the leak evidence of every column from the edatool report and the...
# framer: structural (ids, saved index), statistical (G01, G02, G04) and...
# semantic (G03 name review, or the agent confirming a statistical leak with a...
# drop fix); dismissed issues stay statistical evidence only
def collect_evidence(report, id_columns):
    evidence = {}
    def add(column, kind, code, status):
        evidence.setdefault(column, {"structural": [], "statistical": [], "semantic": []})[kind].append({"code": code, "status": status})
    for issue in report.issues:
        if issue.code in {"A02", "E03"} and issue.status != "dismissed":
            for column in issue.columns[:1]:
                add(column, "structural", issue.code, issue.status)
        elif issue.code in STATISTICAL_CODES:
            for column in issue.columns[:1]:
                add(column, "statistical", issue.code, issue.status)
                agent_confirmed = issue.status == "confirmed" and issue.needs_review and issue.fix is not None and issue.fix.action == "drop_column"
                if agent_confirmed:
                    add(column, "semantic", issue.code, "confirmed by the agent")
        elif issue.code == "G03" and issue.status != "dismissed":
            for column in issue.columns:
                add(column, "semantic", "G03", issue.status)
    for column in id_columns:
        add(column, "structural", "framer", "id column")
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
            reason = "leak removed: the name suggests information known only after the event" if drop_semantic_only else "the name suggests a leak; please check"
        elif statistical:
            action, reason = "warn", "suspiciously strong, please check"
        else:
            continue
        decisions.append({"column": column, "action": action, "reason": reason, "evidence": kinds})
    return decisions

# the leak guard: decisions per column and the protected drop steps for the...
# plan (the target is never dropped)
def leak_guard(report, target, id_columns, drop_semantic_only=True, enabled=True):
    decisions = [item for item in decide(collect_evidence(report, [column for column in id_columns if column != target]), drop_semantic_only, enabled) if item["column"] != target]
    drops = [item["column"] for item in decisions if item["action"] == "drop"]
    steps = []
    if drops:
        issues = sorted({issue.id for issue in report.issues if set(issue.columns[:1]) & set(drops) and issue.code in {"A02", "E03", "G01", "G02", "G03", "G04"}})
        steps.append(Step(op="drop_columns", params={"columns": drops}, issues=issues, reason="leak guard: ids, saved index and leaks", protected=True, source="leakguard"))
    return decisions, steps
