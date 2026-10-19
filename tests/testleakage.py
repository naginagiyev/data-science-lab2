from schemas import AuditReport, Issue
from checks.leakage import collect_evidence, decide, leak_guard

# one audit issue for the decision-table tests
def issue(code, column):
    return Issue(id=f"{code}-{column}", code=code, title=code, columns=[column], priority="must_fix")

# the decision table of §3.6, column by column
def test_decision_table():
    report = AuditReport(issues=[issue("E03", "customer_id"), issue("G01", "refund_issued"), issue("G01", "strong_feature")])
    after_event = [("refund_issued", "known after the refund"), ("closing_note", "written when the case closes")]
    decisions = {item["column"]: item for item in decide(collect_evidence(report, ["row_key"], after_event))}
    assert decisions["customer_id"]["action"] == "drop" and decisions["row_key"]["action"] == "drop"
    assert decisions["refund_issued"]["action"] == "drop" and decisions["refund_issued"]["reason"].startswith("leak removed")
    assert decisions["closing_note"]["action"] == "drop"
    assert decisions["strong_feature"]["action"] == "warn"
    softer = {item["column"]: item for item in decide(collect_evidence(report, [], after_event), drop_semantic_only=False)}
    assert softer["closing_note"]["action"] == "warn"

# without the leak review a statistical leak stays a warning
def test_statistical_only():
    report = AuditReport(issues=[issue("G01", "final_grade")])
    assert decide(collect_evidence(report, []))[0]["action"] == "warn"
    assert decide(collect_evidence(report, [], [("final_grade", "given at the end")]))[0]["action"] == "drop"

# the guard gives one protected drop step, never drops the target, and with...
# the guard off only ids are dropped
def test_guard_steps():
    report = AuditReport(issues=[issue("E03", "id"), issue("G01", "leak"), issue("G01", "y")])
    decisions, steps = leak_guard(report, "y", [], after_event=[("leak", "after the event"), ("y", "the target")])
    assert steps[0].params["columns"] == ["id", "leak"] and steps[0].protected and steps[0].source == "leakguard"
    assert steps[0].issues == ["E03-id", "G01-leak"] and all(item["column"] != "y" for item in decisions)
    decisions, steps = leak_guard(report, "y", [], after_event=[("leak", "after the event")], enabled=False)
    assert steps[0].params["columns"] == ["id"]
