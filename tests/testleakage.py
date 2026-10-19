from edatool.schemas import Fix, Issue, Meta, Report
from checks.leakage import collect_evidence, decide, leak_guard

# one edatool issue for the decision-table tests
def issue(code, column, status="open", needs_review=True, fix="drop_column"):
    return Issue(id=f"{code}-{column}", fingerprint=f"{code}{column}", code=code, title=code, columns=[column], detector="test", priority="must_fix", needs_review=needs_review, status=status, fix=Fix(action=fix))

# a report with the given issues
def report_of(issues):
    meta = Meta(tool_version="t", settings_hash="h", dataset_hash="d", dataset="x", rows=10, columns=3, target="y", task="binary-classification", raw_input=True, created_at="now")
    return Report(meta=meta, issues=issues)

# the decision table of §3.6, column by column
def test_decision_table():
    report = report_of([
        issue("E03", "customer_id", "confirmed", False, "set_role"),
        issue("G01", "refund_issued", "confirmed"),
        issue("G03", "refund_issued", "open", False),
        issue("G03", "closing_note", "open", False),
        issue("G01", "strong_feature", "open"),
        issue("G02", "dismissed_one", "dismissed"),
    ])
    decisions = {item["column"]: item for item in decide(collect_evidence(report, ["row_key"]))}
    assert decisions["customer_id"]["action"] == "drop" and decisions["row_key"]["action"] == "drop"
    assert decisions["refund_issued"]["action"] == "drop" and decisions["refund_issued"]["reason"].startswith("leak removed")
    assert decisions["closing_note"]["action"] == "drop"
    assert decisions["strong_feature"]["action"] == "warn" and decisions["dismissed_one"]["action"] == "warn"
    softer = {item["column"]: item for item in decide(collect_evidence(report, []), drop_semantic_only=False)}
    assert softer["closing_note"]["action"] == "warn"

# the agent confirming a statistical leak with a drop fix counts as semantic...
# evidence; without the agent the same issue stays a warning
def test_agent_confirmation():
    confirmed = decide(collect_evidence(report_of([issue("G01", "final_grade", "confirmed")]), []))
    assert confirmed[0]["action"] == "drop"
    open_issue = decide(collect_evidence(report_of([issue("G01", "final_grade", "open")]), []))
    assert open_issue[0]["action"] == "warn"

# the guard gives one protected drop step, never drops the target, and with...
# the guard off only ids are dropped
def test_guard_steps():
    report = report_of([issue("E03", "id", "confirmed", False, "set_role"), issue("G01", "leak", "confirmed"), issue("G03", "leak", "open", False), issue("G01", "y", "confirmed")])
    decisions, steps = leak_guard(report, "y", [])
    assert steps[0].params["columns"] == ["id", "leak"] and steps[0].protected and steps[0].source == "leakguard"
    assert all(item["column"] != "y" for item in decisions)
    decisions, steps = leak_guard(report, "y", [], enabled=False)
    assert steps[0].params["columns"] == ["id"]
