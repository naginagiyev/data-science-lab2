import json
import numpy as np
import llms.client
import pandas as pd
from orchestrator import run
from llms.client import LlmClient
from settings import load_settings
from llms.providers import FakeProvider
from agents.leakreview import review_leaks
from schemas import AuditReport, Issue, TaskSpec

TASK = TaskSpec(target="churn", task="binary-classification", metric="roc_auc")
GLANCE = {"columns": [{"name": "tenure", "type": "numeric_discrete", "examples": ["1", "24"]}, {"name": "refund_issued", "type": "boolean", "examples": ["yes", "no"]}, {"name": "churn", "type": "boolean"}]}
REPORT = AuditReport(issues=[Issue(id="G01-1", code="G01", title="t", columns=["refund_issued"], evidence={"auc": 0.99}, priority="must_fix")])

# a client with a fake provider and its own ledger and cache
def fake_client(tmp_path, answers):
    provider = FakeProvider(answers, model="gpt-4.1-nano")
    return LlmClient(load_settings(), None, provider, tmp_path / "spend.json", tmp_path / "cache", retry_wait=0), provider

# the review returns the leaking columns; the prompt shows the columns and the...
# strong signals of the audit, never the target as a column line
def test_review(tmp_path):
    client, provider = fake_client(tmp_path, [{"after_event": [{"column": "refund_issued", "reason": "a refund follows the churn"}]}])
    assert review_leaks(client, None, GLANCE, REPORT, TASK) == [("refund_issued", "a refund follows the churn")]
    prompt = provider.calls[0]["messages"][0]["content"]
    assert "- `refund_issued`: auc 0.99" in prompt and "- `tenure` numeric_discrete | e.g. 1, 24" in prompt and "- `churn`" not in prompt

# an unknown column or the target is sent back; the corrected answer is used;...
# a provider failure gives no semantic evidence
def test_review_checks(tmp_path):
    bad = {"after_event": [{"column": "refund", "reason": "x"}, {"column": "churn", "reason": "y"}]}
    client, provider = fake_client(tmp_path, [bad, {"after_event": []}])
    assert review_leaks(client, None, GLANCE, REPORT, TASK) == []
    feedback = provider.calls[1]["messages"][-1]["content"]
    assert "unknown columns ['refund']" in feedback and "is the target" in feedback
    client, _ = fake_client(tmp_path / "b", [ValueError("provider down")])
    assert review_leaks(client, None, GLANCE, REPORT, TASK) == []

# in a full run the review turns the statistical G01 warning into a drop; the...
# other LLM calls fail and fall back to rules
def test_full_run_drops_leak(tmp_path, monkeypatch):
    monkeypatch.setattr(llms.client, "spend_file", tmp_path / "spend.json")
    monkeypatch.setattr(llms.client, "llm_cache_dir", tmp_path / "cache")
    rng = np.random.default_rng(0)
    churn = rng.choice(["yes", "no"], 300)
    frame = pd.DataFrame({"tenure": rng.integers(0, 72, 300), "charges": rng.uniform(20, 90, 300).round(2), "refund_issued": np.where(churn == "yes", "Y", "N"), "churn": churn})
    path = tmp_path / "churn.csv"
    frame.to_csv(path, index=False)
    answers = lambda prompt, number: {"after_event": [{"column": "refund_issued", "reason": "refunds follow a churn"}]} if "target leakage" in prompt else ValueError("provider down")
    settings = load_settings(assignments=["budget.total_seconds=20", "budget.min_model_seconds=5", "gate.max_rows=300", "engine.optuna=false", "engine.candidates=[\"dummy\", \"lightgbm\"]"])
    run(path, settings, "churn", "binary-classification", variant="full", out_dir=tmp_path / "runs", provider=FakeProvider(answers), console=False)
    folder = next((tmp_path / "runs").iterdir())
    leaks = {item["column"]: item for item in json.loads((folder / "leaks.json").read_text(encoding="utf-8"))}
    assert leaks["refund_issued"]["action"] == "drop" and leaks["refund_issued"]["evidence"]["statistical"]
    assert (folder / "model.joblib").exists()
