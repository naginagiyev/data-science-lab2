import pytest
import smolagents
from benchmark import spend
from orchestrator import run
from settings import load_settings
from tools.runfolder import RunFolder
from smolagents.models import ChatMessage
from smolagents.monitoring import TokenUsage
from testpipeline import FAST, RUN_FILES, churn_file
from llms.agentmodel import AgentBudgetError, AgentModel

# a fake answer of the OpenAI model with token usage
def fake_generate(self, messages, **options):
    return ChatMessage(role="assistant", content="ok", token_usage=TokenUsage(input_tokens=1000, output_tokens=100))

# every agent call is logged to llm.jsonl and written to the spend ledger
def test_calls_logged_and_paid(tmp_path, monkeypatch):
    monkeypatch.setattr(smolagents.OpenAIModel, "generate", fake_generate)
    folder = RunFolder(tmp_path / "run")
    model = AgentModel(load_settings(), folder, ledger_path=tmp_path / "spend.json")
    model.generate([{"role": "user", "content": "review this"}])
    record = folder.read_lines("llm.jsonl")[0]
    assert record["stage"] == "audit" and record["prompt"] == "edatool-agent" and record["ok"]
    assert record["cost_usd"] == pytest.approx((1000 * 0.1 + 100 * 0.4) / 1_000_000)
    assert spend.totals(tmp_path / "spend.json")["dev"] == pytest.approx(record["cost_usd"])
    assert model.usage["calls"] == 1

# a call above the run budget is refused before the provider is asked
def test_budget_refusal(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(smolagents.OpenAIModel, "generate", lambda self, messages, **options: calls.append(1))
    model = AgentModel(load_settings(), None, max_cost_usd=0.0, ledger_path=tmp_path / "spend.json")
    with pytest.raises(AgentBudgetError):
        model.generate([{"role": "user", "content": "review this"}])
    assert calls == []

# when the provider raises, the audit keeps its deterministic result and the...
# run still ends with a model and a report; the failed calls are logged
def test_provider_failure_falls_back(tmp_path, monkeypatch):
    def broken(self, messages, **options):
        raise ConnectionError("provider down")
    monkeypatch.setattr(smolagents.OpenAIModel, "generate", broken)
    monkeypatch.setattr("llms.agentmodel.spend_file", tmp_path / "spend.json")
    path = churn_file(tmp_path)
    context = run(path, load_settings(assignments=FAST + ["gate.max_rows=400", "audit.session_seconds=20", "budget.total_seconds=25"]), "Churn", "binary-classification", variant="full", out_dir=tmp_path / "runs", console=False)
    folder = next((tmp_path / "runs").iterdir())
    for name in RUN_FILES:
        assert (folder / name).exists(), name
    failed = [record for record in RunFolder(folder).read_lines("llm.jsonl") if record["prompt"] == "edatool-agent"]
    assert failed and not any(record["ok"] for record in failed)
    assert context.final_plan is not None and context.model["chosen"]
