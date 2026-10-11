import json
import textwrap
import llms.client
import pandas as pd
from orchestrator import run
from tools.glance import glance
from llms.client import LlmClient
from settings import load_settings
from llms.providers import FakeProvider
from schemas import Plan, Step, TaskSpec
from testpipeline import FAST, churn_file
from agents.freecoder import free_code_plan
from modelling.fittedmodel import FittedModel

TASK = TaskSpec(target="Churn", task="binary-classification", metric="roc_auc")
CODE = textwrap.dedent("""    import pandas as pd

    class CleaningStep:
        def fit(self, frame):
            self.contracts = sorted(set(frame["Contract"]))
            return self

        def transform(self, frame):
            output = pd.DataFrame(index=frame.index)
            output["tenure"] = pd.to_numeric(frame["tenure"], errors="coerce")
            output["TotalCharges"] = pd.to_numeric(frame["TotalCharges"].str.strip(), errors="coerce")
            output["contract_code"] = frame["Contract"].map(lambda value: float(self.contracts.index(value)) if value in self.contracts else -1.0)
            return output
""")

# a client with a fake provider and its own ledger and cache
def fake_client(tmp_path, answers):
    provider = FakeProvider(answers, model="gpt-4.1-nano")
    return LlmClient(load_settings(), None, provider, tmp_path / "spend.json", tmp_path / "cache", retry_wait=0), provider

# bad code is sent back with the problem; the second answer becomes one gated...
# freecode step after the protected steps
def test_free_code_plan(tmp_path):
    train = pd.read_csv(churn_file(tmp_path), dtype=str, keep_default_na=False)
    protected = Plan(steps=[Step(op="drop_columns", params={"columns": ["customerID"]}, protected=True, source="leakguard")])
    client, provider = fake_client(tmp_path, [{"code": "import os\n" + CODE}, {"code": "```python\n" + CODE + "```"}])
    plan = free_code_plan(client, None, train, glance(train), TASK, protected, "no issues", load_settings())
    assert "'os' is not allowed" in provider.calls[1]["messages"][-1]["content"]
    assert [step.op for step in plan.steps] == ["drop_columns", "freecode"] and plan.steps[1].params["code"].startswith("import pandas")
    assert "CleaningStep" in provider.calls[0]["messages"][0]["content"]

# when no answer passes the checks the result is None
def test_free_code_gives_up(tmp_path):
    train = pd.read_csv(churn_file(tmp_path), dtype=str, keep_default_na=False)
    client, provider = fake_client(tmp_path, [{"code": "class CleaningStep:\n    def fit(self, frame):\n        pass\n    def transform(self, frame):\n        return frame.head(1)\n"}])
    assert free_code_plan(client, None, train, glance(train), TASK, Plan(), "", load_settings()) is None and len(provider.calls) == 3

# the freecode variant runs end to end with a fake provider; the saved model...
# predicts raw rows through the sandbox
def test_freecode_run(tmp_path, monkeypatch):
    monkeypatch.setattr(llms.client, "spend_file", tmp_path / "spend.json")
    monkeypatch.setattr(llms.client, "llm_cache_dir", tmp_path / "cache")
    path = churn_file(tmp_path)
    provider = FakeProvider([{"code": CODE}], model="gpt-4.1-nano")
    settings = load_settings(assignments=FAST + ["gate.max_rows=400", "audit.use_agent=false"])
    context = run(path, settings, "Churn", "binary-classification", variant="freecode", out_dir=tmp_path / "runs", provider=provider, console=False)
    folder = next((tmp_path / "runs").iterdir())
    decisions = [json.loads(line) for line in (folder / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(item["kind"] == "freecode" and item["accepted"] for item in decisions)
    assert all(stage.ok for stage in context.stages)
    model = FittedModel.load(folder / "model.joblib")
    raw = pd.read_csv(path, dtype=str, keep_default_na=False).head(5)
    assert set(model.predict(raw.drop(columns="Churn"))) <= {"Yes", "No"}
