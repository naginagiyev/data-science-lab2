import json
from testgate import class_table
from llms.client import LlmClient
from settings import load_settings
from llms.providers import FakeProvider
from agents.cleaner import gate_with_refine
from schemas import Plan, SplitInfo, Step, TaskSpec

TASK = TaskSpec(target="y", task="binary-classification", metric="roc_auc")
HARMFUL = Step(op="drop_columns", params={"columns": ["strong"]}, source="llm")

# a plan with a harmful step is revised: the LLM sees the gate evidence and its...
# revision without the harmful step wins
def test_harmful_plan_revised(tmp_path):
    revision = {"operations": [{"op": "drop_columns", "params_json": json.dumps({"columns": ["noise"]}), "issues": [], "reason": "noise only"}], "notes": "removed the drop of strong"}
    provider = FakeProvider([revision], model="gpt-4.1-nano")
    settings = load_settings(assignments=["gate.max_rows=600"])
    client = LlmClient(settings, None, provider, tmp_path / "spend.json", tmp_path / "cache", retry_wait=0)
    final, rounds = gate_with_refine(Plan(steps=[HARMFUL]), class_table(), TASK, SplitInfo(strategy="random"), settings, 0, client)
    prompt = provider.calls[0]["messages"][0]["content"]
    assert "Effect of each operation" in prompt and "drop_columns" in prompt and "roc_auc, higher is better" in prompt
    assert len(rounds) == 2 and rounds[1]["refined"]
    assert [step.params["columns"] for step in final.steps] == [["noise"]]
    assert all(step.source == "refine" for step in final.steps)

# without an LLM client there is no refinement
def test_no_refine_without_client():
    settings = load_settings(assignments=["gate.max_rows=600"])
    final, rounds = gate_with_refine(Plan(steps=[HARMFUL]), class_table(), TASK, SplitInfo(strategy="random"), settings, 0)
    assert len(rounds) == 1 and final.steps == []
