import json
import pandas as pd
from tools.glance import glance
from llms.client import LlmClient
from settings import load_settings
from llms.providers import FakeProvider
from schemas import Plan, Step, TaskSpec
from agents.cleaner import clean_plan, columns_view

TASK = TaskSpec(target="y", task="binary-classification", metric="roc_auc")
PROTECTED = Plan(steps=[Step(op="drop_columns", params={"columns": ["customer_id"]}, protected=True, source="leakguard")])

# a small messy table: an id, a weight with two units and a yes/no target
def table():
    return pd.DataFrame({
        "customer_id": [f"C{index}" for index in range(40)],
        "Weight": ["1.5kg", "800 g", "2 kg", "950 g"] * 10,
        "City": ["NY", "new york", "Paris", "paris"] * 10,
        "y": ["Yes", "No"] * 20,
    })

# an operation in the cleaner's answer format
def operation(op, params, reason="because"):
    return {"op": op, "params_json": json.dumps(params), "issues": [], "reason": reason}

# runs the cleaner with fake answers and returns the plan and the provider
def run_cleaner(tmp_path, answers, show_values=True):
    settings = load_settings()
    provider = FakeProvider(answers, model="gpt-4.1-nano")
    client = LlmClient(settings, None, provider, tmp_path / "spend.json", tmp_path / "cache", retry_wait=0)
    raw = table()
    plan = clean_plan(client, None, raw, glance(raw), TASK, PROTECTED, PROTECTED, "issues", settings, show_values)
    return plan, provider

UNITS = operation("parse_units", {"column": "Weight", "units": [{"unit": "kg", "factor": 1}, {"unit": "g", "factor": 0.001}]})

# a valid answer: protected steps come first, LLM steps are gated and marked llm
def test_valid_plan(tmp_path):
    plan, provider = run_cleaner(tmp_path, [{"operations": [UNITS], "notes": "units"}])
    assert [step.op for step in plan.steps] == ["drop_columns", "parse_units"]
    assert plan.steps[0].protected and not plan.steps[1].protected and plan.steps[1].source == "llm"
    prompt = provider.calls[0]["messages"][0]["content"]
    assert "parse_units (column)" in prompt and '"1.5kg" (10)' in prompt and "Already decided by code" in prompt

# an invalid answer (unknown column, bad JSON) is sent back with the errors
def test_invalid_answer_retried(tmp_path):
    bad = {"operations": [operation("as_category", {"column": "Town"}), {"op": "parse_number", "params_json": "{column: Weight", "issues": [], "reason": "x"}], "notes": ""}
    plan, provider = run_cleaner(tmp_path, [bad, {"operations": [UNITS], "notes": ""}])
    feedback = provider.calls[1]["messages"][-1]["content"]
    assert "unknown column" in feedback and "not valid JSON" in feedback
    assert [step.op for step in plan.steps] == ["drop_columns", "parse_units"]

# after the retries the cleaner gives None (the caller uses the translator plan)
def test_cleaner_gives_up(tmp_path):
    plan, provider = run_cleaner(tmp_path, [{"operations": [operation("magic", {})], "notes": ""}])
    assert plan is None and len(provider.calls) == 3

# a repeated protected step is not added twice; merging target classes is refused
def test_protected_and_target_rules(tmp_path):
    repeat = operation("drop_columns", {"columns": ["customer_id"]})
    plan, provider = run_cleaner(tmp_path, [{"operations": [repeat, UNITS], "notes": ""}])
    assert [step.op for step in plan.steps] == ["drop_columns", "parse_units"]
    merge = operation("map_target_labels", {"items": [{"value": "No", "canonical": "Yes"}]})
    plan, provider = run_cleaner(tmp_path / "b", [{"operations": [merge], "notes": ""}, {"operations": [UNITS], "notes": ""}])
    assert "never merged" in provider.calls[1]["messages"][-1]["content"] and plan.target_steps == []

# without samples the columns view shows names, types and counts only
def test_columns_view_without_values():
    raw = table()
    assert "values:" not in columns_view(raw, glance(raw), "y", show_values=False)
    assert '"NY" (10)' in columns_view(raw, glance(raw), "y")
