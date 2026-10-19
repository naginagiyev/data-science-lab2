import json
import pandas as pd
from schemas import Plan, Step
from llms.client import LlmClient
from settings import load_settings
from agents.cleaner import repair_plan
from llms.providers import FakeProvider

WRONG = Step(op="parse_units", params={"column": "Weight", "units": [{"unit": "kg", "factor": 1}]}, source="llm")
FIXED = {"action": "replace", "op": "parse_units", "params_json": json.dumps({"column": "Weight", "units": [{"unit": "kg", "factor": 1}, {"unit": "g", "factor": 0.001}]}), "reason": "grams were missing"}

# a frame where the weight mixes kg and g, and simple folds
def data():
    frame = pd.DataFrame({"Weight": ["1.5kg", "800 g", "2 kg", "950 g", "1.2kg"] * 20, "City": ["a", "b"] * 50})
    return frame, ["x"] * 100, [list(range(50)), list(range(50, 100))]

# a client with fake repair answers
def fake_client(tmp_path, answers):
    provider = FakeProvider(answers, model="gpt-4.1-nano")
    return LlmClient(load_settings(), None, provider, tmp_path / "spend.json", tmp_path / "cache", retry_wait=0), provider

# a wrong unit map is repaired: the LLM sees the structured error and the value...
# counts, and the corrected step passes the checks
def test_wrong_unit_map_repaired(tmp_path):
    frame, values, folds = data()
    client, provider = fake_client(tmp_path, [FIXED])
    plan, results, records = repair_plan(Plan(steps=[WRONG]), frame, values, folds, "y", "binary-classification", load_settings(), client)
    assert plan.steps[0].params["units"][1] == {"unit": "g", "factor": 0.001} and plan.steps[0].source == "repair"
    prompt = provider.calls[0]["messages"][0]["content"]
    assert "parse_units on Weight: 40.0% of values could not be parsed" in prompt and '"800 g"' in prompt
    assert records[0]["accepted"] and results[0]["problems"] == []

# after max_repairs failed attempts the step is dropped; every attempt is...
# recorded (identical repair prompts come from the LLM cache)
def test_repairs_capped(tmp_path):
    frame, values, folds = data()
    same = {"action": "replace", "op": "parse_units", "params_json": json.dumps(WRONG.params), "reason": "try again"}
    client, provider = fake_client(tmp_path, [same])
    plan, results, records = repair_plan(Plan(steps=[WRONG]), frame, values, folds, "y", "binary-classification", load_settings(), client)
    assert plan.steps == [] and 1 <= len(provider.calls) <= 3
    assert [record["accepted"] for record in records] == [True, True, True, False]

# the LLM may drop the step; without repair the step is dropped at once
def test_drop_and_no_repair(tmp_path):
    frame, values, folds = data()
    client, provider = fake_client(tmp_path, [{"action": "drop", "op": "", "params_json": "{}", "reason": "not fixable"}])
    plan, _, records = repair_plan(Plan(steps=[WRONG]), frame, values, folds, "y", "binary-classification", load_settings(), client)
    assert plan.steps == [] and "not fixable" in records[0]["reason"]
    plan, _, records = repair_plan(Plan(steps=[WRONG]), frame, values, folds, "y", "binary-classification", load_settings(), client, repair=False)
    assert plan.steps == [] and len(provider.calls) == 1
