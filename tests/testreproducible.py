import json
import llms.client
from orchestrator import run
from settings import load_settings
from torture import units_and_packed
from llms.providers import FakeProvider

SETTINGS = ["budget.total_seconds=60", "budget.min_model_seconds=8", "gate.max_rows=300", "engine.optuna=false", "engine.candidates=[\"dummy\", \"lightgbm\"]", "audit.use_agent=false"]
PLAN = {"operations": [{"op": "extract_pattern", "params_json": json.dumps({"column": "size", "pattern": "([0-9.]+)x([0-9.]+)", "into": ["width", "height"], "types": ["number", "number"]}), "issues": [], "reason": "sizes"}], "notes": ""}

# the plans and the CV scores of one run folder
def outcome(folder):
    run_folder = next(folder.iterdir())
    plans = json.loads((run_folder / "plans.json").read_text(encoding="utf-8"))
    scores = json.loads((run_folder / "scores.json").read_text(encoding="utf-8"))
    return plans, scores["cv"], scores["test"]["test"]

# the same seed and a warm LLM cache give the same plans and the same scores;...
# the second run asks the provider only for the summary, whose facts hold the...
# run id and times (the search runs without the time-bound Optuna part, which...
# would make the number of trials depend on the machine load)
def test_same_seed_same_result(tmp_path, monkeypatch):
    monkeypatch.setattr(llms.client, "spend_file", tmp_path / "spend.json")
    monkeypatch.setattr(llms.client, "llm_cache_dir", tmp_path / "cache")
    path = units_and_packed(tmp_path)
    answer = lambda prompt, count: {"text": "The run finished."} if "data scientist" in prompt else PLAN
    first = FakeProvider(answer, model="gpt-4.1-nano")
    run(path, load_settings(assignments=SETTINGS), "y", "binary-classification", variant="full", out_dir=tmp_path / "a", provider=first, console=False)
    second = FakeProvider(answer, model="gpt-4.1-nano")
    run(path, load_settings(assignments=SETTINGS), "y", "binary-classification", variant="full", out_dir=tmp_path / "b", provider=second, console=False)
    asked = [call["messages"][0]["content"] for call in second.calls]
    assert first.calls and all("data scientist" in prompt for prompt in asked)
    assert outcome(tmp_path / "a") == outcome(tmp_path / "b")
