import json
import pytest
from benchmark import spend
from pydantic import BaseModel
from schemas import RunContext
from settings import load_settings
from tools.runfolder import RunFolder
from llms.client import LlmClient, render
from llms.providers import FakeProvider, UnsupportedParameter

# the answer schema of the test prompt
class Capital(BaseModel):
    city: str
    confidence: float

# a client with a fake provider and its own ledger, cache and prompt folders
def make_client(tmp_path, answers, assignments=None, ledger=None):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "capital.md").write_text("Capital of {country}? Answer as JSON like {\"city\": ...}.", encoding="utf-8")
    ledger_path = tmp_path / "spend.json"
    if ledger is not None:
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    settings = load_settings(assignments=assignments or [])
    provider = FakeProvider(answers, model="gpt-4.1-nano")
    folder = RunFolder(tmp_path / "run")
    client = LlmClient(settings, folder, provider, ledger_path, tmp_path / "cache", prompt_dir, retry_wait=0)
    return client, provider, folder

# placeholders are filled, other braces stay
def test_render():
    assert render("a {x} {\"k\": 1} {y}", {"x": 5}) == "a 5 {\"k\": 1} {y}"

# a valid answer is returned, logged, paid and counted
def test_valid_answer(tmp_path):
    client, provider, folder = make_client(tmp_path, [{"city": "Paris", "confidence": 0.9}])
    answer = client.ask("capital", {"country": "France"}, Capital, "test")
    assert answer == Capital(city="Paris", confidence=0.9)
    assert "Capital of France?" in provider.calls[0]["messages"][0]["content"]
    assert provider.calls[0]["options"] == {"temperature": 0.0, "seed": 0}
    record = folder.read_lines("llm.jsonl")[0]
    assert record["ok"] and not record["cached"] and record["prompt"] == "capital"
    expected = (record["input_tokens"] * 0.1 + record["output_tokens"] * 0.4) / 1_000_000
    assert record["cost_usd"] == pytest.approx(expected)
    assert spend.totals(tmp_path / "spend.json")["dev"] == pytest.approx(expected)
    assert client.usage["calls"] == 1

# a schema error is sent back to the LLM, and the second answer is used
def test_schema_retry(tmp_path):
    client, provider, folder = make_client(tmp_path, [{"city": "Paris"}, {"city": "Paris", "confidence": 1}])
    answer = client.ask("capital", {"country": "France"}, Capital, "test")
    assert answer.confidence == 1
    assert len(provider.calls) == 2
    assert "confidence" in provider.calls[1]["messages"][-1]["content"]
    records = folder.read_lines("llm.jsonl")
    assert [record["ok"] for record in records] == [False, True]

# after the schema retries the client gives up and returns None
def test_schema_retries_exhausted(tmp_path):
    client, provider, folder = make_client(tmp_path, ["not json"])
    assert client.ask("capital", {"country": "France"}, Capital, "test") is None
    assert len(provider.calls) == 3
    assert client.usage["failed_calls"] == 3

# errors of the caller's own check are sent back like schema errors
def test_check_callback(tmp_path):
    client, provider, folder = make_client(tmp_path, [{"city": "Lyon", "confidence": 1}, {"city": "Paris", "confidence": 1}])
    check = lambda answer: [] if answer.city == "Paris" else [f"{answer.city} is not the capital"]
    assert client.ask("capital", {"country": "France"}, Capital, "test", check=check).city == "Paris"
    assert "Lyon is not the capital" in provider.calls[1]["messages"][-1]["content"]

# a second identical call comes from the cache: no provider call, no cost
def test_cache_hit(tmp_path):
    client, provider, folder = make_client(tmp_path, [{"city": "Paris", "confidence": 1}])
    client.ask("capital", {"country": "France"}, Capital, "test")
    spent = spend.totals(tmp_path / "spend.json")["total"]
    second, second_provider, second_folder = make_client(tmp_path, [RuntimeError("must not be called")])
    assert second.ask("capital", {"country": "France"}, Capital, "test").city == "Paris"
    assert second_provider.calls == []
    assert second_folder.read_lines("llm.jsonl")[-1]["cached"] is True
    assert spend.totals(tmp_path / "spend.json")["total"] == spent
    assert second.usage["cached_calls"] == 1 and second.usage["calls"] == 0

# transient API errors are retried; a lasting error gives None and is logged
def test_api_retries(tmp_path):
    answers = [ConnectionError("down"), TimeoutError("slow"), {"city": "Paris", "confidence": 1}]
    client, provider, folder = make_client(tmp_path, answers)
    assert client.ask("capital", {"country": "France"}, Capital, "test").city == "Paris"
    assert len(provider.calls) == 3
    failing, failing_provider, failing_folder = make_client(tmp_path / "b", [ConnectionError("down")])
    assert failing.ask("capital", {"country": "Italy"}, Capital, "test") is None
    assert len(failing_provider.calls) == failing.llm.api_retries + 1
    assert "ConnectionError" in failing_folder.read_lines("llm.jsonl")[0]["error"]

# a rejected parameter is dropped once and the call is repeated without it
def test_unsupported_parameter(tmp_path):
    answers = [UnsupportedParameter("temperature", "temperature is not supported"), {"city": "Paris", "confidence": 1}]
    client, provider, folder = make_client(tmp_path, answers)
    assert client.ask("capital", {"country": "France"}, Capital, "test").city == "Paris"
    assert provider.calls[1]["options"] == {"seed": 0}
    assert "temperature" in client.dropped_options

# the run call budget stops calls before the provider is reached
def test_run_budget(tmp_path):
    client, provider, folder = make_client(tmp_path, [{"city": "Paris", "confidence": 1}], ["llm.max_calls_per_run=1"])
    client.ask("capital", {"country": "France"}, Capital, "test")
    assert client.ask("capital", {"country": "Spain"}, Capital, "test") is None
    assert len(provider.calls) == 1
    assert folder.read_lines("llm.jsonl")[-1]["error"].startswith("budget")

# the client refuses a call that would cross total_cap_usd
def test_total_cap_refusal(tmp_path):
    ledger = {"totals": {"dev": 0.0, "experiments": 4.9999}, "calls": []}
    client, provider, folder = make_client(tmp_path, [{"city": "Paris", "confidence": 1}], ledger=ledger)
    assert client.ask("capital", {"country": "France"}, Capital, "test") is None
    assert provider.calls == []
    assert "total cap" in folder.read_lines("llm.jsonl")[0]["error"]

# the client refuses a call that would cross the cap of its category
def test_category_cap_refusal(tmp_path):
    ledger = {"totals": {"dev": 1.4999, "experiments": 0.0}, "calls": []}
    client, provider, folder = make_client(tmp_path, [{"city": "Paris", "confidence": 1}], ledger=ledger)
    assert client.ask("capital", {"country": "France"}, Capital, "test") is None
    assert "dev cap" in folder.read_lines("llm.jsonl")[0]["error"]
    experiments, experiments_provider, _ = make_client(tmp_path, [{"city": "Paris", "confidence": 1}], ["spend.category=experiments"], ledger=ledger)
    assert experiments.ask("capital", {"country": "France"}, Capital, "test").city == "Paris"

# the run context gets the usage totals
def test_context_usage(tmp_path):
    context = RunContext(run_id="r", mode="demo", variant="full", data_path="x", settings_hash="h")
    client, provider, folder = make_client(tmp_path, [{"city": "Paris", "confidence": 1}])
    client.ask("capital", {"country": "France"}, Capital, "test", context=context)
    assert context.llm["calls"] == 1 and context.llm["cost_usd"] > 0
