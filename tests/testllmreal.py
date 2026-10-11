import pytest
from benchmark import spend
from pydantic import BaseModel
from settings import load_settings
from llms.client import LlmClient
from tools.runfolder import RunFolder

# the answer schema of the real-API check
class Capital(BaseModel):
    city: str
    confidence: float

# one real structured-output call with gpt-4.1-nano through the client; it...
# costs a fraction of a cent and is written to the real spend ledger
@pytest.mark.llm
def test_real_openai_call(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "capital.md").write_text("What is the capital of {country}? Give a confidence from 0 to 1.", encoding="utf-8")
    settings = load_settings(assignments=["llm.model=gpt-4.1-nano"])
    before = spend.totals()["total"]
    folder = RunFolder(tmp_path / "run")
    client = LlmClient(settings, folder, cache_dir=tmp_path / "cache", prompt_dir=prompt_dir)
    answer = client.ask("capital", {"country": "France"}, Capital, "llm-test")
    assert answer is not None and answer.city == "Paris"
    record = folder.read_lines("llm.jsonl")[0]
    assert record["ok"] and record["input_tokens"] > 0
    assert 0 < record["cost_usd"] < 0.002
    assert spend.totals()["total"] == pytest.approx(before + record["cost_usd"])
