from llms.client import LlmClient
from settings import load_settings
from llms.providers import FakeProvider
from agents.summarizer import SummaryAnswer, summarize, summary_problems

FACTS = "Test roc_auc 0.8248 (dummy 0.5000). The gate removed 2 steps. Read 5,000 rows."

# numbers that are not in the facts and long answers are problems; small...
# counts and numbers with thousands commas are fine
def test_summary_check():
    assert summary_problems(SummaryAnswer(text="The model reached 0.8248 on 5000 rows; 3 steps ran."), FACTS) == []
    problems = summary_problems(SummaryAnswer(text="The model reached 0.91. " + "word " * 130), FACTS)
    assert any("0.91" in problem for problem in problems) and any("words" in problem for problem in problems)

# a wrong number is sent back; the corrected answer is used
def test_summarize_retry(tmp_path):
    provider = FakeProvider([{"text": "Score 0.95."}, {"text": "Score 0.8248."}], model="gpt-4.1-nano")
    client = LlmClient(load_settings(), None, provider, tmp_path / "spend.json", tmp_path / "cache", retry_wait=0)
    assert summarize(client, None, FACTS) == "Score 0.8248."
    assert "0.95" in provider.calls[1]["messages"][-1]["content"]

# a provider failure gives no summary (the caller keeps the template report)
def test_summarize_provider_failure(tmp_path):
    provider = FakeProvider([ValueError("provider down")], model="gpt-4.1-nano")
    client = LlmClient(load_settings(), None, provider, tmp_path / "spend.json", tmp_path / "cache", retry_wait=0)
    assert summarize(client, None, FACTS) is None
