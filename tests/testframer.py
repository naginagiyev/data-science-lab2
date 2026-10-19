import numpy as np
import pandas as pd
from agents import framer
from tools.glance import glance
from llms.client import LlmClient
from settings import load_settings
from llms.providers import FakeProvider
from agents.framer import frame_task, rank_candidates, rule_task

# a table with an id, a numeric score with 6 levels, a churn flag and amounts
def table(rows=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customer_id": [f"C{index:05d}" for index in range(rows)],
        "monthly_spend": [f"{value:.2f}" for value in rng.uniform(10, 90, rows)],
        "quality": [str(value) for value in rng.integers(3, 9, rows)],
        "churn": rng.choice(["Yes", "No"], rows),
    })

# a client with a fake provider and its own ledger and cache
def fake_client(tmp_path, answers):
    settings = load_settings()
    provider = FakeProvider(answers, model="gpt-4.1-nano")
    return LlmClient(settings, None, provider, tmp_path / "spend.json", tmp_path / "cache", retry_wait=0), provider

# the rule task: 2 values binary, text multi-class, decimals or many integers...
# regression, few integer levels ambiguous
def test_rule_task():
    assert rule_task(["a", "b", "a"])[0] == "binary-classification"
    assert rule_task(["red", "green", "blue"])[0] == "multi-class-classification"
    assert rule_task(["1.5", "2.25", "3.0"])[0] == "regression"
    assert rule_task([str(value) for value in range(50)])[0] == "regression"
    task, ambiguous, reason = rule_task(["3", "4", "5", "6", "7"])
    assert task == "multi-class-classification" and ambiguous

# candidates: name hints and the last column rank first, ids never appear
def test_candidates():
    ranked = rank_candidates(glance(table()), load_settings().framer.name_hints)
    names = [name for name, _, _ in ranked]
    assert names[0] == "churn" and "customer_id" not in names

# the LLM picks the target among the candidates; the metric comes from the task
def test_llm_frames(tmp_path):
    raw = table()
    answer = {"target": "quality", "task": "regression", "confidence": 0.8, "reason": "an ordered score", "id_columns": ["customer_id"], "time_column": None}
    client, provider = fake_client(tmp_path, [answer])
    spec = frame_task(raw, glance(raw), load_settings(), client=client, glance_text="profile")
    assert spec.target == "quality" and spec.task == "regression" and spec.metric == "rmse" and spec.source == "llm"
    assert "customer_id" in spec.id_columns
    assert "Rule candidates" in provider.calls[0]["messages"][0]["content"]

# an invalid answer is retried with the error; when it stays invalid the...
# rules win
def test_invalid_answer_falls_back(tmp_path):
    raw = table()
    bad = {"target": "made_up", "task": "regression", "confidence": 0.9, "reason": "x", "id_columns": [], "time_column": None}
    client, provider = fake_client(tmp_path, [bad])
    spec = frame_task(raw, glance(raw), load_settings(), client=client)
    assert len(provider.calls) == 3 and "not in the candidate list" in provider.calls[1]["messages"][-1]["content"]
    assert spec.source == "rules" and spec.target == "churn"

# flags win and no LLM call is made; a given target with a clear task needs...
# no call either
def test_flags_win(tmp_path):
    raw = table()
    client, provider = fake_client(tmp_path, [RuntimeError("must not be called")])
    spec = frame_task(raw, glance(raw), load_settings(), "churn", "binary-classification", client)
    assert spec.source == "flags" and provider.calls == []
    spec = frame_task(raw, glance(raw), load_settings(), "monthly_spend", None, client)
    assert spec.task == "regression" and provider.calls == []

# with --ask, a low confidence is confirmed by the user
def test_ask_low_confidence(tmp_path, monkeypatch):
    raw = table()
    answer = {"target": "quality", "task": "multi-class-classification", "confidence": 0.3, "reason": "unsure", "id_columns": [], "time_column": None}
    client, provider = fake_client(tmp_path, [answer])
    replies = iter(["quality", "regression"])
    monkeypatch.setattr(framer.questionary, "select", lambda *arguments, **options: type("Prompt", (), {"ask": lambda self: next(replies)})())
    spec = frame_task(raw, glance(raw), load_settings(), client=client, ask=True)
    assert spec.task == "regression" and spec.source == "flags" and spec.confidence == 1.0
