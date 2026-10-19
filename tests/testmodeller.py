import numpy as np
import pandas as pd
from schemas import TaskSpec
from modelling import engine
from tools.glance import glance
from llms.client import LlmClient
from settings import load_settings
from llms.providers import FakeProvider
from agents.modeller import choose_models
from testengine import make_prep, toy_table

TASK = TaskSpec(target="y", task="binary-classification", metric="roc_auc")

# a client with a fake provider and its own ledger and cache
def fake_client(tmp_path, answers):
    provider = FakeProvider(answers, model="gpt-4.1-nano")
    return LlmClient(load_settings(), None, provider, tmp_path / "spend.json", tmp_path / "cache", retry_wait=0), provider

# the modeller gets the facts of the table and its shares are normalized
def test_choose_models(tmp_path):
    train = toy_table("binary-classification")
    client, provider = fake_client(tmp_path, [{"families": [{"family": "lightgbm", "share": 3}, {"family": "linear", "share": 1}], "reason": "small table"}])
    choice = choose_models(client, None, glance(train), train, TASK, 300)
    assert choice == {"shares": {"lightgbm": 0.75, "linear": 0.25}, "reason": "small table"}
    prompt = provider.calls[0]["messages"][0]["content"]
    assert "600 training rows, 3 feature columns" in prompt and "2 classes" in prompt and "300 seconds" in prompt

# an answer without a boosting family is sent back; a failure keeps the defaults
def test_choose_models_checks(tmp_path):
    train = toy_table("binary-classification")
    client, provider = fake_client(tmp_path, [{"families": [{"family": "linear", "share": 1}], "reason": "x"}, {"families": [{"family": "catboost", "share": 1}], "reason": "y"}])
    assert choose_models(client, None, glance(train), train, TASK, 300)["shares"] == {"catboost": 1.0}
    assert "at least one of lightgbm and catboost" in provider.calls[1]["messages"][-1]["content"]
    client, _ = fake_client(tmp_path / "b", [ValueError("provider down")])
    assert choose_models(client, None, glance(train), train, TASK, 300) is None

# the engine searches only the given families and splits the time by the shares
def test_engine_shares():
    frame = toy_table("regression", 300)
    y = pd.to_numeric(frame["y"]).to_numpy()
    folds = [list(range(start, 300, 3)) for start in range(3)]
    settings = load_settings().engine.model_copy(update={"candidates": ["dummy", "linear", "lightgbm"], "optuna": False})
    _, summary, _ = engine.train(frame.drop(columns="y"), y, "regression", "rmse", folds, make_prep, 20, settings, 0, shares={"linear": 0.5, "lightgbm": 0.5})
    assert {"linear", "lightgbm"} <= set(summary["cv"]) and np.isfinite(summary["cv"]["lightgbm"])
