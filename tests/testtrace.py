import io
import json
import numpy as np
import llms.client
import pandas as pd
from orchestrator import run
from rich.console import Console
from settings import load_settings
from tools.runfolder import RunFolder
from llms.providers import FakeProvider
from tools.trace import Trace, VIEWS, replay

# a small churn table with a leaking column, written as CSV
def churn_file(tmp_path):
    rng = np.random.default_rng(0)
    churn = rng.choice(["yes", "no"], 300)
    frame = pd.DataFrame({"tenure": rng.integers(0, 72, 300), "charges": rng.uniform(20, 90, 300).round(2), "refund_issued": np.where(churn == "yes", "Y", "N"), "churn": churn})
    path = tmp_path / "churn.csv"
    frame.to_csv(path, index=False)
    return path

# every event is one JSON line; decisions, LLM calls and errors written to the...
# run folder reach the trace through the folder listener
def test_trace_events(tmp_path):
    folder = RunFolder(tmp_path / "run")
    trace = Trace(folder)
    folder.listeners.append(trace.listen)
    folder.append("decisions.jsonl", {"stage": "plan", "kind": "step", "subject": "drop_columns", "accepted": True, "reason": "id", "source": "rules"})
    folder.append("llm.jsonl", {"stage": "plan", "prompt": "cleaner", "ok": True, "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001, "answer": "{}"})
    folder.append("errors.jsonl", {"stage": "gate", "error": "ValueError: x"})
    trace.close({"outcome": "done"})
    events = [json.loads(line) for line in folder.file("trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == ["decision", "llm_call", "error", "run_end"]
    assert events[-1]["data"]["usage"]["calls"] == 1 and not folder.file("trace.html").exists()

# a full run with a fake LLM shows stage rules, an LLM call and decisions on...
# the screen, and writes trace.html (offline, every stage) and trace.txt; the...
# replay rewrites trace.html in every view
def test_live_view_and_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(llms.client, "spend_file", tmp_path / "spend.json")
    monkeypatch.setattr(llms.client, "llm_cache_dir", tmp_path / "cache")
    answers = lambda prompt, number: {"after_event": [{"column": "refund_issued", "reason": "refunds follow a churn"}]} if "target leakage" in prompt else ValueError("provider down")
    settings = load_settings(assignments=["budget.total_seconds=20", "budget.min_model_seconds=5", "gate.max_rows=300", "engine.optuna=false", "engine.candidates=[\"dummy\", \"lightgbm\"]"])
    screen = io.StringIO()
    console = Console(file=screen, width=120, color_system=None)
    run(churn_file(tmp_path), settings, "churn", "binary-classification", variant="full", out_dir=tmp_path / "runs", provider=FakeProvider(answers), view="detailed", screen=console)
    shown = screen.getvalue()
    assert "01  read" in shown and "12  report" in shown and "◆ LLM leakreview" in shown and "refund_issued" in shown and "run finished" in shown
    folder = next((tmp_path / "runs").iterdir())
    html = (folder / "trace.html").read_text(encoding="utf-8")
    assert all(f"id='stage-{name}'" in html for name in ["read", "audit", "leakguard", "plan", "gate", "model", "report"])
    assert "http://" not in html and "https://" not in html and (folder / "trace.txt").stat().st_size > 0
    (folder / "trace.html").unlink()
    for view in VIEWS:
        assert replay(folder, view, Console(file=io.StringIO(), width=100, color_system=None)) == folder / "trace.html"
    assert (folder / "trace.html").exists() and replay(tmp_path, None) is None
