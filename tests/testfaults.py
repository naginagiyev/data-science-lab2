import json
import pytest
import optuna
import llms.client
import orchestrator
from orchestrator import run
from modelling import engine
from settings import load_settings
from torture import units_and_packed
from llms.providers import FakeProvider
from operations.columns import ExtractPattern
from modelling.fittedmodel import FittedModel

SETTINGS = ["budget.total_seconds=20", "budget.min_model_seconds=8", "gate.max_rows=300", "engine.candidates=[\"dummy\", \"lightgbm\", \"catboost\"]"]

# raises the injected fault
def broken(*args, **kwargs):
    raise RuntimeError("injected fault")

# the run folder files of one finished run
def finished(folder):
    run_folder = next(folder.iterdir())
    assert (run_folder / "report.md").exists() and (run_folder / "model.joblib").exists()
    decisions = [json.loads(line) for line in (run_folder / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    return run_folder, decisions

# an operation, Optuna, one candidate model and verify all raise: the step is...
# dropped, LightGBM uses its default parameters, CatBoost is skipped, verify is...
# logged; the run still ends with a model that predicts
def test_code_faults(tmp_path, monkeypatch):
    monkeypatch.setattr(ExtractPattern, "apply", broken)
    monkeypatch.setattr(optuna.study.Study, "optimize", broken)
    original = engine.make_model
    monkeypatch.setattr(engine, "make_model", lambda family, *args, **kwargs: broken() if family == "catboost" else original(family, *args, **kwargs))
    monkeypatch.setattr(orchestrator, "verify_plan", broken)
    path = units_and_packed(tmp_path)
    context = run(path, load_settings(assignments=SETTINGS), "y", "binary-classification", variant="rules", out_dir=tmp_path / "runs", console=False)
    run_folder, decisions = finished(tmp_path / "runs")
    assert any(item["subject"] == "extract_pattern" and not item["accepted"] for item in decisions if item["stage"] == "compile")
    assert context.model["chosen"] == ["lightgbm"] and "catboost" not in context.model["cv"]
    checks = json.loads((run_folder / "checks.json").read_text(encoding="utf-8"))
    assert "injected fault" in checks["verify"]["error"]
    assert len(FittedModel.load(run_folder / "model.joblib").predict(path)) == 300

# every LLM call raises: the full variant falls back to the translator plan and...
# the template report (no summary) and still ends with a model
@pytest.mark.slow
def test_provider_faults(tmp_path, monkeypatch):
    monkeypatch.setattr(llms.client, "spend_file", tmp_path / "spend.json")
    monkeypatch.setattr(llms.client, "llm_cache_dir", tmp_path / "cache")
    path = units_and_packed(tmp_path)
    settings = load_settings(assignments=SETTINGS + ["audit.use_agent=false"])
    context = run(path, settings, "y", "binary-classification", variant="full", out_dir=tmp_path / "runs", provider=FakeProvider([ValueError("provider down")]), console=False)
    run_folder, decisions = finished(tmp_path / "runs")
    assert any(item["kind"] == "cleaner" and not item["accepted"] for item in decisions)
    assert "## Summary" not in (run_folder / "report.md").read_text(encoding="utf-8")
    assert all(stage.ok for stage in context.stages)
