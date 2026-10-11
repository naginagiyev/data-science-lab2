import pytest
import orchestrator
from schemas import RunContext
from settings import load_settings
from tools.runfolder import RunFolder
from orchestrator import Budget, RunState, StopRun, run, run_stage

# a run state with a fresh folder and context, for stage tests
def make_state(tmp_path):
    settings = load_settings()
    folder = RunFolder(tmp_path / "run")
    context = RunContext(run_id="r", mode="demo", variant="generic", data_path="x.csv", settings_hash="h")
    return RunState(settings, folder, context, Budget(300, {"audit": 90}, 60), None)

# a failing stage is logged with its traceback and its fallback is applied
def test_run_stage_fallback(tmp_path):
    state = make_state(tmp_path)
    def broken(state):
        raise ValueError("broken stage")
    status = run_stage(state, "glance", broken, lambda state: "skip stats")
    assert not status.ok and status.fallback == "skip stats"
    assert "ValueError" in state.folder.read_lines("errors.jsonl")[0]["error"]
    saved = RunContext.model_validate_json(state.folder.file("context.json").read_text(encoding="utf-8"))
    assert saved.stages[-1].name == "glance"

# a failing fallback is also logged and the run goes on
def test_run_stage_fallback_fails(tmp_path):
    state = make_state(tmp_path)
    def broken(state):
        raise ValueError("first")
    def broken_fallback(state):
        raise KeyError("second")
    status = run_stage(state, "plan", broken, broken_fallback)
    assert status.fallback.startswith("fallback failed")
    assert len(state.folder.read_lines("errors.jsonl")) == 2

# a hard stop is not swallowed by run_stage
def test_stop_run_propagates(tmp_path):
    state = make_state(tmp_path)
    def stop(state):
        raise StopRun("no data rows")
    with pytest.raises(StopRun):
        run_stage(state, "read", stop)

# stage limits leave time for the model, which gets at least its minimum
def test_budget():
    budget = Budget(100, {"audit": 90, "plan": 10}, 60)
    assert budget.stage_limit("audit") <= 40
    assert budget.stage_limit("plan") == 10
    assert budget.model_seconds() >= 60
    assert Budget(10, {}, 60).stage_limit("audit") == 0

# a run creates its folder with settings, context and run.log
def test_run_creates_folder(tmp_path):
    context = run("data/ibmchurn.csv", load_settings(), "Churn", "binary-classification", variant="generic", out_dir=tmp_path, console=False)
    folder = next(tmp_path.iterdir())
    for name in ["settings.json", "context.json", "run.log"]:
        assert (folder / name).exists()
    assert len(context.stages) == len(orchestrator.STAGES)

# an unknown variant is an error
def test_unknown_variant(tmp_path):
    with pytest.raises(ValueError):
        run("x.csv", load_settings(), variant="magic", out_dir=tmp_path, console=False)
