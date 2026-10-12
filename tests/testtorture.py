import json
import pytest
from schemas import StopRun
from orchestrator import run
from settings import load_settings
from torture import CASES, MILLION
from modelling.fittedmodel import FittedModel

SETTINGS = ["budget.total_seconds=20", "budget.min_model_seconds=8", "gate.max_rows=300", "engine.candidates=[\"dummy\", \"lightgbm\"]"]
STOPS = [case for case in CASES if case[4] != "model"]
MODELS = [case for case in CASES if case[4] == "model"]

# runs the rules variant on one torture file: a model and a report, or the...
# clear stop message of the reader
def check_case(case, folder, assignments=SETTINGS):
    name, writer, target, task, expected = case
    path = writer(folder)
    settings = load_settings(assignments=assignments)
    if expected != "model":
        with pytest.raises(StopRun, match=expected):
            run(path, settings, target, task, variant="rules", out_dir=folder / "runs", console=False)
        return
    context = run(path, settings, target, task, variant="rules", out_dir=folder / "runs", console=False)
    run_folder = next((folder / "runs").iterdir())
    assert (run_folder / "report.md").exists() and (run_folder / "model.joblib").exists(), name
    scores = json.loads((run_folder / "scores.json").read_text(encoding="utf-8"))
    assert scores["test"] is not None, name
    model = FittedModel.load(run_folder / "model.joblib")
    assert len(model.predict(path)) > 0, name
    return context

# the hard stops end at once with the clear message of the reader
@pytest.mark.parametrize("case", STOPS, ids=[case[0] for case in STOPS])
def test_torture_stop(case, tmp_path):
    check_case(case, tmp_path)

# every other torture file ends with a model, a report and predictions from...
# the file itself
@pytest.mark.slow
@pytest.mark.parametrize("case", MODELS, ids=[case[0] for case in MODELS])
def test_torture_model(case, tmp_path):
    check_case(case, tmp_path)

# the 1,000,000-row file finishes with a model inside a 300 s budget
@pytest.mark.slow
def test_torture_million(tmp_path):
    check_case(MILLION, tmp_path, ["budget.total_seconds=300"])
