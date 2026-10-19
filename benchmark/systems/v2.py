import sys
import time
from pathlib import Path
from orchestrator import run
from settings import load_settings
from tools.runfolder import RunFolder
from modelling.fittedmodel import FittedModel
from benchmark.systems.common import fit_seconds, run_adapter

STARTED = time.perf_counter()

# runs a v2 variant in benchmark mode on the train file (same orchestrator as...
# the demo), keeps the run folder next to the predictions, and predicts every...
# test file with the saved model
def fit_predict(arguments):
    seconds = fit_seconds(arguments, STARTED)
    assignments = list(arguments.set) + [
        f"budget.total_seconds={seconds}", f"budget.min_model_seconds={min(60.0, seconds / 2)}", f"seed={arguments.seed}",
    ]
    settings = load_settings(assignments=assignments)
    folder = RunFolder(Path(arguments.out) / "run")
    context = run(arguments.train, settings, arguments.target, arguments.task, "benchmark", arguments.variant, folder=folder, console=False)
    model = FittedModel.load(folder.file("model.joblib"))
    predictions = [model.raw_values(path) for path in arguments.tests]
    usage = context.llm or {}
    meta = {
        "llm_calls": int(usage.get("calls", 0)), "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)), "cost_usd": float(usage.get("cost_usd", 0.0)),
        "run_dir": str(folder.path), "settings_hash": context.settings_hash, "variant": arguments.variant,
    }
    return {"predictions": predictions, "classes": model.classes or None, "meta": meta}

if __name__ == "__main__":
    extra = {"--variant": {"default": "full"}, "--set": {"action": "append", "default": []}}
    sys.exit(run_adapter(fit_predict, extra))
