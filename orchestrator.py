import time
import logging
from paths import runs_dir
from llms.client import LlmClient
from settings import settings_hash
from schemas import RunContext, StageStatus, StopRun
from tools.runfolder import RunFolder, close_logging, setup_logging

logger = logging.getLogger("automl")
STAGES = ["read", "glance", "frame", "split", "audit", "leakguard", "plan", "compile", "gate", "model", "test", "report"]
VARIANTS = [
    "generic", "rules", "full", "full-noreview", "full-nogate", "full-norepair", "full-nosamples",
    "full-noleakguard", "freecode", "full+ag", "rules+ag",
]

# the time budget of one run: a deadline for the whole run and a limit per...
# stage; the model stage gets everything that is left, at least min_model_seconds
class Budget:
    def __init__(self, total_seconds, stage_seconds, min_model_seconds):
        self.start = time.monotonic()
        self.total_seconds = total_seconds
        self.stage_seconds = stage_seconds
        self.min_model_seconds = min_model_seconds

    def elapsed(self):
        return time.monotonic() - self.start

    def remaining(self):
        return max(0.0, self.total_seconds - self.elapsed())

    def stage_limit(self, name):
        available = max(0.0, self.remaining() - self.min_model_seconds)
        return min(self.stage_seconds.get(name, available), available)

    def model_seconds(self):
        return max(self.remaining(), self.min_model_seconds)

# everything one run needs: settings, run folder, context, budget, LLM client...
# and the objects that stages hand to each other (frames, reports, models)
class RunState:
    def __init__(self, settings, folder, context, budget, client):
        self.settings = settings
        self.folder = folder
        self.context = context
        self.budget = budget
        self.client = client
        self.objects = {}

# runs one stage: times it, catches any exception (written to errors.jsonl...
# and run.log), applies the fallback, records a StageStatus and saves the context
def run_stage(state, name, function, fallback=None):
    start = time.perf_counter()
    status = StageStatus(name=name, ok=True, seconds=0.0)
    try:
        function(state)
    except StopRun:
        raise
    except Exception as error:
        state.folder.log_error(name, error)
        status.ok = False
        status.error = f"{type(error).__name__}: {error}"
        if fallback is not None:
            try:
                status.fallback = fallback(state) or "fallback"
            except Exception as second_error:
                state.folder.log_error(f"{name} fallback", second_error)
                status.fallback = f"fallback failed: {type(second_error).__name__}"
    status.seconds = round(time.perf_counter() - start, 3)
    state.context.stages.append(status)
    state.folder.save_context(state.context)
    note = "ok" if status.ok else f"failed -> {status.fallback or 'no fallback'}"
    logger.info(f"  {STAGES.index(name) + 1 if name in STAGES else 0:02d} {name:<10} {note:<30} {status.seconds:6.1f} s")
    return status

# a stage that is not built yet; it only logs that it was skipped
def stub_stage(state):
    logger.debug("stage not implemented yet")

# the stages in order with their function and fallback
def stage_table():
    return [(name, stub_stage, None) for name in STAGES]

# runs the whole pipeline on one file and returns the RunContext; demo and...
# benchmark use this same function
def run(data_path, settings, target=None, task=None, mode="demo", variant="full", out_dir=None, folder=None, provider=None, console=True, options=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose one of: {', '.join(VARIANTS)}.")
    folder = folder or RunFolder.create(data_path, out_dir or runs_dir)
    handlers = setup_logging(folder, console=console)
    try:
        run_hash = settings_hash(settings)
        folder.write_json("settings.json", {"hash": run_hash, "settings": settings.model_dump()})
        context = RunContext(run_id=folder.path.name, mode=mode, variant=variant, data_path=str(data_path), settings_hash=run_hash)
        budget = Budget(settings.budget.total_seconds, settings.budget.stage_seconds, settings.budget.min_model_seconds)
        state = RunState(settings, folder, context, budget, LlmClient(settings, folder, provider))
        state.objects["target"] = target
        state.objects["task"] = task
        state.objects["options"] = options or {}
        folder.save_context(context)
        logger.info(f"Run {context.run_id} | {variant} | {mode} | budget {settings.budget.total_seconds:.0f} s")
        for name, function, fallback in stage_table():
            run_stage(state, name, function, fallback)
        logger.info(f"Done in {budget.elapsed():.1f} s | run folder: {folder.path}")
        return context
    except StopRun as error:
        logger.error(f"Stopped: {error}")
        folder.append("errors.jsonl", {"stage": "stop", "error": str(error)})
        raise
    finally:
        close_logging(handlers)
