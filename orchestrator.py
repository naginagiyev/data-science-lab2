import time
import json
import logging
import numpy as np
import pandas as pd
from paths import runs_dir
from modelling import engine
from tools.glance import glance
from llms.client import LlmClient
from settings import settings_hash
from tools.report import write_report
from sklearn.pipeline import Pipeline
from modelling.cv import encode_target
from modelling.evaluator import evaluate
from agents.framer import frame_by_rules
from modelling.fittedmodel import FittedModel
from tools.splitter import holdout, make_folds, split
from operations.generic import GenericStage, TypingStep
from tools.runfolder import RunFolder, close_logging, setup_logging
from schemas import Decision, RunContext, SplitInfo, StageStatus, StopRun
from tools.reader import check_target, drop_duplicate_rows, drop_missing_target, read_table

logger = logging.getLogger("automl")
STAGES = ["read", "glance", "frame", "split", "audit", "leakguard", "plan", "compile", "gate", "model", "test", "report"]
BASE = {"audit": True, "review": True, "operations": True, "cleaner": True, "gate": True, "repair": True, "samples": True, "leakguard": True, "freecode": False, "engine": "own"}
VARIANT_OPTIONS = {
    "generic": dict(BASE, audit=False, review=False, operations=False, cleaner=False, gate=False, leakguard=False),
    "rules": dict(BASE, review=False, cleaner=False),
    "full": dict(BASE),
    "full-noreview": dict(BASE, review=False),
    "full-nogate": dict(BASE, gate=False),
    "full-norepair": dict(BASE, repair=False),
    "full-nosamples": dict(BASE, samples=False),
    "full-noleakguard": dict(BASE, leakguard=False),
    "freecode": dict(BASE, freecode=True),
    "full+ag": dict(BASE, engine="autogluon"),
    "rules+ag": dict(BASE, review=False, cleaner=False, engine="autogluon"),
}
VARIANTS = list(VARIANT_OPTIONS)

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

# everything one run needs: settings, run folder, context, budget, LLM client,...
# variant options and the objects that stages hand to each other
class RunState:
    def __init__(self, settings, folder, context, budget, client, options=None):
        self.settings = settings
        self.folder = folder
        self.context = context
        self.budget = budget
        self.client = client
        self.options = options or dict(BASE)
        self.objects = {"row_notes": []}

# writes one decision line to decisions.jsonl
def decide(state, stage, kind, subject, accepted, reason, source, proposal=None, evidence=None):
    decision = Decision(stage=stage, kind=kind, subject=subject, proposal=proposal or {}, evidence=evidence or {}, accepted=accepted, reason=reason, source=source)
    state.folder.append("decisions.jsonl", decision.model_dump())

# runs one stage: times it, catches any exception (written to errors.jsonl...
# and run.log), applies the fallback, records a StageStatus and saves the context
def run_stage(state, name, function, fallback=None):
    start = time.perf_counter()
    status = StageStatus(name=name, ok=True, seconds=0.0)
    note = None
    try:
        note = function(state)
    except StopRun:
        raise
    except Exception as error:
        state.folder.log_error(name, error)
        status.ok = False
        status.error = f"{type(error).__name__}: {error}"
        if fallback is not None:
            try:
                status.fallback = fallback(state) or "fallback"
            except StopRun:
                raise
            except Exception as second_error:
                state.folder.log_error(f"{name} fallback", second_error)
                status.fallback = f"fallback failed: {type(second_error).__name__}"
    status.seconds = round(time.perf_counter() - start, 3)
    state.context.stages.append(status)
    state.folder.save_context(state.context)
    text = (note or "ok") if status.ok else f"failed -> {status.fallback or 'no fallback'}"
    logger.info(f"  {STAGES.index(name) + 1 if name in STAGES else 0:02d} {name:<10} {text[:50]:<50} {status.seconds:6.1f} s")
    return status

# stage 01: reads the file as the raw text view (the only hard stops of the run)
def stage_read(state):
    raw, report = read_table(state.objects["data"])
    state.objects["raw"] = raw
    state.objects["read"] = report
    return f"{report['rows']} rows, {report['columns']} columns"

# stage 02: the quick look at every column
def stage_glance(state):
    result = glance(state.objects["raw"], state.settings.seed)
    state.objects["glance"] = result
    state.folder.write_json("glance.json", {"read": state.objects["read"], **result})
    return f"{len(result['columns'])} columns profiled"

# fallback of stage 02: an empty glance (the framer then uses the last column)
def fallback_glance(state):
    state.objects["glance"] = {"rows": len(state.objects["raw"]), "columns": []}
    return "empty glance"

# removes rows without a target (and, in demo mode, exact duplicate rows),...
# keeps regression targets that parse as numbers, and applies the hard stops
def clean_rows(state, task):
    raw, missing = drop_missing_target(state.objects["raw"], task.target) if task.target in state.objects["raw"] else (state.objects["raw"], 0)
    if missing:
        state.objects["row_notes"].append(f"{missing} rows without a target were removed.")
        decide(state, "frame", "rows", "missing target", True, f"{missing} rows without a target removed", "rules")
    if task.task == "regression" and task.target in raw:
        numbers = pd.to_numeric(raw[task.target].astype(str).str.strip(), errors="coerce")
        if numbers.isna().any():
            count = int(numbers.isna().sum())
            raw = raw[numbers.notna().to_numpy()].reset_index(drop=True)
            state.objects["row_notes"].append(f"{count} rows with a non-numeric regression target were removed.")
    if state.context.mode == "demo":
        raw, duplicates = drop_duplicate_rows(raw)
        if duplicates:
            state.objects["row_notes"].append(f"{duplicates} exact duplicate rows were removed before the split.")
            decide(state, "frame", "rows", "duplicate rows", True, f"{duplicates} exact duplicate rows removed before the split", "rules")
    check_target(raw, task.target, task.task, state.settings.split.min_rows)
    state.objects["raw"] = raw

# stage 03: target, task, metric and id columns (flags win; rules otherwise)
def stage_frame(state):
    task = frame_by_rules(state.objects["raw"], state.objects["glance"], state.settings, state.objects["target"], state.objects["task"])
    state.context.task = task
    state.folder.write_json("task.json", task.model_dump())
    decide(state, "frame", "task", task.target, True, task.reason, "user" if task.source == "flags" else "rules", proposal=task.model_dump())
    clean_rows(state, task)
    return f"{task.target} | {task.task} | {task.metric}"

# saves the split and hands the frames to the next stages
def finish_split(state, info, train, test):
    state.context.split = info
    state.objects["train"] = train
    state.objects["test"] = test
    state.folder.write_json("split.json", info.model_dump())
    decide(state, "split", "strategy", info.strategy, True, info.reason, "edatool", evidence={"key": info.key, "warnings": info.warnings})

# stage 04: locked test rows (demo mode) and CV folds on the training rows
def stage_split(state):
    task = state.context.task
    split_settings = state.settings.split
    info, train, test = split(state.objects["raw"], task.target, task.task, state.context.mode, split_settings.test_size, split_settings.cv_folds, state.settings.seed)
    finish_split(state, info, train, test)
    return f"{info.strategy}{' on ' + info.key if info.key else ''}, {len(train)} train rows"

# fallback of stage 04: a random hold-out and shuffled folds
def fallback_split(state):
    task = state.context.task
    raw = state.objects["raw"]
    if state.context.mode == "demo":
        train_rows, test_rows, _, _ = holdout(raw, task.target, "regression", "random", None, state.settings.split.test_size, state.settings.seed)
    else:
        train_rows, test_rows = np.arange(len(raw)), np.array([], dtype=int)
    train = raw.iloc[train_rows].reset_index(drop=True)
    folds, notes = make_folds(train, task.target, "regression", "random", None, state.settings.split.cv_folds, state.settings.seed)
    info = SplitInfo(strategy="random", test_rows=test_rows.tolist(), folds=folds, reason="split failed; random fallback", warnings=["the planned split failed; a random split was used"] + notes)
    finish_split(state, info, train, raw.iloc[test_rows].reset_index(drop=True))
    return "random split"

# a stage the variant does not use
def skipped(state):
    return "skipped (variant)"

# builds the preparation pipeline of a model kind: agent steps (P5) + typing...
# step + generic stage
def preparation(state, kind):
    return Pipeline([("typing", TypingStep()), ("generic", GenericStage(kind, state.settings.seed))])

# saves the Optuna trial log as optuna.csv
def save_trials(state, trials):
    if trials:
        frame = pd.DataFrame(trials)
        frame["params"] = frame["params"].map(json.dumps)
        frame.to_csv(state.folder.file("optuna.csv"), index=False, lineterminator="\n")

# trains with the engine, saves model.joblib and the CV results
def train_model(state, families=None):
    task = state.context.task
    train = state.objects["train"]
    y, encoder = encode_target(train[task.target], task.task)
    features = train.drop(columns=[task.target])
    settings = state.settings.engine if families is None else state.settings.engine.model_copy(update={"candidates": families, "optuna": False})
    n_classes = len(encoder.classes_) if encoder is not None else None
    make_prep = lambda kind: preparation(state, kind)
    trained, summary, trials = engine.train(features, y, task.task, task.metric, state.context.split.folds, make_prep, state.budget.model_seconds(), settings, state.settings.seed, n_classes)
    save_trials(state, trials)
    chosen = summary["chosen"]
    summary["cv_score"] = summary["ensemble_cv"] if len(chosen) == 2 else summary["cv"].get(chosen[0])
    state.context.model = summary
    fitted = FittedModel(trained, encoder, features.columns, task.target, task.task, task.metric, state.context.settings_hash, plan=state.context.final_plan)
    fitted.save(state.folder.file("model.joblib"))
    state.objects["fitted"] = fitted
    cv_text = f"{summary['cv_score']:.4f}" if summary["cv_score"] is not None else "n/a"
    return f"{' + '.join(chosen)} | CV {task.metric} {cv_text}"

# stage 10: the model search (own engine)
def stage_model(state):
    return train_model(state)

# fallback of stage 10: the dummy model only
def fallback_model(state):
    train_model(state, ["dummy"])
    return "dummy model"

# stage 11: the locked test set, used once (demo mode only)
def stage_test(state):
    model = state.context.model
    cv = {"score": model.get("cv_score"), "dummy": model.get("cv", {}).get("dummy"), "families": model.get("cv", {})}
    state.context.scores = {"metric": state.context.task.metric, "cv": cv, "test": None}
    if state.context.mode != "demo" or state.objects["test"].empty:
        return "no locked test set (benchmark mode)"
    result = evaluate(state.objects["fitted"], state.objects["test"], state.objects["train"][state.context.task.target])
    state.context.scores = {"metric": state.context.task.metric, "cv": cv, "test": result}
    return f"test {result['metric']} {result['test']:.4f} (dummy {result['dummy_test']:.4f})"

# stage 12: scores.json and report.md
def stage_report(state):
    scores = dict(state.context.scores)
    scores["seconds"] = {stage.name: stage.seconds for stage in state.context.stages}
    scores["llm"] = state.context.llm
    state.folder.write_json("scores.json", scores)
    extras = {"read": state.objects.get("read", {}), "row_notes": state.objects["row_notes"]}
    write_report(state.folder, state.context, extras)
    return "report.md written"

# the stages in order with their function and fallback for the variant
def stage_table(state):
    table = {
        "read": (stage_read, None), "glance": (stage_glance, fallback_glance), "frame": (stage_frame, None),
        "split": (stage_split, fallback_split), "audit": (skipped, None), "leakguard": (skipped, None), "plan": (skipped, None),
        "compile": (skipped, None), "gate": (skipped, None), "model": (stage_model, fallback_model), "test": (stage_test, None),
        "report": (stage_report, None),
    }
    return [(name, *table[name]) for name in STAGES]

# runs the whole pipeline on one file and returns the RunContext; demo and...
# benchmark use this same function
def run(data_path, settings, target=None, task=None, mode="demo", variant="full", out_dir=None, folder=None, provider=None, console=True, options=None):
    if variant not in VARIANT_OPTIONS:
        raise ValueError(f"Unknown variant '{variant}'. Choose one of: {', '.join(VARIANTS)}.")
    is_frame = isinstance(data_path, pd.DataFrame)
    folder = folder or RunFolder.create("dataframe" if is_frame else data_path, out_dir or runs_dir)
    handlers = setup_logging(folder, console=console)
    try:
        run_hash = settings_hash(settings)
        folder.write_json("settings.json", {"hash": run_hash, "settings": settings.model_dump()})
        context = RunContext(run_id=folder.path.name, mode=mode, variant=variant, data_path="dataframe" if is_frame else str(data_path), settings_hash=run_hash)
        budget = Budget(settings.budget.total_seconds, settings.budget.stage_seconds, settings.budget.min_model_seconds)
        state = RunState(settings, folder, context, budget, LlmClient(settings, folder, provider), VARIANT_OPTIONS[variant])
        state.objects.update(data=data_path, target=target, task=task, options=options or {})
        folder.save_context(context)
        logger.info(f"Run {context.run_id} | {variant} | {mode} | budget {settings.budget.total_seconds:.0f} s")
        for name, function, fallback in stage_table(state):
            run_stage(state, name, function, fallback)
        logger.info(f"Done in {budget.elapsed():.1f} s | run folder: {folder.path}")
        return context
    except StopRun as error:
        logger.error(f"Stopped: {error}")
        folder.append("errors.jsonl", {"stage": "stop", "error": str(error)})
        raise
    finally:
        close_logging(handlers)
