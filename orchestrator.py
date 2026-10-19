import time
import json
import logging
import numpy as np
import pandas as pd
from paths import runs_dir
from modelling import engine
from llms.client import LlmClient
from settings import settings_hash
from tools.metrics import oriented
from agents.framer import frame_task
from tools.report import write_report
from checks.leakage import leak_guard
from modelling.cv import encode_target
from agents.summarizer import summarize
from modelling.evaluator import evaluate
from agents.leakreview import review_leaks
from agents.freecoder import free_code_plan
from modelling.backends import train_backend
from tools.glance import glance, glance_text
from audit.core import issues_text, run_audit
from modelling.fittedmodel import FittedModel
from operations.translator import default_plan
from checks.gate import gate_data, protected_only
from tools.splitter import holdout, make_folds, split
from tools.runfolder import RunFolder, close_logging, setup_logging
from tools.reader import check_target, drop_duplicate_rows, drop_missing_target, read_table
from schemas import AuditReport, Decision, Plan, RunContext, SplitInfo, StageStatus, Step, StopRun
from operations.compiler import AgentSteps, feature_pipeline, row_filter, target_steps, validate_plan
from agents.cleaner import clean_plan, gate_with_refine, prepare_custom_steps, protected_part, repair_plan

logger = logging.getLogger("automl")
STAGES = ["read", "glance", "frame", "split", "audit", "leakguard", "plan", "compile", "gate", "model", "test", "report"]
BASE = {"audit": True, "leakreview": True, "operations": True, "cleaner": True, "gate": True, "repair": True, "samples": True, "leakguard": True, "freecode": False, "engine": "own"}
VARIANT_OPTIONS = {
    "generic": dict(BASE, audit=False, leakreview=False, operations=False, cleaner=False, gate=False, leakguard=False),
    "rules": dict(BASE, leakreview=False, cleaner=False),
    "full": dict(BASE),
    "full-noreview": dict(BASE, leakreview=False),
    "full-nogate": dict(BASE, gate=False),
    "full-norepair": dict(BASE, repair=False),
    "full-nosamples": dict(BASE, samples=False),
    "full-noleakguard": dict(BASE, leakguard=False),
    "freecode": dict(BASE, freecode=True, cleaner=False),
    "full+ag": dict(BASE, engine="autogluon"),
    "rules+ag": dict(BASE, leakreview=False, cleaner=False, engine="autogluon"),
}
VARIANTS = list(VARIANT_OPTIONS)
FINISH_SHARE = 0.04

# the time budget of one run: a deadline for the whole run and a limit per...
# stage; the model stage gets everything that is left except a small reserve...
# for the test and report stages, at least min_model_seconds
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
        return max(self.remaining() - FINISH_SHARE * self.total_seconds, self.min_model_seconds)

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
    duplicates = 0
    if state.context.mode == "demo":
        raw, duplicates = drop_duplicate_rows(raw)
        if duplicates:
            state.objects["row_notes"].append(f"{duplicates} exact duplicate rows were removed before the split.")
            decide(state, "frame", "rows", "duplicate rows", True, f"{duplicates} exact duplicate rows removed before the split", "rules")
    check_target(raw, task.target, task.task, state.settings.split.min_rows, duplicates)
    state.objects["raw"] = raw

# stage 03: target, task, metric and id columns (flags win; rules, and the...
# LLM in the LLM variants when a flag is missing or the rule task is ambiguous)
def stage_frame(state):
    uses_llm = state.options["cleaner"] or state.options["leakreview"] or state.options["freecode"]
    client = state.client if uses_llm else None
    profile = glance_text(state.objects["glance"], examples=state.options["samples"]) if uses_llm and state.objects["glance"]["columns"] else ""
    ask = state.objects["options"].get("ask", False)
    task = frame_task(state.objects["raw"], state.objects["glance"], state.settings, state.objects["target"], state.objects["task"], client, state.context, ask, profile)
    state.context.task = task
    state.folder.write_json("task.json", task.model_dump())
    decide(state, "frame", "task", task.target, True, task.reason, {"flags": "user", "rules": "rules", "llm": "llm"}[task.source], proposal=task.model_dump())
    clean_rows(state, task)
    return f"{task.target} | {task.task} | {task.metric}"

# saves the split and hands the frames to the next stages
def finish_split(state, info, train, test):
    state.context.split = info
    state.objects["train"] = train
    state.objects["test"] = test
    state.folder.write_json("split.json", info.model_dump())
    decide(state, "split", "strategy", info.strategy, True, info.reason, "rules", evidence={"key": info.key, "warnings": info.warnings})

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

# keeps the audit report, saves it as audit.json and counts the issues
def keep_report(state, report):
    state.objects["report"] = report
    state.folder.write_json("audit.json", report.model_dump())
    counts = {priority: sum(issue.priority == priority for issue in report.issues) for priority in ["must_fix", "should_fix"]}
    return f"{len(report.issues)} issues ({counts['must_fix']} must_fix, {counts['should_fix']} should_fix)"

# stage 05: the audit of the training rows (code only): types, then the checks...
# of structure, formats, values and the target
def stage_audit(state):
    task = state.context.task
    audit_settings = state.settings.audit
    return keep_report(state, run_audit(state.objects["train"], task.target, task.task, audit_settings.sample_rows, audit_settings.n_jobs))

# fallback of stage 05: an empty report (the later stages still run)
def fallback_audit(state):
    keep_report(state, AuditReport())
    return "empty report"

# stage 06: the LLM variants first ask which columns are known only after the...
# event (leak review); the leak guard then decides ids, saved index columns and...
# leaks from the audit, the framer and the review; drops are protected plan steps
def stage_leakguard(state):
    task = state.context.task
    split = state.context.split
    ids = list(task.id_columns)
    if split.strategy == "group" and split.key and any(entry["name"] == split.key and entry.get("type") == "id" for entry in state.objects["glance"]["columns"]):
        ids.append(split.key)
    after_event = review_leaks(state.client, state.context, state.objects["glance"], state.objects["report"], task) if state.options["leakreview"] else []
    for column, reason in after_event:
        decide(state, "leakguard", "review", column, True, f"known only after the event: {reason}", "llm")
    decisions, steps = leak_guard(state.objects["report"], task.target, ids, after_event, state.settings.leakage.drop_semantic_only, state.options["leakguard"])
    state.context.leaks = decisions
    state.objects["leak_steps"] = steps
    state.folder.write_json("leaks.json", decisions)
    for item in decisions:
        decide(state, "leakguard", "column", item["column"], item["action"] == "drop", f"{item['action']}: {item['reason']}", "rules", evidence=item["evidence"])
    dropped = sum(item["action"] == "drop" for item in decisions)
    return f"{dropped} dropped, {sum(item['action'] == 'warn' for item in decisions)} warnings"

# fallback of stage 06: structural drops only (framer ids)
def fallback_leakguard(state):
    task = state.context.task
    steps = [Step(op="drop_columns", params={"columns": list(task.id_columns)}, reason="framer id columns", protected=True, source="leakguard")] if task.id_columns else []
    state.objects["leak_steps"] = steps
    return "framer ids only"

# keeps the steps of a plan that validate one after the other (a step that...
# breaks the plan is left out and logged)
def valid_steps(state, plan, columns):
    task = state.context.task
    kept = Plan(version=plan.version, notes=plan.notes)
    for group in ["steps", "row_steps", "target_steps"]:
        for step in getattr(plan, group):
            trial = kept.model_copy(deep=True)
            getattr(trial, group).append(step)
            errors = validate_plan(trial, columns, task.target, task.task)
            if errors:
                decide(state, "plan", "step", step.op, False, f"invalid step left out: {errors[0]}", "checks", proposal=step.model_dump())
                continue
            getattr(kept, group).append(step)
    return kept

# saves every plan version to plans.json
def save_plans(state):
    state.folder.write_json("plans.json", [plan.model_dump() for plan in state.context.plans])

# stage 07: the default plan of the translator (Appendix C) after the leak...
# guard's protected drops (steps on dropped columns are left out); the LLM...
# variants then ask the cleaner for a full plan (the translator plan stays...
# when the cleaner gives no valid answer)
def stage_plan(state):
    task = state.context.task
    train = state.objects["train"]
    leak_steps = state.objects.get("leak_steps", [])
    dropped = {column for step in leak_steps for column in step.params.get("columns", [])}
    plan, skipped_issues = default_plan(state.objects["report"], task.target) if state.options["operations"] else (Plan(), [])
    for item in skipped_issues:
        decide(state, "plan", "issue", item["issue"], False, f"translator skipped: {item['reason']}", "rules")
    steps = [step for step in plan.steps if not ({step.params.get("column")} | set(step.params.get("columns", []))) & dropped]
    combined = Plan(version=0, steps=leak_steps + steps, row_steps=plan.row_steps, target_steps=plan.target_steps, notes="translator")
    combined = valid_steps(state, combined, list(train.columns))
    state.context.plans.append(combined)
    source = "translator"
    if state.options["freecode"]:
        report = state.objects["report"]
        written = free_code_plan(state.client, state.context, train, state.objects["glance"], task, protected_part(combined), issues_text(report), state.settings, state.options["samples"])
        decide(state, "plan", "freecode", "code", written is not None, "LLM cleaning code written and tested" if written is not None else "no valid cleaning code; protected steps only", "llm")
        combined = written if written is not None else protected_part(combined)
        state.context.plans.append(combined)
        source = "freecode" if written is not None else "freecode failed"
    elif state.options["cleaner"] and state.settings.cleaner.enabled:
        report = state.objects["report"]
        cleaned = clean_plan(state.client, state.context, train, state.objects["glance"], task, protected_part(combined), combined, issues_text(report), state.settings, state.options["samples"])
        if cleaned is not None:
            combined = valid_steps(state, cleaned, list(train.columns))
            combined.version = 1
            state.context.plans.append(combined)
            source = "llm"
        decide(state, "plan", "cleaner", "plan", cleaned is not None, "LLM plan used" if cleaned is not None else "no valid LLM plan; translator plan used", "llm" if cleaned is not None else "rules", proposal=combined.model_dump() if cleaned is not None else {})
    state.objects["plan"] = combined
    save_plans(state)
    return f"{source}: {len(combined.steps)} column, {len(combined.row_steps)} row, {len(combined.target_steps)} target steps"

# fallback of stage 07: the protected steps of the leak guard only
def fallback_plan(state):
    plan = Plan(version=0, steps=list(state.objects.get("leak_steps", [])), notes="fallback: leak guard steps only")
    state.objects["plan"] = plan
    state.context.plans.append(plan)
    save_plans(state)
    return "leak guard steps only"

# the gate sample of the training rows: frame, raw target values and folds
def check_data(state):
    task = state.context.task
    if "check_data" not in state.objects:
        sample, folds, _ = gate_data(state.objects["train"], task.target, task.task, state.context.split, state.settings.gate, state.settings.seed)
        state.objects["check_data"] = (sample.drop(columns=[task.target]), sample[task.target].to_numpy(), folds)
    return state.objects["check_data"]

# stage 08: data checks of every step on the folds of a sample; the LLM...
# variants repair failing steps (at most max_repairs times), the others drop them
def stage_compile(state):
    task = state.context.task
    plan = state.objects["plan"]
    frame, values, folds = check_data(state)
    llm = state.options["cleaner"] and state.settings.cleaner.enabled
    plan, custom_records = prepare_custom_steps(plan, state.client if llm else None, state.context, frame)
    for record in custom_records:
        decide(state, "compile", "custom", record["subject"], record["accepted"], record["reason"], "llm", proposal=record["proposal"])
    repaired, results, records = repair_plan(plan, frame, values, folds, task.target, task.task, state.settings, state.client if llm else None, state.context, state.options["repair"])
    for record in records:
        kind = "repair" if record["reason"].startswith("repair") else "check"
        decide(state, "compile", kind, record["subject"], record["accepted"], record["reason"], "llm" if kind == "repair" else "checks", proposal=record["proposal"])
    failing = [record["subject"] for record in records if not record["accepted"]]
    compiled = valid_steps(state, repaired, list(state.objects["train"].columns))
    compiled.version = plan.version + 1
    compiled.notes = "compiled: failing steps repaired or dropped"
    state.folder.write_json("checks.json", {"steps": results, "failing": failing})
    state.objects["plan"] = compiled
    state.context.plans.append(compiled)
    save_plans(state)
    return f"{sum(record['accepted'] for record in records)} repaired, {len(failing)} dropped"

# fallback of stage 08: the protected steps only
def fallback_compile(state):
    state.objects["plan"] = protected_only(state.objects["plan"])
    return "protected steps only"

# stage 09: the validation gate (or, without the gate, every step is accepted);...
# the LLM variants may refine the plan from the gate evidence (at most...
# max_refines rounds; the best gated plan wins)
def stage_gate(state):
    task = state.context.task
    plan = state.objects["plan"]
    if not (state.options["gate"] and state.settings.gate.enabled):
        final = plan.model_copy(deep=True)
        final.notes = "gate off: all steps accepted"
        for step in plan.steps + plan.row_steps + plan.target_steps:
            decide(state, "gate", "step", step.op, True, "gate off", "gate", proposal=step.model_dump())
    else:
        deadline = time.monotonic() + state.budget.stage_limit("gate")
        llm = state.options["cleaner"] and state.settings.cleaner.enabled
        final, rounds = gate_with_refine(plan, state.objects["train"], task, state.context.split, state.settings, state.settings.seed, state.client if llm else None, state.context, deadline, state.options["repair"])
        best = max(rounds, key=lambda result: oriented(task.metric, result["final_score"]))
        state.context.gate = {"p0": rounds[0]["p0"], "final_score": best["final_score"], "rounds": len(rounds), "accepted": best["accepted"]}
        state.folder.write_json("gate.json", {"rounds": rounds, "final_score": best["final_score"]})
        for number, result in enumerate(rounds):
            for item in result["effects"]:
                kept = result["accepted"] and not item.get("removed")
                reason = item.get("note") or ("kept" if kept else "removed by the gate")
                subject = f"{item['op']} {item['params'].get('column', '')}".strip()
                decide(state, "gate", "step", subject, kept, f"round {number}: {reason}", "gate", proposal={"op": item["op"], "params": item["params"]}, evidence={"effect": item["effect"], "round": number})
    state.context.final_plan = final
    state.context.plans.append(final)
    save_plans(state)
    rounds_text = f", {state.context.gate['rounds']} rounds" if state.context.gate.get("rounds", 1) > 1 else ""
    return f"{len(final.steps)} column steps kept" + (f" | P0 {state.context.gate['p0']['score']:.4f} -> {state.context.gate['final_score']:.4f}{rounds_text}" if state.context.gate else "")

# fallback of stage 09: P0 (protected steps only)
def fallback_gate(state):
    final = protected_only(state.objects["plan"])
    state.context.final_plan = final
    state.context.plans.append(final)
    save_plans(state)
    return "P0 (protected steps only)"

# builds the preparation pipeline of a model kind: the final plan's agent...
# steps + typing step + generic stage
def preparation(state, kind):
    return feature_pipeline(state.context.final_plan, kind, state.settings.seed)

# saves the Optuna trial log as optuna.csv
def save_trials(state, trials):
    if trials:
        frame = pd.DataFrame(trials)
        frame["params"] = frame["params"].map(json.dumps)
        frame.to_csv(state.folder.file("optuna.csv"), index=False, lineterminator="\n")

# benchmark mode: the training frame after the agent steps, for repair accuracy
def save_prepared(state, features):
    plan = state.context.final_plan
    if state.context.mode != "benchmark" or plan is None or not plan.steps:
        return
    prepared = AgentSteps(list(plan.steps)).fit(features).transform(features)
    prepared = prepared.apply(lambda column: column.astype(str) if column.dtype == object or str(column.dtype) == "category" else column)
    prepared.to_parquet(state.folder.file("prepared.parquet"))

# trains with the engine (target spellings mapped, training-only row filters,...
# log target), saves model.joblib and the CV results
def train_model(state, families=None):
    task = state.context.task
    plan = state.context.final_plan
    train = state.objects["train"]
    targets = target_steps(plan)
    labels = targets.labels(train[task.target]) if task.task != "regression" else train[task.target]
    y, encoder = encode_target(labels, task.task)
    features = train.drop(columns=[task.target])
    settings = state.settings.engine if families is None else state.settings.engine.model_copy(update={"candidates": families, "optuna": False})
    n_classes = len(encoder.classes_) if encoder is not None else None
    make_prep = lambda kind: preparation(state, kind)
    transform = targets.transform if task.task == "regression" else None
    backend = state.options["engine"] if state.options["engine"] != "own" else state.settings.backend
    trained = None
    if families is None and backend != "own":
        try:
            trained, summary = train_backend(backend, plan, features, y, task.task, task.metric, state.budget.model_seconds(), state.settings.seed, n_classes, row_filter(plan), transform)
            decide(state, "model", "backend", backend, True, f"{backend} trained on the prepared data", "checks")
        except Exception as error:
            state.folder.log_error("backend", error)
            decide(state, "model", "backend", backend, False, f"{backend} failed ({type(error).__name__}: {error}); own engine used", "checks")
    if trained is None:
        trained, summary, trials = engine.train(features, y, task.task, task.metric, state.context.split.folds, make_prep, state.budget.model_seconds(), settings, state.settings.seed, n_classes, row_filter(plan), transform)
        save_trials(state, trials)
    chosen = summary["chosen"]
    summary["cv_score"] = summary["ensemble_cv"] if len(chosen) == 2 else summary["cv"].get(chosen[0])
    state.context.model = summary
    fitted = FittedModel(trained, encoder, features.columns, task.target, task.task, task.metric, state.context.settings_hash, plan=plan, target_transform=transform)
    fitted.save(state.folder.file("model.joblib"))
    state.objects["fitted"] = fitted
    try:
        save_prepared(state, features)
    except Exception as error:
        state.folder.log_error("prepared.parquet", error)
    cv_text = f"{summary['cv_score']:.4f}" if summary["cv_score"] is not None else "n/a"
    return f"{' + '.join(chosen)} | CV {task.metric} {cv_text}"

# stage 10: the model search (own engine, or a backend on the prepared data...
# with the own engine as its fallback)
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

# the top audit issues for the report (must_fix and should_fix, at most 10)
def top_issues(state):
    report = state.objects.get("report")
    if report is None:
        return []
    chosen = [issue for issue in report.issues if issue.priority in {"must_fix", "should_fix"}][:10]
    return [f"{issue.id} [{issue.priority}] {issue.title} ({', '.join(issue.columns[:3])})" for issue in chosen]

# stage 12: scores.json and report.md; the LLM variants add a summary...
# paragraph (the template report stays when the call fails)
def stage_report(state):
    scores = dict(state.context.scores)
    scores["seconds"] = {stage.name: stage.seconds for stage in state.context.stages}
    scores["llm"] = state.context.llm
    state.folder.write_json("scores.json", scores)
    extras = {"read": state.objects.get("read", {}), "row_notes": state.objects["row_notes"], "top_issues": top_issues(state)}
    report_text = write_report(state.folder, state.context, extras)
    uses_llm = state.options["cleaner"] or state.options["leakreview"] or state.options["freecode"]
    if not (uses_llm and state.settings.report.llm_summary):
        return "report.md written"
    extras["summary"] = summarize(state.client, state.context, report_text)
    state.context.llm = dict(state.client.usage)
    write_report(state.folder, state.context, extras)
    return "report.md written" + (" with a summary" if extras["summary"] else " (no summary)")

# the stages in order with their function and fallback for the variant
def stage_table(state):
    preparing = state.options["audit"]
    table = {
        "read": (stage_read, None), "glance": (stage_glance, fallback_glance), "frame": (stage_frame, None),
        "split": (stage_split, fallback_split),
        "audit": (stage_audit, fallback_audit) if preparing else (skipped, None),
        "leakguard": (stage_leakguard, fallback_leakguard) if preparing else (skipped, None),
        "plan": (stage_plan, fallback_plan) if preparing else (skipped, None),
        "compile": (stage_compile, fallback_compile) if preparing else (skipped, None),
        "gate": (stage_gate, fallback_gate) if preparing else (skipped, None),
        "model": (stage_model, fallback_model), "test": (stage_test, None), "report": (stage_report, None),
    }
    return [(name, *table[name]) for name in STAGES]

# runs the whole pipeline on one file and returns the RunContext; demo and...
# benchmark use this same function; options["stop_after"] ends the run after...
# that stage (the cleaning-only repair checks stop after the gate)
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
            if name == state.objects["options"].get("stop_after"):
                state.context.final_plan = state.context.final_plan or state.objects.get("plan")
                return context
        logger.info(f"Done in {budget.elapsed():.1f} s | run folder: {folder.path}")
        return context
    except StopRun as error:
        logger.error(f"Stopped: {error}")
        folder.append("errors.jsonl", {"stage": "stop", "error": str(error)})
        raise
    finally:
        close_logging(handlers)
