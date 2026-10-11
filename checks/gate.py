import time
import numpy as np
import lightgbm as lgb
from schemas import Plan
from tools.splitter import make_folds
from benchmark.splits import fixed_sample
from tools.metrics import compute_scores, oriented
from modelling.cv import model_predictions, prepare_folds
from operations.compiler import feature_pipeline, row_filter, target_steps, validate_plan

# the quick model of the gate: LightGBM with fixed parameters
def quick_model(task, n_jobs, seed):
    options = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "n_jobs": n_jobs, "verbose": -1, "random_state": seed}
    return lgb.LGBMRegressor(**options) if task == "regression" else lgb.LGBMClassifier(**options)

# the allowed difference between two scores of a metric: absolute or...
# relative to the reference score
def tolerance(settings, metric, reference):
    rule = settings.tolerance.get(metric, {})
    if "absolute" in rule:
        return rule["absolute"]
    return rule.get("relative", 0.0) * abs(reference)

# the labels the gate trains on: target spellings mapped, then integer codes...
# (classification) or floats, with a log transform when the plan has one
def gate_labels(plan, values, task):
    steps = target_steps(plan)
    if task == "regression":
        numbers = np.asarray(values, dtype=float)
        return numbers, steps.transform, None
    labels = steps.labels(values)
    classes = sorted(set(labels.tolist()))
    position = {label: index for index, label in enumerate(classes)}
    return np.array([position[label] for label in labels]), None, len(classes)

# scores a plan on the gate folds; validation parts are never row-filtered;...
# returns the mean score and the score of every fold
def score_plan(plan, frame, values, folds, task, metric, settings, seed):
    y, transform, n_classes = gate_labels(plan, values, task)
    train_y = transform.forward(y) if transform is not None else y
    fold_data = prepare_folds(frame, train_y, folds, lambda: feature_pipeline(plan, "tree", seed), row_filter(plan))
    scores = []
    for fold in fold_data:
        model = quick_model(task, settings.n_jobs, seed).fit(fold.train_x, fold.train_y)
        prediction = model_predictions(model, fold.valid_x, task, n_classes or 0)
        if transform is not None:
            prediction = transform.inverse(prediction)
        classes = None if task == "regression" else list(range(n_classes))
        scores.append(compute_scores(task, y[fold.valid_rows], prediction, classes)["score"])
    return float(np.nanmean(scores)), scores

# a copy of a plan without one step (by group and position)
def without(plan, group, position):
    copy = plan.model_copy(deep=True)
    getattr(copy, group).pop(position)
    return copy

# keeps only the protected steps of a plan (P0)
def protected_only(plan):
    return Plan(version=plan.version, steps=[step for step in plan.steps if step.protected], row_steps=[step for step in plan.row_steps if step.protected], target_steps=[step for step in plan.target_steps if step.protected], notes="P0")

# the gated steps of a plan as (group, position, step)
def gated_steps(plan):
    return [(group, position, step) for group in ["steps", "row_steps", "target_steps"] for position, step in enumerate(getattr(plan, group)) if not step.protected]

# the gate sample and folds: a stratified sample of at most max_rows training...
# rows and gate.folds folds with the split strategy of the run
def gate_data(train, target, task, split, settings, seed):
    sample = fixed_sample(train, target, task, settings.max_rows, seed)
    folds, notes = make_folds(sample, target, task, split.strategy, split.key, settings.folds, seed)
    return sample, folds, notes

# the validation gate: P1 (all steps) is kept when its CV score is not worse...
# than P0 (protected steps only) within the tolerance; each gated step is...
# tested by leaving it out, steps that hurt by more than the tolerance are...
# removed, and the better of P0 and the reduced plan is the result; the...
# deadline stops the per-step tests (untested steps stay)
def run_gate(plan, train, target, task, metric, split, settings, seed, columns, deadline=None):
    sample, folds, notes = gate_data(train, target, task, split, settings, seed)
    frame = sample.drop(columns=[target])
    values = sample[target].to_numpy()
    started = time.monotonic()
    base = protected_only(plan)
    base_score, base_folds = score_plan(base, frame, values, folds, task, metric, settings, seed)
    full_score, full_folds = score_plan(plan, frame, values, folds, task, metric, settings, seed)
    result = {
        "metric": metric, "rows": len(sample), "folds": len(folds), "notes": notes,
        "p0": {"score": base_score, "folds": base_folds}, "p1": {"score": full_score, "folds": full_folds}, "effects": [],
    }
    gated = gated_steps(plan)
    tol = tolerance(settings, metric, base_score)
    effects = []
    if settings.per_op and 0 < len(gated) <= settings.max_per_op_tests:
        for group, position, step in gated:
            if deadline is not None and time.monotonic() > deadline:
                effects.append({"group": group, "position": position, "op": step.op, "params": step.params, "effect": None, "note": "not tested (time)"})
                continue
            reduced = without(plan, group, position)
            if validate_plan(reduced, columns, target, task):
                effects.append({"group": group, "position": position, "op": step.op, "params": step.params, "effect": None, "note": "other steps need it"})
                continue
            score, _ = score_plan(reduced, frame, values, folds, task, metric, settings, seed)
            effect = oriented(metric, full_score) - oriented(metric, score)
            effects.append({"group": group, "position": position, "op": step.op, "params": step.params, "effect": effect, "score_without": score})
    result["effects"] = effects
    harmful = [item for item in effects if item["effect"] is not None and item["effect"] < -tol]
    reduced = plan.model_copy(deep=True)
    for item in sorted(harmful, key=lambda item: -item["position"]):
        getattr(reduced, item["group"]).pop(item["position"])
    reduced_score, reduced_folds = (full_score, full_folds)
    if harmful:
        reduced_score, reduced_folds = score_plan(reduced, frame, values, folds, task, metric, settings, seed)
    result["reduced"] = {"score": reduced_score, "folds": reduced_folds, "removed": [item["op"] for item in harmful]}
    accepted = oriented(metric, reduced_score) >= oriented(metric, base_score) - tol
    final = reduced if accepted else base
    result.update(tolerance=tol, accepted=accepted, final_score=reduced_score if accepted else base_score, seconds=round(time.monotonic() - started, 2))
    final.version = plan.version + 1
    final.notes = "gate: reduced P1" if accepted else "gate: P0 (protected steps only)"
    return final, result
