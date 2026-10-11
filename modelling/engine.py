import time
import optuna
import logging
import warnings
import numpy as np
import lightgbm as lgb
from tools.metrics import oriented
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from catboost import CatBoostClassifier, CatBoostError, CatBoostRegressor
from modelling.cv import empty_oof, model_predictions, oof_score, prepare_folds

SHARES = {"linear": 0.05, "lightgbm": 0.5, "catboost": 0.35}
RESERVE_SHARE = 0.10
MAX_TREES = 2000
EARLY_STOPPING = 50
logger = logging.getLogger("automl.engine")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# the preparation kind each family needs (one-hot and scaling for linear)
def prep_kind(family):
    return "linear" if family == "linear" else "tree"

# the default parameters of a family (always tried first)
def default_params(family):
    if family == "lightgbm":
        return {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 0.001}
    if family == "catboost":
        return {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0}
    if family == "linear":
        return {"strength": 1.0}
    return {}

# the Optuna search space of a family (§3.12)
def suggest(trial, family):
    if family == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
        }
    if family == "catboost":
        return {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
    if family == "linear":
        return {"strength": trial.suggest_float("strength", 0.001, 100.0, log=True)}
    return {}

# stops LightGBM boosting when the deadline has passed
def lightgbm_deadline(deadline):
    def callback(environment):
        if time.monotonic() > deadline:
            raise lgb.callback.EarlyStopException(environment.iteration, environment.evaluation_result_list)
    callback.order = 30
    return callback

# stops CatBoost boosting when the deadline has passed
class CatBoostDeadline:
    def __init__(self, deadline):
        self.deadline = deadline

    def after_iteration(self, info):
        return time.monotonic() <= self.deadline

# builds an unfitted model of a family; trees is the fixed number of trees of a...
# refit (None: up to 2,000 with early stopping)
def make_model(family, task, params, seed, n_jobs, trees=None):
    classify = task != "regression"
    if family == "dummy":
        return DummyClassifier(strategy="prior") if classify else DummyRegressor(strategy="mean")
    if family == "linear":
        if classify:
            return LogisticRegression(C=params["strength"], max_iter=1000)
        return Ridge(alpha=1.0 / params["strength"])
    if family == "lightgbm":
        options = dict(params, n_estimators=trees or MAX_TREES, random_state=seed, n_jobs=n_jobs, verbose=-1, subsample_freq=1)
        return lgb.LGBMClassifier(**options) if classify else lgb.LGBMRegressor(**options)
    options = dict(params, iterations=trees or MAX_TREES, random_seed=seed, thread_count=n_jobs, verbose=False, allow_writing_files=False)
    return CatBoostClassifier(**options) if classify else CatBoostRegressor(**options)

# fits one model; boosted models stop early on a 10% inner split of the...
# training rows (stratified for classification) and at the deadline; returns...
# the model and its best number of trees
def fit_model(family, task, params, x, y, seed, n_jobs, deadline):
    model = make_model(family, task, params, seed, n_jobs)
    if family not in {"lightgbm", "catboost"}:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit(x, y), None
    stratify = y if task != "regression" and np.bincount(y).min() >= 2 and len(np.unique(y)) > 1 else None
    if len(y) < 30:
        fit_x, stop_x, fit_y, stop_y = x, x, y, y
    else:
        fit_x, stop_x, fit_y, stop_y = train_test_split(x, y, test_size=0.1, random_state=seed, stratify=stratify)
    if family == "lightgbm":
        callbacks = [lgb.early_stopping(EARLY_STOPPING, verbose=False), lightgbm_deadline(deadline)]
        model.fit(fit_x, fit_y, eval_X=(stop_x,), eval_y=(stop_y,), callbacks=callbacks)
        return model, max(1, model.best_iteration_ or model.n_estimators)
    model.fit(fit_x, fit_y, eval_set=(stop_x, stop_y), early_stopping_rounds=EARLY_STOPPING, callbacks=[CatBoostDeadline(deadline)])
    return model, max(1, model.get_best_iteration() or 1) + 1

# fits a family with given parameters on every fold; each fold gets an equal...
# share of the time left, so a cut by the deadline treats all folds alike;...
# returns the out-of-fold predictions and the mean best number of trees
def evaluate(family, task, params, fold_data, rows, n_classes, seed, n_jobs, deadline):
    oof = empty_oof(rows, task, n_classes)
    trees = []
    for position, fold in enumerate(fold_data):
        now = time.monotonic()
        fold_deadline = now + max(0.0, deadline - now) / (len(fold_data) - position)
        model, best = fit_model(family, task, params, fold.train_x, fold.train_y, seed, n_jobs, fold_deadline)
        oof[fold.valid_rows] = model_predictions(model, fold.valid_x, task, n_classes)
        if best is not None:
            trees.append(best)
    return oof, int(np.mean(trees)) if trees else None

# searches one family with Optuna (TPE, seeded) inside its time; the default...
# parameters come first; the search stops after patience trials without...
# improvement; returns the best result and the trial log
def search_family(family, task, metric, fold_data, y, n_classes, seed, settings, seconds, hard_deadline, scorer=None):
    start = time.monotonic()
    family_deadline = min(start + seconds, hard_deadline)
    best = {"score": -np.inf}
    log = []

    def objective(trial):
        params = suggest(trial, family)
        trial_start = time.perf_counter()
        deadline = hard_deadline if trial.number == 0 else family_deadline
        oof, trees = evaluate(family, task, params, fold_data, len(y), n_classes, seed, settings.n_jobs, deadline)
        score = scorer(oof) if scorer is not None else oof_score(task, y, oof, n_classes)
        value = oriented(metric, score) if np.isfinite(score) else -np.inf
        log.append({"family": family, "trial": trial.number, "score": score, "seconds": round(time.perf_counter() - trial_start, 3), "trees": trees, "params": params})
        if value > best["score"]:
            best.update(score=value, raw_score=score, params=params, oof=oof, trees=trees, trial=trial.number)
        return value

    def patience(study, trial):
        if trial.number - best.get("trial", 0) >= settings.patience_trials or time.monotonic() >= family_deadline:
            study.stop()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(default_params(family))
    trials = None if settings.optuna and family != "dummy" else 1
    try:
        study.optimize(objective, n_trials=trials, timeout=max(1.0, seconds), callbacks=[patience], catch=(ValueError, RuntimeError, lgb.basic.LightGBMError, CatBoostError))
    except Exception as error:
        logger.warning(f"Optuna failed for {family}: {type(error).__name__}: {error}; using default parameters")
        if "params" not in best:
            params = default_params(family)
            oof, trees = evaluate(family, task, params, fold_data, len(y), n_classes, seed, settings.n_jobs, hard_deadline)
            score = scorer(oof) if scorer is not None else oof_score(task, y, oof, n_classes)
            best.update(score=oriented(metric, score), raw_score=score, params=params, oof=oof, trees=trees, trial=0)
    return best, log

# a trained model: one or more (preparation, model) members whose predictions...
# are averaged; probabilities are aligned to all classes
class TrainedModel:
    def __init__(self, members, task, n_classes):
        self.members = members
        self.task = task
        self.n_classes = n_classes

    def predict_values(self, frame):
        outputs = [model_predictions(model, prep.transform(frame), self.task, self.n_classes) for _, prep, model in self.members]
        return np.mean(outputs, axis=0)

    def families(self):
        return [family for family, _, _ in self.members]

# refits one family on all training rows (row filter applied) with its best...
# parameters and best number of trees
def refit(family, task, best, frame, y, make_prep, seed, n_jobs, row_filter):
    if row_filter is not None:
        keep = row_filter.mask(frame, y)
        frame, y = frame[keep], y[keep]
    prep = make_prep(prep_kind(family))
    x = prep.fit_transform(frame, y)
    model = make_model(family, task, best.get("params", {}), seed, n_jobs, trees=best.get("trees"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, y)
    return prep, model

# for large tables the search runs on a fixed sample of at most max_rows rows...
# (stratified for classification); the folds are mapped to the sample
def search_sample(frame, y, folds, task, max_rows, seed):
    if len(frame) <= max_rows:
        return frame, y, folds
    stratify = y if task != "regression" and np.bincount(y).min() >= 2 else None
    chosen, _ = train_test_split(np.arange(len(frame)), train_size=max_rows, random_state=seed, stratify=stratify)
    chosen = np.sort(chosen)
    position = {old: new for new, old in enumerate(chosen)}
    sample_folds = [[position[row] for row in fold if row in position] for fold in folds]
    logger.info(f"Search on a sample of {max_rows} of {len(frame)} rows; the refit uses all rows")
    return frame.iloc[chosen], y[chosen], [fold for fold in sample_folds if fold]

# the model stage: cross-validated search of every candidate family under the...
# time budget, an average of the best two families when it beats the best one,...
# and a refit on all training rows; the dummy model always exists; with a...
# target transform (log_target) models learn the transformed target and are...
# scored on the original scale
def train(frame, y, task, metric, folds, make_prep, seconds, settings, seed, n_classes=None, row_filter=None, target_transform=None):
    start = time.monotonic()
    hard_deadline = start + seconds * (1 - RESERVE_SHARE)
    n_classes = n_classes or (int(y.max()) + 1 if task != "regression" else 0)
    full_frame, full_y = frame, y
    frame, y, folds = search_sample(frame, y, folds, task, settings.max_search_rows, seed)
    y_true = y
    scorer = None
    if target_transform is not None:
        y = target_transform.forward(y)
        full_y = target_transform.forward(full_y)
        scorer = lambda oof: oof_score(task, y_true, target_transform.inverse(oof), n_classes)
    prepared = {}
    results = {}
    trial_log = []
    candidates = [family for family in settings.candidates if family in {"dummy", "linear", "lightgbm", "catboost"}]
    if "dummy" not in candidates:
        candidates = ["dummy"] + candidates
    for position, family in enumerate(candidates):
        kind = prep_kind(family)
        try:
            if kind not in prepared:
                prepared[kind] = prepare_folds(frame, y, folds, lambda kind=kind: make_prep(kind), row_filter)
            left = hard_deadline - time.monotonic()
            share = SHARES.get(family, 0.0) / max(1e-9, sum(SHARES.get(name, 0.0) for name in candidates[position:]))
            family_seconds = left * share if family != "dummy" else left
            if family != "dummy" and left <= 1:
                logger.info(f"No time left for {family}; skipped")
                continue
            best, log = search_family(family, task, metric, prepared[kind], y, n_classes, seed, settings, family_seconds, hard_deadline, scorer)
            trial_log += log
            if "params" in best:
                results[family] = best
                logger.info(f"{family}: CV {metric} {best['raw_score']:.5f} ({len(log)} trials)")
        except Exception as error:
            logger.warning(f"Candidate {family} failed: {type(error).__name__}: {error}")
    ranked = sorted((family for family in results if family != "dummy"), key=lambda family: -results[family]["score"])
    chosen = ranked[:1] or ["dummy"]
    ensemble_score = None
    if settings.ensemble and len(ranked) >= 2:
        average = (results[ranked[0]]["oof"] + results[ranked[1]]["oof"]) / 2
        ensemble_score = scorer(average) if scorer is not None else oof_score(task, y, average, n_classes)
        if oriented(metric, ensemble_score) > results[ranked[0]]["score"]:
            chosen = ranked[:2]
    members = []
    for family in chosen:
        try:
            members.append((family, *refit(family, task, results[family], full_frame, full_y, make_prep, seed, settings.n_jobs, row_filter)))
        except Exception as error:
            logger.warning(f"Refit of {family} failed: {type(error).__name__}: {error}")
    if not members:
        members.append(("dummy", *refit("dummy", task, {}, full_frame, full_y, make_prep, seed, settings.n_jobs, row_filter)))
    summary = {
        "chosen": [family for family, _, _ in members],
        "cv": {family: result["raw_score"] for family, result in results.items()},
        "ensemble_cv": ensemble_score,
        "best_params": {family: result["params"] for family, result in results.items()},
        "trees": {family: result.get("trees") for family, result in results.items()},
        "seconds": round(time.monotonic() - start, 3),
    }
    return TrainedModel(members, task, n_classes), summary, trial_log
