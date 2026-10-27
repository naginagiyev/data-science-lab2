import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from benchmark.systems.lightgbm import prepare
from paths import natural_cache_dir, results_dir
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from benchmark.natural import NATURAL_DATASETS, natural_table, read_text
from sklearn.model_selection import GroupShuffleSplit, KFold, StratifiedKFold, cross_val_score, train_test_split

MEASURES_FILE = results_dir / "natural-measures.json"
NUMBER = re.compile(r"-?\d+(\.\d+)?")

# the checks of the quality profile (M2), written from each provider's...
# documentation before any system ran: missing tokens, values that are "not...
# applicable" (not missing), invalid values, entity columns and documented...
# label inconsistencies; the source of every rule is named
DOCUMENTED_RULES = {
    "beers": {"missing_tokens": ["N/A"], "source": "the owner's clean file writes N/A as empty"},
    "coupon": {"source": "UCI 603: has missing values (empty cells)"},
    "hr-analytics": {"missing_tokens": ["nan"], "source": "no documentation; 'nan' is the text of a missing value (observed)"},
    "bank-additional": {"missing_tokens": ["unknown"], "source": "bank-additional-names.txt: missing values are coded 'unknown'; pdays 999 = not contacted (a code, not missing)"},
    "titanic": {"source": "titanic3: empty cells are missing"},
    "student-performance": {"source": "UCI 320: no missing values"},
    "diabetes130": {
        "missing_tokens": ["?"], "invalid": {"gender": ["Unknown/Invalid"]}, "entity": "patient_nbr",
        "source": "UCI 296 file and Strack et al. 2014 (missing shares); gender 'Unknown/Invalid' is an invalid value",
    },
    "hotels": {
        "missing_tokens": ["NA"], "not_applicable": ["NULL"], "synonyms": {"Meal": [["Undefined", "SC"]]},
        "rules": ["ADR < 0", "zero guests"],
        "source": "Antonio et al. 2019: NULL means 'not applicable'; Meal 'Undefined/SC - no meal package'; ADR is a price, a booking has guests",
    },
    "airbnb-austin": {"source": "Inside Airbnb data dictionary: price is the daily price; no missing tokens documented"},
}

# the empty or token-missing cells of a raw text table
def missing_mask(raw, tokens):
    stripped = raw.apply(lambda column: column.str.strip())
    return stripped.eq("") | stripped.isin(tokens)

# the rows that break the documented hotel rules (negative rate, no guests)
def hotel_rule_rows(raw):
    numbers = raw[["ADR", "Adults", "Children", "Babies"]].apply(lambda column: pd.to_numeric(column.str.strip(), errors="coerce"))
    return int((numbers["ADR"] < 0).sum()), int((numbers[["Adults", "Children", "Babies"]].fillna(0).sum(axis=1) == 0).sum())

# M2: completeness, validity, uniqueness and consistency ratios of the...
# published table, with only the documented checks (a lower bound)
def quality_profile(name, raw):
    rules = DOCUMENTED_RULES[name]
    missing = missing_mask(raw, rules.get("missing_tokens", []))
    cells = raw.size
    profile = {
        "rows": len(raw), "columns": raw.shape[1], "source": rules["source"],
        "completeness": round(1 - missing.values.sum() / cells, 4),
        "hidden_missing_cells": int(raw.apply(lambda column: column.str.strip()).isin(rules.get("missing_tokens", [])).values.sum()),
        "empty_columns": int((missing.mean() == 1).sum()),
        "columns_over_half_missing": int((missing.mean() > 0.5).sum()),
        "not_applicable_cells": int(raw.apply(lambda column: column.str.strip()).isin(rules.get("not_applicable", [])).values.sum()),
        "uniqueness": round(1 - raw.duplicated().mean(), 4),
        "duplicate_rows": int(raw.duplicated().sum()),
    }
    invalid = sum(int(raw[column].str.strip().isin(values).sum()) for column, values in rules.get("invalid", {}).items())
    if name == "hotels":
        negative, no_guests = hotel_rule_rows(raw)
        profile.update(negative_rate_rows=negative, zero_guest_rows=no_guests)
        invalid += negative + no_guests
    profile["invalid_values"] = invalid
    if "entity" in rules:
        repeated = raw[rules["entity"]].duplicated(keep=False)
        profile.update(entity_rows_repeated=int(repeated.sum()), entities=int(raw[rules["entity"]].nunique()))
    synonyms = [(column, pair, int(raw[column].str.strip().isin(pair[:1]).sum())) for column, pairs in rules.get("synonyms", {}).items() for pair in pairs]
    profile["label_synonym_cells"] = {f"{column}: {' = '.join(pair)}": count for column, pair, count in synonyms}
    return profile

# the type of one beers cell error: missing token, format or unit only, a...
# value packed with another column, or a wrong value
def error_type(dirty, clean, row_clean):
    if dirty.strip() in ("N/A", "") and clean.strip() == "":
        return "missing_token"
    numbers = [NUMBER.search(text) for text in (dirty, clean)]
    if all(numbers) and float(numbers[0].group()) == float(numbers[1].group()):
        return "format_or_unit"
    if clean and dirty.startswith(clean) and dirty[len(clean):].strip() in set(row_clean):
        return "packed"
    if dirty.strip() == "" and clean.strip():
        return "missing_value"
    return "wrong_value"

# M1: erroneous cells / all cells of beers against the owner's clean file,...
# per column and per error type
def cell_error_rate():
    dirty = read_text(natural_cache_dir / "beers" / "dirty.csv")
    clean = read_text(natural_cache_dir / "beers" / "clean.csv")
    clean.columns = dirty.columns
    differ = dirty.values != clean.values
    types = {}
    for row, column in zip(*np.nonzero(differ)):
        kind = error_type(dirty.iat[row, column], clean.iat[row, column], clean.iloc[row].tolist())
        types[kind] = types.get(kind, 0) + 1
    return {
        "cells": int(differ.size), "erroneous_cells": int(differ.sum()), "error_rate": round(differ.mean(), 4),
        "per_column": {column: int(differ[:, index].sum()) for index, column in enumerate(dirty.columns) if differ[:, index].any()},
        "per_type": dict(sorted(types.items(), key=lambda item: -item[1])),
    }

# the primary score of a fitted reference model on a test frame (higher is...
# better: AUC, minus log loss, minus RMSE)
def reference_score(task, train, test, target):
    train_features = train.drop(columns=[target])
    features, categories = prepare(train_features.copy())
    test_features = prepare(test[train_features.columns].copy(), categories)[0]
    options = {"n_jobs": 4, "verbose": -1, "random_state": 0}
    if task == "regression":
        model = LGBMRegressor(**options).fit(features, train[target].astype(float))
        return -float(np.sqrt(mean_squared_error(test[target].astype(float), model.predict(test_features))))
    model = LGBMClassifier(**options).fit(features, train[target])
    probabilities = model.predict_proba(test_features)
    if task == "binary-classification":
        return float(roc_auc_score(test[target] == model.classes_[1], probabilities[:, 1]))
    return -float(log_loss(test[target], probabilities, labels=model.classes_))

# types a raw text table the way pd.read_csv would (a column becomes numeric...
# only when every non-empty value is a number), so the reference model sees...
# what a quick data scientist sees
def typed(raw):
    frame = raw.replace({"": np.nan})
    for column in frame.columns:
        numbers = pd.to_numeric(frame[column], errors="coerce")
        if numbers.notna().sum() == frame[column].notna().sum():
            frame[column] = numbers
    return frame

# the mean, standard deviation and 95% t-interval of paired differences
def summary(differences):
    values = np.asarray(differences, dtype=float)
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return {"mean": round(values.mean(), 4), "sd": round(values.std(ddof=1), 4), "ci95": [round(values.mean() - half, 4), round(values.mean() + half, 4)], "n": len(values)}

# M4: score of the reference model trained on the dirty vs the clean beers...
# features, both tested on the clean test rows (CleanML's BD scenario), over...
# 20 random splits, with a paired t-test
def cleaning_impact(splits=20):
    raw, spec = natural_table("beers")
    clean = read_text(natural_cache_dir / "beers" / "clean.csv")
    clean.columns = read_text(natural_cache_dir / "beers" / "dirty.csv").columns
    clean = clean.set_index("index").loc[raw["index"]].reset_index()
    dirty_frame, clean_frame = typed(raw), typed(clean.assign(abv=raw["abv"]))
    differences = []
    for seed in range(splits):
        train_rows, test_rows = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=seed)
        clean_test = clean_frame.iloc[test_rows]
        dirty_score = reference_score(spec["task"], dirty_frame.iloc[train_rows], clean_test, "abv")
        clean_score = reference_score(spec["task"], clean_frame.iloc[train_rows], clean_test, "abv")
        differences.append(clean_score - dirty_score)
    result = summary(differences)
    result["p_value"] = round(float(stats.ttest_1samp(differences, 0.0).pvalue), 6)
    result["metric"] = "rmse (difference = dirty rmse - clean rmse; positive means cleaning helps)"
    return result

# the single-column features of a leak column: its numbers (or category codes)...
# and whether it is empty
def leak_features(column):
    numbers = pd.to_numeric(column.str.replace(r"[$,]", "", regex=True), errors="coerce")
    values = numbers if numbers.notna().mean() > 0.5 else column.astype("category").cat.codes.astype(float).where(column.str.strip() != "")
    return pd.DataFrame({"value": values.fillna(-1e9), "empty": (column.str.strip() == "").astype(int)})

# leak strength: cross-validated AUC or R2 of a depth-3 tree on the leak...
# column alone (and, for a combined leak, on the two columns together)
def leak_strength(raw, spec, columns):
    target, task = spec["target"], spec["task"]
    features = pd.concat([leak_features(raw[column]).add_prefix(f"{column}_") for column in columns], axis=1)
    if task == "regression":
        model, folds, scoring, labels = DecisionTreeRegressor(max_depth=3, random_state=0), KFold(5, shuffle=True, random_state=0), "r2", raw[target].astype(float)
    else:
        model, folds = DecisionTreeClassifier(max_depth=3, random_state=0), StratifiedKFold(5, shuffle=True, random_state=0)
        scoring, labels = ("roc_auc", raw[target]) if task == "binary-classification" else ("neg_log_loss", raw[target])
    return round(float(cross_val_score(model, features, labels, cv=folds, scoring=scoring).mean()), 4)

# leak inflation: reference score with the leak columns minus without them...
# (both trained and tested the same way) over repeated 80/20 splits; for...
# diabetes130, a random split minus a split by patient
def leak_inflation(raw, spec, splits=5):
    target, task, frame = spec["target"], spec["task"], typed(raw)
    differences = []
    for seed in range(splits):
        if spec.get("groups"):
            random_train, random_test = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=seed, stratify=raw[target])
            group_train, group_test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(raw, groups=raw[spec["groups"]]))
            differences.append(reference_score(task, frame.iloc[random_train], frame.iloc[random_test], target)
                               - reference_score(task, frame.iloc[group_train], frame.iloc[group_test], target))
            continue
        stratify = raw[target] if task != "regression" else None
        train_rows, test_rows = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=seed, stratify=stratify)
        with_leak = reference_score(task, frame.iloc[train_rows], frame.iloc[test_rows], target)
        without = frame.drop(columns=spec["leaks"])
        differences.append(with_leak - reference_score(task, without.iloc[train_rows], without.iloc[test_rows], target))
    return summary(differences)

# all measures of one natural dataset
def measure(name):
    raw, spec = natural_table(name)
    table = read_text(natural_cache_dir / name / spec["table"], spec.get("sep", ","))
    result = {"kind": spec["kind"], "task": spec["task"], "profile": quality_profile(name, table)}
    if name == "beers":
        result["cell_errors"] = cell_error_rate()
        result["cleaning_impact"] = cleaning_impact()
    strengths = {column: leak_strength(raw, spec, [column]) for column in spec.get("leaks", [])}
    if name == "airbnb-austin":
        strengths["estimated_revenue_l365d + estimated_occupancy_l365d"] = leak_strength(raw, spec, ["estimated_revenue_l365d", "estimated_occupancy_l365d"])
    if strengths:
        result["leak_strength"] = strengths
    if spec.get("leaks") or spec.get("groups"):
        result["leak_inflation"] = leak_inflation(raw, spec)
    return result

# command line: python -m benchmark.messmeasures [names...] writes...
# benchmark/results/natural-measures.json
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Measure the mess and the leaks of the natural datasets.")
    parser.add_argument("names", nargs="*")
    arguments = parser.parse_args()
    results = json.loads(MEASURES_FILE.read_text(encoding="utf-8")) if MEASURES_FILE.exists() else {}
    for name in arguments.names or list(NATURAL_DATASETS):
        results[name] = measure(name)
        print(name, json.dumps(results[name], ensure_ascii=False)[:600], flush=True)
    MEASURES_FILE.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
