import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from benchmark import analysis
from paths import docs_dir, results_dir
from benchmark.runner import expand_cells, load_grid

COMPARISONS = [("full", "autogluon"), ("full", "skrub"), ("full", "lightgbm"), ("full", "generic"), ("full+ag", "autogluon")]
HOLDOUT = [("E-CommereShippingData", "full"), ("E-CommereShippingData", "autogluon"), ("concrete_compressive_strength", "full"), ("concrete_compressive_strength", "autogluon")]
FULL_GRIDS = ["main", "leaks", "ablations", "types", "subsets", "budgets"]

# the standard deviation of the normalized score between seeds 0 and 1 of the...
# free systems (pilot-main seed 0 vs pilot-seed seed 1, same cells); both seeds...
# are scaled by the seed-0 cell's (best - dummy), because the seed-1 cells hold...
# only the free systems and would have another best
def seed_noise(main, seed):
    first, second = analysis.oriented(main), analysis.oriented(seed)
    spread = (first.groupby(["dataset", "level"])["oriented"].max() - first.groupby(["dataset", "level"])["dummy_oriented"].median()).rename("spread")
    key = ["system", "dataset", "level"]
    both = first.groupby(key)["oriented"].mean().rename("s0").to_frame().join(second.groupby(key)["oriented"].mean().rename("s1"), how="inner").join(spread, on=["dataset", "level"])
    both = both[both["spread"] > 1e-12]
    if both.empty:
        return np.nan
    return float((((both["s0"] - both["s1"]) / both["spread"]) ** 2).mean() ** 0.5 / np.sqrt(2))

# the effect of one comparison at some levels: the bootstrap interval of the...
# mean normalized difference, widened by the seed noise of the one-seed runs
def effect(main, system, baseline, levels, noise, resamples=10000):
    interval = analysis.bootstrap_difference(main, system, baseline, levels, resamples)
    if interval["datasets"] == 0:
        return dict(interval, low_seed=np.nan, high_seed=np.nan)
    extra = 1.96 * (noise if np.isfinite(noise) else 0.0) * np.sqrt(2) / np.sqrt(interval["datasets"] * len(levels))
    half_low, half_high = interval["mean"] - interval["low"], interval["high"] - interval["mean"]
    return dict(interval, low_seed=interval["mean"] - np.hypot(half_low, extra), high_seed=interval["mean"] + np.hypot(half_high, extra))

# the per-dataset mean and spread of one comparison's normalized differences
def dataset_effects(main, system, baseline, levels):
    pairs = analysis.paired(main, system, baseline, levels).reset_index()
    if pairs.empty:
        return np.nan, np.nan
    per_dataset = (pairs[system] - pairs[baseline]).groupby(pairs["dataset"]).mean()
    return float(per_dataset.mean()), float(per_dataset.std(ddof=1)) if len(per_dataset) > 1 else 0.0

# simulated power of the full benchmark for a family of comparisons (Holm over...
# the family): 15 datasets x 3 seeds x the given levels; dataset effects drawn...
# from the pilot mean and spread, seed noise added, Wilcoxon per comparison
def power(effects, noise, datasets=15, seeds=3, levels=2, repetitions=2000, seed=0):
    rng = np.random.default_rng(seed)
    hits = np.zeros(len(effects))
    usable = [index for index, (mean, spread) in enumerate(effects) if np.isfinite(mean)]
    for _ in range(repetitions):
        p_values = np.ones(len(effects))
        for index in usable:
            mean, spread = effects[index]
            dataset_effect = rng.normal(mean, max(spread, 1e-9), size=(datasets, 1))
            cells = dataset_effect + rng.normal(0, (noise if np.isfinite(noise) else 0.0) * np.sqrt(2), size=(datasets, levels * seeds))
            differences = cells.reshape(datasets, levels, seeds).mean(axis=2).ravel()
            p_values[index] = stats.wilcoxon(differences).pvalue if np.any(differences != 0) else 1.0
        adjusted = np.array(analysis.holm(p_values))
        hits += (adjusted < 0.05) & (np.array([effects[index][0] for index in range(len(effects))]) > 0)
    return (hits / repetitions).tolist()

# the pre-registered held-out prediction: linear interpolation between the...
# level-0 and level-3 test scores of the same system and dataset; the 95%...
# interval from the relative interpolation errors on the other pilot datasets
def holdout_predictions(main):
    data = main[main["status"].eq("ok")]
    score = data.pivot_table(index=["system", "dataset"], columns="level", values="score", aggfunc="first")
    rows = []
    for dataset, system in HOLDOUT:
        if (system, dataset) not in score.index or score.loc[(system, dataset)][[0, 3]].isna().any():
            continue
        known = score.loc[system].dropna(subset=[0, 2, 3]).drop(index=[name for name, _ in HOLDOUT], errors="ignore")
        interpolated = known[0] + 2 / 3 * (known[3] - known[0])
        errors = ((known[2] - interpolated) / interpolated.abs()).to_numpy()
        point = score.loc[(system, dataset), 0] + 2 / 3 * (score.loc[(system, dataset), 3] - score.loc[(system, dataset), 0])
        low, high = (np.quantile(errors, [0.025, 0.975]) if len(errors) else (np.nan, np.nan))
        rows.append({"dataset": dataset, "system": system, "level": 2, "prediction": point, "low": point * (1 + low) if point >= 0 else point * (1 - low), "high": point * (1 + high) if point >= 0 else point * (1 - high), "errors_from": len(errors)})
    return pd.DataFrame(rows)

# compares the held-out predictions with the actual held-out results
def holdout_check(predictions, main):
    if predictions.empty:
        return predictions
    actual = main[main["status"].eq("ok") & (main["level"] == 2)].set_index(["dataset", "system"])["score"]
    checked = predictions.copy()
    checked["actual"] = [actual.get((row.dataset, row.system), np.nan) for row in predictions.itertuples()]
    bounds = np.sort(checked[["low", "high"]].to_numpy(), axis=1)
    checked["inside"] = (checked["actual"] >= bounds[:, 0]) & (checked["actual"] <= bounds[:, 1])
    return checked

# the order of full and autogluon at 120 s (pilot-main) and 300 s (pilot-budget)
def budget_check(main, budget):
    if budget.empty:
        return pd.DataFrame()
    rows = []
    for dataset in sorted(budget["dataset"].unique()):
        for frame, seconds in [(main[(main["level"] == 3) & (main["dataset"] == dataset)], 120), (budget[budget["dataset"] == dataset], 300)]:
            data = analysis.oriented(frame[frame["system"].isin(["full", "autogluon"])]).set_index("system")["oriented"]
            if {"full", "autogluon"} <= set(data.index):
                rows.append({"dataset": dataset, "budget_s": seconds, "full_ahead": bool(data["full"] > data["autogluon"])})
    return pd.DataFrame(rows)

# the mechanism check: gain over generic per mess type next to the repair...
# accuracy of that type (benchmark/results/repairs.csv when present)
def mechanism_check(types):
    gains = []
    data = analysis.normalized(types) if not types.empty else pd.DataFrame()
    for mess, part in (data.groupby("mess_types") if not data.empty else []):
        pivot = part.pivot_table(index="dataset", columns="system", values="normalized")
        if {"full", "generic"} <= set(pivot.columns):
            gains.append({"mess": mess, "gain_over_generic": float((pivot["full"] - pivot["generic"]).mean())})
    table = pd.DataFrame(gains)
    repairs_file = results_dir / "repairs.csv"
    if table.empty or not repairs_file.exists():
        return table
    repairs = pd.read_csv(repairs_file).groupby("mess")["accuracy"].mean().rename("repair_accuracy")
    table = table.join(repairs, on="mess")
    table["explained"] = (table["gain_over_generic"] <= 0) | (table["repair_accuracy"] >= 0.5)
    return table

# the estimated computer time (2 workers) and LLM cost of every full grid, from...
# the measured pilot runs (median seconds and mean cost per system)
def grid_estimates(measured):
    per_system = measured.groupby("system").agg(seconds=("seconds", "median"), cost=("cost_usd", "mean"))
    rows = []
    for name in FULL_GRIDS:
        grid = load_grid(name)
        cells = pd.DataFrame(expand_cells(grid))
        scale = cells["budget"] / 120.0
        seconds = cells.apply(lambda cell: per_system["seconds"].get(cell["system"], cell["budget"]), axis=1) * scale
        cost = cells["system"].map(lambda system: per_system["cost"].get(system, np.nan)).fillna(0.0)
        rows.append({"grid": name, "runs": len(cells), "hours_2_workers": float(seconds.sum() / 3600 / 2), "llm_cost_usd": float(cost.sum())})
    return pd.DataFrame(rows)

# one recommendation line from the power of the main hypotheses and the checks
def recommendation(powers, holdout_ok, budget_ok):
    if not holdout_ok:
        return "not worth running yet: the projection failed its held-out check, so its estimates are unreliable"
    strong = [name for name, value in powers.items() if value >= 0.8]
    if len(strong) == len(powers) and budget_ok:
        return "run the full benchmark"
    if strong:
        return f"run a reduced benchmark: the grids for {', '.join(strong)} (power >= 0.8); the others are not likely to give a clear answer" + ("" if budget_ok else "; use 300 s, because the order changes between 120 s and 300 s")
    return "not worth running yet: no comparison reaches a power of 0.8 with the effects seen in the pilot"

# formats a number for the report
def show(value, digits=3):
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"

# builds the projection from the pilot result files and writes docs/projection.md
def build(folder=None, out=None, repetitions=2000):
    load = lambda name: analysis.load_results([name], folder)
    main, seed, budget, types = load("pilot-main"), load("pilot-seed"), load("pilot-budget"), load("pilot-types")
    if main.empty:
        raise SystemExit("pilot-main.csv is missing; run the pilot first")
    noise = seed_noise(main, seed) if not seed.empty else np.nan
    effects = {}
    for system, baseline in COMPARISONS:
        for levels in [[0], [2], [3], [2, 3]]:
            effects[(system, baseline, ",".join(map(str, levels)))] = effect(main, system, baseline, levels, noise)
    family = [dataset_effects(main, "full", baseline, [2, 3]) for _, baseline in COMPARISONS[:4]]
    powers = dict(zip([f"full vs {baseline} (levels 2+3)" for _, baseline in COMPARISONS[:4]], power(family, noise, repetitions=repetitions)))
    powers["full+ag vs autogluon (levels 2+3)"] = power([dataset_effects(main, "full+ag", "autogluon", [2, 3])], noise, repetitions=repetitions)[0]
    level0 = dataset_effects(main, "full", "generic", [0])
    predictions_file = (folder or results_dir) / "holdout-predictions.csv"
    predictions = pd.read_csv(predictions_file) if predictions_file.exists() else pd.DataFrame()
    checked = holdout_check(predictions, main)
    holdout_ok = bool(checked.empty or "inside" not in checked or checked["inside"].sum() >= 3)
    budgets = budget_check(main, budget)
    budget_ok = bool(budgets.empty or budgets.groupby("dataset")["full_ahead"].nunique().max() == 1)
    mechanism = mechanism_check(types)
    measured = pd.concat([load(name) for name in ["pilot-main", "pilot-seed", "pilot-leaks", "pilot-ablations", "pilot-types", "pilot-natural"]], ignore_index=True)
    estimates = grid_estimates(measured)
    lines = ["# Pilot projection", "", "Generated by `python -m benchmark.projection` from the pilot result files. Pilot results are a small sample (8 datasets, one seed for the LLM systems); they estimate what the full benchmark would show.", ""]
    lines += ["## Effects (mean normalized-score difference, bootstrap 95% interval over datasets)", "", f"Seed noise (sd of the normalized score between seeds, free systems): {show(noise)}. The `with seed noise` interval adds it to the one-seed estimates.", ""]
    lines += ["| comparison | levels | datasets | mean | 95% interval | with seed noise |", "|---|---|---|---|---|---|"]
    for (system, baseline, levels), value in effects.items():
        lines.append(f"| {system} vs {baseline} | {levels} | {value['datasets']} | {show(value['mean'])} | [{show(value['low'])}, {show(value['high'])}] | [{show(value['low_seed'])}, {show(value['high_seed'])}] |")
    lines += ["", "## Power of the full benchmark (15 datasets x 3 seeds, Holm-corrected Wilcoxon, 2,000 simulations)", ""]
    lines += [f"- {name}: {show(value, 2)}" for name, value in powers.items()]
    lines += ["", "## Checks", ""]
    if checked.empty:
        lines.append("- Held-out check: the predictions are not written yet.")
    elif checked["actual"].isna().all():
        lines.append("- Held-out check: predictions written; the held-out cells have not run yet.")
    else:
        lines.append(f"- Held-out check: {int(checked['inside'].sum())} of {len(checked)} actual scores inside their predicted intervals ({'reliable' if holdout_ok else 'the projection is unreliable'}).")
        lines += [f"  - {row.dataset} {row.system}: predicted {show(row.prediction, 4)} [{show(min(row.low, row.high), 4)}, {show(max(row.low, row.high), 4)}], actual {show(row.actual, 4)}" for row in checked.itertuples()]
    if budgets.empty:
        lines.append("- Budget check: no pilot-budget results.")
    else:
        lines.append("- Budget check (full ahead of autogluon at 120 s / 300 s): " + "; ".join(f"{dataset}: " + " / ".join("yes" if value else "no" for value in part.sort_values("budget_s")["full_ahead"]) for dataset, part in budgets.groupby("dataset")) + (". The order is the same." if budget_ok else ". The order changes: the 120 s results may not carry over to 300 s."))
    if mechanism.empty:
        lines.append("- Mechanism check: no pilot-types results.")
    else:
        lines.append("- Mechanism check (gain of full over generic per mess type, with repair accuracy):")
        for row in mechanism.itertuples():
            accuracy = getattr(row, "repair_accuracy", np.nan)
            flag = "" if not hasattr(row, "explained") else (" (explained)" if row.explained else " (suspicious: gain without repairs)")
            lines.append(f"  - {row.mess}: gain {show(row.gain_over_generic)}, repair accuracy {show(accuracy, 2)}{flag}")
    lines += ["", "## Recommendation per research question", ""]
    lines.append(f"- RQ1 (messy data): {recommendation({key: value for key, value in powers.items() if key.startswith('full vs')}, holdout_ok, budget_ok)}.")
    lines.append(f"- RQ2 (no harm, level 0): mean difference of full vs generic {show(level0[0])} (spread over datasets {show(level0[1])}); equivalence margin 0.02.")
    lines.append("- RQ3 to RQ5: see `docs/results.md` (leak detection, reliability and cost, ablations); their full grids are `leaks`, `subsets` and `ablations`.")
    lines.append(f"- Plug-in (full+ag vs autogluon): power {show(powers['full+ag vs autogluon (levels 2+3)'], 2)}.")
    lines += ["", "## Estimated time and cost of the full grids (from the measured pilot runs)", "", "| grid | runs | hours (2 workers) | LLM cost (USD) |", "|---|---|---|---|"]
    lines += [f"| {row.grid} | {row.runs} | {row.hours_2_workers:.1f} | {row.llm_cost_usd:.2f} |" for row in estimates.itertuples()]
    text = "\n".join(lines) + "\n"
    (out or docs_dir / "projection.md").write_text(text, encoding="utf-8", newline="\n")
    return {"effects": effects, "powers": powers, "holdout": checked, "budget": budgets, "mechanism": mechanism, "estimates": estimates, "noise": noise}

# command line: python -m benchmark.projection [--predict]; --predict writes...
# the held-out predictions (before the held-out cells run)
def main():
    if "--predict" in sys.argv:
        predictions = holdout_predictions(analysis.load_results(["pilot-main"]))
        predictions.to_csv(results_dir / "holdout-predictions.csv", index=False, lineterminator="\n")
        print(predictions.to_string(index=False))
        return 0
    result = build()
    print(json.dumps({"powers": result["powers"], "noise": result["noise"]}, indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
