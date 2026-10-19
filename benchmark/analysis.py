import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from paths import results_dir

LOWER_IS_BETTER = {"rmse", "log_loss"}
CELL = ["dataset", "level", "seed", "budget_s"]
NEMENYI_Q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

# reads the results of one or more grids from runs.csv; a missing file or grid...
# gives no rows
def load_results(names, folder=None):
    path = (folder or results_dir) / "runs.csv"
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["grid"])
    frames = [frame[frame["grid"] == name] for name in names if (frame["grid"] == name).any()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# higher-is-better scores: rmse and log_loss change sign; a failed run gets the...
# dummy score of its cell (failures are not removed)
def oriented(frame):
    frame = frame.copy()
    sign = np.where(frame["metric"].isin(LOWER_IS_BETTER), -1.0, 1.0)
    frame["dummy_oriented"] = frame.groupby(CELL)["dummy_score"].transform("median") * sign
    frame["oriented"] = np.where(frame["status"].eq("ok") & frame["score"].notna(), frame["score"] * sign, frame["dummy_oriented"])
    return frame

# the normalized score per run: (score - dummy) / (best - dummy) in its cell,...
# best = the best score of any system in the cell; cells where the best equals...
# the dummy are left out
def normalized(frame):
    frame = oriented(frame)
    best = frame.groupby(CELL)["oriented"].transform("max")
    spread = best - frame["dummy_oriented"]
    frame["normalized"] = np.where(spread > 1e-12, (frame["oriented"] - frame["dummy_oriented"]) / spread.where(spread > 1e-12, 1.0), np.nan)
    return frame[frame["normalized"].notna()]

# the mean normalized score per system and level: mean over seeds, then over...
# datasets
def mean_normalized(frame):
    per_dataset = normalized(frame).groupby(["system", "level", "dataset"])["normalized"].mean()
    return per_dataset.groupby(["system", "level"]).mean().unstack("level")

# the average rank per system and level (rank 1 = best in its cell)
def average_ranks(frame):
    data = oriented(frame)
    data["rank"] = data.groupby(CELL)["oriented"].rank(ascending=False, method="average")
    return data.groupby(["system", "level"])["rank"].mean().unstack("level")

# Friedman test and Nemenyi critical difference (Demšar 2006) on one level:...
# blocks are datasets (mean over seeds) with a result for every system
def friedman_nemenyi(frame, level, alpha_q=NEMENYI_Q):
    data = oriented(frame[frame["level"] == level])
    table = data.groupby(["dataset", "system"])["oriented"].mean().unstack("system").dropna()
    systems = list(table.columns)
    if len(table) < 2 or len(systems) < 2:
        return {"level": level, "datasets": len(table), "systems": systems, "ranks": {}, "p_value": None, "cd": None}
    ranks = table.rank(axis=1, ascending=False).mean()
    statistic, p_value = stats.friedmanchisquare(*[table[system] for system in systems]) if len(systems) >= 3 else (None, None)
    k, n = len(systems), len(table)
    cd = alpha_q.get(k, alpha_q[max(alpha_q)]) * np.sqrt(k * (k + 1) / (6 * n))
    return {"level": level, "datasets": n, "systems": systems, "ranks": ranks.sort_values().to_dict(), "p_value": p_value, "cd": float(cd)}

# the paired normalized scores of two systems on the same (dataset, level,...
# seed) cells of the given levels
def paired(frame, system, baseline, levels):
    data = normalized(frame[frame["level"].isin(levels)])
    table = data.pivot_table(index=CELL, columns="system", values="normalized")
    if system not in table or baseline not in table:
        return pd.DataFrame(columns=[system, baseline])
    return table[[system, baseline]].dropna()

# Holm-adjusted p-values (in the order given)
def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (len(p_values) - position) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()

# Wilcoxon signed-rank tests of one system against each baseline on the given...
# levels, with wins, ties, losses and Holm correction over the baselines
def wilcoxon_table(frame, system, baselines, levels):
    rows = []
    for baseline in baselines:
        pairs = paired(frame, system, baseline, levels)
        difference = pairs[system] - pairs[baseline]
        wins, ties, losses = int((difference > 1e-9).sum()), int((difference.abs() <= 1e-9).sum()), int((difference < -1e-9).sum())
        p_value = float(stats.wilcoxon(difference).pvalue) if (difference.abs() > 1e-9).sum() >= 1 else 1.0
        rows.append({"system": system, "baseline": baseline, "levels": ",".join(map(str, levels)), "pairs": len(pairs), "mean_difference": float(difference.mean()) if len(pairs) else np.nan, "wins": wins, "ties": ties, "losses": losses, "p_value": p_value})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = holm(table["p_value"].to_numpy())
    return table

# the bootstrap 95% interval (resampling datasets) of the mean normalized-score...
# difference of two systems on the given levels
def bootstrap_difference(frame, system, baseline, levels, resamples=10000, seed=0):
    pairs = paired(frame, system, baseline, levels).reset_index()
    if pairs.empty:
        return {"mean": np.nan, "low": np.nan, "high": np.nan, "datasets": 0}
    per_dataset = (pairs[system] - pairs[baseline]).groupby(pairs["dataset"]).mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(per_dataset, size=(resamples, len(per_dataset)), replace=True).mean(axis=1)
    return {"mean": float(per_dataset.mean()), "low": float(np.quantile(draws, 0.025)), "high": float(np.quantile(draws, 0.975)), "datasets": len(per_dataset)}

# reliability and cost per system: success rate, median seconds, mean tokens,...
# mean dollars, mean standard deviation of the normalized score over seeds
def reliability(frame):
    data = frame.assign(ok=frame["status"].eq("ok"), tokens=frame["input_tokens"].fillna(0) + frame["output_tokens"].fillna(0))
    table = data.groupby("system").agg(runs=("ok", "size"), success=("ok", "mean"), median_seconds=("seconds", "median"), mean_tokens=("tokens", "mean"), mean_cost_usd=("cost_usd", "mean"))
    spread = normalized(frame).groupby(["system", "dataset", "level"])["normalized"].std().groupby("system").mean()
    return table.join(spread.rename("seed_std"))

# leak detection of one run folder against the truth of its cell: the leak...
# column dropped or not, and the other non-id columns dropped as leaks
def leak_outcome(run_dir, truth):
    leaks_file = Path(run_dir) / "leaks.json"
    leak = (truth.get("leak") or {}).get("column")
    if not leaks_file.exists():
        return {"found": False, "false_drops": 0, "leak": leak}
    decisions = json.loads(leaks_file.read_text(encoding="utf-8"))
    dropped = {item["column"] for item in decisions if item.get("action") == "drop" and not item.get("evidence", {}).get("structural")}
    return {"found": leak in dropped, "false_drops": len(dropped - {leak}), "leak": leak}

# leak recall, precision and false drops per system from run folders and the...
# truth files of Group D cells
def leak_metrics(frame, cell_truth):
    rows = []
    for _, row in frame[frame["status"].eq("ok") & frame["run_dir"].notna()].iterrows():
        truth = cell_truth(row)
        if not truth or not truth.get("leak"):
            continue
        outcome = leak_outcome(Path(str(row["run_dir"])), truth)
        rows.append({"system": row["system"], "dataset": row["dataset"], **outcome})
    if not rows:
        return pd.DataFrame()
    data = pd.DataFrame(rows)
    table = data.groupby("system").agg(runs=("found", "size"), found=("found", "sum"), false_drops=("false_drops", "sum"))
    table["recall"] = table["found"] / table["runs"]
    table["precision"] = table["found"] / (table["found"] + table["false_drops"]).where(lambda value: value > 0, np.nan)
    return table
