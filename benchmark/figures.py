import sys
import json
import matplotlib
import numpy as np
import pandas as pd
from benchmark import analysis
import matplotlib.pyplot as plt
from benchmark.materialize import cell_dir
from paths import figures_dir, tables_file

SYSTEMS = ["lightgbm", "skrub", "flaml", "autogluon", "generic", "rules", "full", "full+ag", "full-nogate", "freecode"]
COLORS = dict(zip(SYSTEMS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#52514e", "#8f6b00"]))
MARKERS = dict(zip(SYSTEMS, ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASELINES = ["autogluon", "skrub", "lightgbm", "flaml", "generic"]
TABLES = {}
matplotlib.use("Agg")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

# saves a figure as PDF and PNG
def save(figure, name):
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for suffix in ["pdf", "png"]:
        figure.savefig(figures_dir / f"{name}.{suffix}", dpi=200)
    plt.close(figure)

# keeps a table as Markdown until write_tables puts every table into one file
def table(frame, name, index=True):
    TABLES[name] = frame.to_markdown(index=index, floatfmt=".4f")
    return frame

# writes every kept table into tables.md, one section per table
def write_tables():
    sections = [f"## {name}\n\n{text}\n" for name, text in TABLES.items()]
    tables_file.write_text("# Pilot tables\n\n" + "\n".join(sections), encoding="utf-8", newline="\n")

# the bootstrap 95% interval of a system's mean normalized score at one level...
# (resampling datasets)
def level_interval(data, system, level, resamples=10000, seed=0):
    values = data[(data["system"] == system) & (data["level"] == level)].groupby("dataset")["normalized"].mean().to_numpy()
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    draws = np.random.default_rng(seed).choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return values.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)

# figure 1: mean normalized score per mess level and system with 95% intervals
def score_vs_level(frame):
    data = analysis.normalized(frame)
    levels = sorted(data["level"].unique())
    figure, axis = plt.subplots(figsize=(7.4, 4.0))
    shown = [name for name in SYSTEMS if name in set(data["system"])]
    for position, system in enumerate(shown):
        points = [level_interval(data, system, level) for level in levels]
        means = [point[0] for point in points]
        offset = (position - (len(shown) - 1) / 2) * 0.04
        axis.errorbar([level + offset for level in levels], means, yerr=[[mean - low for mean, low, _ in points], [high - mean for mean, _, high in points]], color=COLORS[system], marker=MARKERS[system], markersize=6, linewidth=2, capsize=3, label=system)
    axis.set_xticks(levels)
    axis.set_xlabel("mess level")
    axis.set_ylabel("mean normalized score (1 = best in cell, 0 = dummy)")
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(figure, "score-vs-level")

# figure 2: critical-difference diagram (average ranks and the Nemenyi CD) at...
# one level
def cd_diagram(frame, level):
    result = analysis.friedman_nemenyi(frame[frame["system"] != "dummy"], level)
    if not result["ranks"]:
        return result
    ranks = result["ranks"]
    figure, axis = plt.subplots(figsize=(6.4, 0.35 * len(ranks) + 1.4))
    for position, (system, rank) in enumerate(sorted(ranks.items(), key=lambda item: item[1])):
        axis.plot([rank], [position], marker=MARKERS.get(system, "o"), color=COLORS.get(system, MUTED), markersize=7)
        axis.text(rank + 0.08, position, f"{system} ({rank:.2f})", va="center", color=INK, fontsize=8)
    best = min(ranks.values())
    axis.plot([best, best + result["cd"]], [-1, -1], color=INK, linewidth=2)
    axis.text(best + result["cd"] / 2, -1.4, f"CD = {result['cd']:.2f} (Nemenyi, alpha 0.05)", ha="center", va="top", color=MUTED, fontsize=8)
    axis.set_yticks([])
    axis.invert_yaxis()
    axis.set_xlabel("average rank (lower is better)")
    p_text = f"Friedman p = {result['p_value']:.3g}" if result["p_value"] is not None else "Friedman test needs 3 systems"
    axis.set_title(f"Level {level}: {result['datasets']} datasets; {p_text}", fontsize=9, color=INK, loc="left")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    save(figure, f"cd-level{level}")
    return result

# leak experiment: mean naive score (leak present at test time) and...
# deployment score (leak masked) per system, normalized with the deployment...
# cell's dummy and best
def leak_summary(frame):
    data = analysis.normalized(frame)
    naive = frame["secondary"].map(lambda text: json.loads(text).get("naive_score") if isinstance(text, str) and text.startswith("{") else None)
    data = data.join(naive.rename("naive"), how="left")
    if data["naive"].isna().all():
        return pd.DataFrame()
    sign = np.where(data["metric"].isin(analysis.LOWER_IS_BETTER), -1.0, 1.0)
    best = data.groupby(analysis.CELL)["oriented"].transform("max")
    data["naive_normalized"] = (data["naive"] * sign - data["dummy_oriented"]) / (best - data["dummy_oriented"])
    return data.groupby("system")[["naive_normalized", "normalized"]].mean().rename(columns={"normalized": "deployment_normalized"})

# figure 3: the leak summary as naive vs deployment points per system
def leak_scores(frame):
    summary = leak_summary(frame)
    if summary.empty:
        return summary
    figure, axis = plt.subplots(figsize=(6.4, 0.45 * len(summary) + 1.2))
    for position, (system, row) in enumerate(summary.iterrows()):
        axis.plot([row["deployment_normalized"], row["naive_normalized"]], [position, position], color=GRID, linewidth=3)
        axis.plot(row["naive_normalized"], position, "o", color=MUTED, markersize=7, label="naive (leak in the test file)" if position == 0 else None)
        axis.plot(row["deployment_normalized"], position, "o", color="#2a78d6", markersize=8, label="deployment (leak masked)" if position == 0 else None)
    axis.set_yticks(range(len(summary)), summary.index)
    axis.set_xlabel("mean normalized score (1 = best deployment score in the cell, 0 = dummy)")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(figure, "leaks")
    return table(summary, "leaks-naive-vs-deployment")

# figure 4: ablations at level 3 on the ablation datasets (one hue, sorted)
def ablations(frame, datasets):
    data = analysis.normalized(frame[(frame["level"] == 3) & frame["dataset"].isin(datasets)])
    means = data.groupby(["system", "dataset"])["normalized"].mean().groupby("system").mean().sort_values()
    means = means[[name for name in means.index if name != "dummy"]]
    if means.empty:
        return means
    figure, axis = plt.subplots(figsize=(6.4, 0.35 * len(means) + 1.0))
    axis.barh(range(len(means)), means.to_numpy(), color="#2a78d6", height=0.6)
    for position, value in enumerate(means.to_numpy()):
        axis.text(value + 0.01, position, f"{value:.3f}", va="center", color=INK, fontsize=8)
    axis.set_yticks(range(len(means)), means.index)
    axis.set_xlabel(f"mean normalized score, level 3, {len(datasets)} datasets")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    save(figure, "ablations")
    return table(means.to_frame("normalized"), "ablations")

# figure 5: cost vs score per system (dollars per run and mean normalized score)
def cost_vs_score(frame):
    data = analysis.normalized(frame)
    summary = data.groupby("system").agg(score=("normalized", "mean"), cost=("cost_usd", "mean"), seconds=("seconds", "median"))
    summary = summary[summary.index != "dummy"]
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for system, row in summary.iterrows():
        axis.plot(row["cost"], row["score"], marker=MARKERS.get(system, "o"), color=COLORS.get(system, MUTED), markersize=8, linestyle="none")
        axis.annotate(system, (row["cost"], row["score"]), textcoords="offset points", xytext=(6, 3), fontsize=8, color=INK)
    axis.set_xlabel("mean LLM cost per run (USD)")
    axis.set_ylabel("mean normalized score")
    axis.grid(color=GRID, linewidth=0.8)
    save(figure, "cost-vs-score")
    return table(summary, "cost-vs-score")

# the truth.json of a result row's cell
def cell_truth(row):
    path = cell_dir(row["group"], row["dataset"], int(row["level"]), int(row["seed"]), row["mess_types"]) / "truth.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

# regenerates every pilot figure and table from the results files
def build():
    main = analysis.load_results(["pilot-main"])
    outputs = {}
    if not main.empty:
        score_vs_level(main)
        outputs["cd"] = [cd_diagram(main, level) for level in sorted(main["level"].unique())]
        table(analysis.mean_normalized(main), "mean-normalized")
        table(analysis.average_ranks(main), "average-ranks")
        tests = [analysis.wilcoxon_table(main, "full", [name for name in BASELINES if name in set(main["system"])], levels) for levels in [[0], [2], [3], [2, 3]]]
        if "full+ag" in set(main["system"]):
            tests.append(analysis.wilcoxon_table(main, "full+ag", ["autogluon"], [2, 3]))
        table(pd.concat(tests, ignore_index=True), "wilcoxon", index=False)
        table(main.pivot_table(index=["dataset", "level"], columns="system", values="score", aggfunc="first"), "per-dataset-scores")
        table(analysis.reliability(main), "reliability")
        cost_vs_score(main)
    ablation = analysis.load_results(["pilot-ablations"])
    if not ablation.empty:
        names = sorted(ablation["dataset"].unique())
        ablations(pd.concat([main, ablation], ignore_index=True), names)
    leaks = analysis.load_results(["pilot-leaks"])
    if not leaks.empty:
        leak_scores(leaks)
        table(analysis.leak_metrics(leaks, cell_truth), "leak-detection")
    write_tables()
    return outputs

# command line: python -m benchmark.figures
def main():
    build()
    print(f"Figures in {figures_dir}, tables in {tables_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
