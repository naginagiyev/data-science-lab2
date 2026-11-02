import sys
import json
import pandas as pd
from benchmark import analysis, figures
from benchmark.resultsdoc import markdown
from paths import benchmark_outputs_dir, docs_dir, results_dir

GRIDS = ["agentic-main", "agentic-leaks", "agentic-natural"]
ABLATIONS = "agentic-ablations"
AGENTIC = ["aide", "mlzero"]
SYSTEMS = ["dummy", "lightgbm", "skrub", "flaml", "autogluon", "generic", "rules", "full", "full+ag", "full-nogate", "freecode", "aide", "mlzero"]
NATURAL_LEAKS = ["bank-additional", "titanic", "student-performance", "hotels", "airbnb-austin"]
REPORT = docs_dir / "agentic.md"

# the score table of one grid: mean normalized score and average rank per...
# system and level, in a fixed system order
def score_tables(frame):
    order = [name for name in SYSTEMS if name in set(frame["system"])]
    return analysis.mean_normalized(frame).reindex(order), analysis.average_ranks(frame).reindex(order)

# successes, median seconds, LLM calls, tokens and dollars per system over all...
# grids (part 3 of next step 2)
def cost_table(frame):
    data = frame.assign(ok=frame["status"].eq("ok"))
    table = data.groupby("system").agg(
        runs=("ok", "size"), success=("ok", "mean"), median_seconds=("seconds", "median"), mean_calls=("llm_calls", "mean"),
        mean_input_tokens=("input_tokens", "mean"), mean_output_tokens=("output_tokens", "mean"),
        mean_cost_usd=("cost_usd", "mean"), total_cost_usd=("cost_usd", "sum"),
    )
    return table.reindex([name for name in SYSTEMS if name in table.index])

# the bootstrap intervals and Holm-corrected Wilcoxon tests of one system against...
# the baselines on the given frame and levels (H6, H8)
def differences(frame, levels, label, baselines, system="full"):
    present = [baseline for baseline in baselines if baseline in set(frame["system"])]
    tests = analysis.wilcoxon_table(frame, system, present, levels).set_index("baseline")
    rows = []
    for baseline in present:
        value = analysis.bootstrap_difference(frame, system, baseline, levels)
        test = tests.loc[baseline]
        rows.append({"cells": label, "comparison": f"{system} vs {baseline}", "mean": value["mean"], "low": value["low"], "high": value["high"], "datasets": value["datasets"], "wins": test["wins"], "losses": test["losses"], "p_holm": test["p_holm"]})
    return pd.DataFrame(rows).round(3)

# Friedman p-value and Nemenyi critical difference per level of one grid
def rank_tests(frame):
    rows = [analysis.friedman_nemenyi(frame, level) for level in sorted(frame["level"].unique())]
    return pd.DataFrame([{"level": row["level"], "datasets": row["datasets"], "systems": len(row["systems"]), "p_value": row["p_value"], "critical_difference": row["cd"]} for row in rows])

# the ablation variants next to the agentic-main rows of the same level 3 cells,...
# so that one best and one dummy hold for all systems: normalized per dataset
def ablation_table(frame):
    cells = frame[frame["grid"] == ABLATIONS]
    main = frame[(frame["grid"] == "agentic-main") & (frame["level"] == 3) & frame["dataset"].isin(set(cells["dataset"]))]
    table = analysis.normalized(pd.concat([main, cells], ignore_index=True)).pivot_table(index="system", columns="dataset", values="normalized")
    table["mean"] = table.mean(axis=1)
    return table.reindex([name for name in SYSTEMS if name in table.index])

# the normalized score per dataset and system of one grid and level
def dataset_table(frame, level):
    data = analysis.normalized(frame[frame["level"] == level])
    table = data.pivot_table(index="dataset", columns="system", values="normalized")
    return table[[name for name in SYSTEMS if name in table.columns]]

# the leak recall of the systems with a run folder; run_dir holds the Linux path...
# of the WSL run, the copied folder is under benchmark_outputs_dir
def leak_detection(frame):
    local = frame["run_dir"].map(lambda text: str(benchmark_outputs_dir / text.split("outputs/", 1)[1]) if isinstance(text, str) and "outputs/" in text else None)
    return analysis.leak_metrics(frame.assign(run_dir=local), figures.cell_truth)

# the cause of one failed agentic run, read from the tool files kept next to...
# its predictions (stderr of MLZero's scripts, AIDE's journal)
def failure_cause(row):
    cell = f"{row['system']}-{row['dataset']}-l{row['level']}-s{row['seed']}-b{int(row['budget_s'])}"
    folder = benchmark_outputs_dir / row["grid"] / cell / "tool" / "output"
    if row["system"] == "mlzero":
        errors = " ".join(path.read_text(encoding="utf-8", errors="replace") for path in (folder / "code").glob("*stderr_*.txt")) if folder.exists() else ""
        if "--prerelease" in errors or "No module named" in errors:
            return "its environment script failed (pip install --prerelease or packages not installed)"
        return "time limit reached before any model code ran" if folder.exists() else "no tool files"
    journal = folder / "journal.json"
    if not journal.exists():
        return "no journal"
    nodes = json.loads(journal.read_text(encoding="utf-8")).get("nodes", [])
    crashed = [node for node in nodes if node.get("is_buggy")]
    if len(crashed) == len(nodes):
        kinds = pd.Series([node.get("exc_type") or "error" for node in crashed]).value_counts()
        return f"every script crashed (most often {kinds.index[0]})" if len(kinds) else "no script finished"
    return "scripts finished but wrote no submission (scores given without output)"

# the failed agentic runs, counted by cause
def failures(frame):
    failed = frame[frame["system"].isin(AGENTIC) & ~frame["status"].eq("ok")]
    if failed.empty:
        return pd.DataFrame()
    return failed.assign(cause=failed.apply(failure_cause, axis=1)).groupby(["system", "cause"]).size().rename("runs").reset_index()

# the naive (leak kept) and deployment (leak empty) scores of the Group D and...
# natural leak cells, for the systems that predict both files
def leak_scores(frame):
    data = frame[(frame["grid"] == "agentic-leaks") | frame["dataset"].isin(NATURAL_LEAKS)].copy()
    data["naive_score"] = data["secondary"].map(lambda text: json.loads(text).get("naive_score") if isinstance(text, str) and text.startswith("{") else None)
    table = data.pivot_table(index=["dataset", "metric"], columns="system", values=["score", "naive_score"], aggfunc="first")
    table.columns = [f"{system} {'naive' if kind == 'naive_score' else 'deploy'}" for kind, system in table.columns]
    table.index = [f"{dataset} ({metric})" for dataset, metric in table.index]
    order = [f"{system} {kind}" for system in SYSTEMS if system != "dummy" for kind in ("deploy", "naive")]
    return table[[column for column in order if column in table.columns]].astype(float).round(3)

# writes docs/agentic.md from the agentic grids in runs.csv
def build():
    frame = analysis.load_results(GRIDS + [ABLATIONS])
    if frame.empty:
        raise SystemExit("no agentic results in runs.csv")
    lines = [
        "# Agentic comparison and natural data (E11, E12)", "",
        "Generated by `uv run python -m benchmark.agenticreport` from `benchmark/results/runs.csv`. Protocol: "
        "`docs/protocol.md` section 7 (written before the runs). One seed, 300 s per run, `gpt-4.1-mini`, every system on "
        "native Linux ARM (WSL). Normalized score: 1 = best system in the cell, 0 = dummy; failed runs count as the dummy.", "",
    ]
    for grid in GRIDS:
        part = frame[frame["grid"] == grid]
        if part.empty:
            continue
        normalized_table, ranks = score_tables(part)
        lines += [f"## `{grid}`", "", "Mean normalized score:", "", markdown(normalized_table), "", "Average rank (1 = best):", "", markdown(ranks), ""]
    main, natural = frame[frame["grid"] == "agentic-main"], frame[frame["grid"] == "agentic-natural"]
    rivals = AGENTIC + ["autogluon", "flaml", "skrub", "lightgbm", "generic", "rules"]
    tests = [differences(main, [2, 3], "agentic-main levels 2, 3", rivals), differences(main, [3], "agentic-main level 3", rivals), differences(main, [0], "agentic-main level 0", rivals),
             differences(main, [2, 3], "agentic-main levels 2, 3", ["autogluon"], "full+ag"), differences(natural, [0], "agentic-natural", rivals)]
    tests = [table for table in tests if not table.empty]
    if tests:
        lines += ["## Differences (bootstrap 95% interval over datasets, Wilcoxon with Holm, normalized score)", "", markdown(pd.concat(tests, ignore_index=True), index=False), ""]
    lines += ["## Rank tests on `agentic-main` (Friedman, Nemenyi)", "", markdown(rank_tests(main), index=False), ""]
    lines += ["## `agentic-main`: difference full minus AutoGluon per dataset, levels 2 and 3", "",
              markdown(analysis.paired(main, "full", "autogluon", [2, 3]).assign(difference=lambda table: table["full"] - table["autogluon"])["difference"].groupby("dataset").mean().sort_values().to_frame().round(3)), ""]
    if (frame["grid"] == ABLATIONS).any():
        lines += ["## Ablations: level 3, 4 datasets, normalized score", "", markdown(ablation_table(frame)), ""]
    lines += ["## `agentic-natural`: normalized score per dataset", "", "A dataset where no system beats the dummy has no normalized score and is not listed.", "", markdown(dataset_table(natural, 0)), ""]
    lines += ["## Cost, time and tokens per run (all grids)", "", markdown(cost_table(frame)), ""]
    failed = failures(frame)
    if not failed.empty:
        lines += ["## Why the agentic runs failed", "", markdown(failed, index=False), ""]
    leaks = frame[frame["grid"] == "agentic-leaks"]
    lines += ["## Planted leaks (`agentic-leaks`)", "", "Mean normalized score with the leak in the test file (naive) and with the leak empty (deployment):", "",
              markdown(figures.leak_summary(leaks).reindex([name for name in SYSTEMS if name in set(leaks["system"])]).round(3)), "", "Leak columns dropped by the variants:", "", markdown(leak_detection(leaks)), ""]
    lines += ["## Leak cells: deployment score (leak empty) and naive score (leak kept)", "",
              "The agentic tools predict only the deployment file.", "", markdown(leak_scores(frame)), ""]
    REPORT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    cost_table(frame).to_csv(results_dir / "agentic-cost.csv", lineterminator="\n")
    return REPORT

# command line: python -m benchmark.agenticreport
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"written {build()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
