import sys
import pandas as pd
from benchmark import analysis, figures
from paths import docs_dir, results_dir

BASELINES = ["autogluon", "skrub", "lightgbm", "flaml", "generic"]

# a frame as a Markdown table with 3 decimals ("n/a" for missing values)
def markdown(frame, index=True):
    if frame.empty:
        return "(no results)"
    return frame.astype(object).where(pd.notna(frame), "n/a").to_markdown(index=index, floatfmt=".3f")

# the RQ1 and RQ2 section from pilot-main (and pilot-seed for the reliability)
def main_section(main):
    lines = ["## RQ1 and RQ2: score vs mess level (`pilot-main`)", "", "Mean normalized score (1 = best system in the cell, 0 = dummy; mean over datasets):", ""]
    lines += [markdown(analysis.mean_normalized(main)), "", "Average rank (1 = best):", "", markdown(analysis.average_ranks(main)), ""]
    for level in sorted(main["level"].unique()):
        result = analysis.friedman_nemenyi(main[main["system"] != "dummy"], level)
        p_text = f"{result['p_value']:.3g}" if result["p_value"] is not None else "n/a"
        lines.append(f"- Level {level}: Friedman p = {p_text} over {result['datasets']} datasets; Nemenyi critical difference {result['cd']:.2f} ranks (figure `cd-level{level}`).")
    tests = [analysis.wilcoxon_table(main, "full", [name for name in BASELINES if name in set(main["system"])], levels) for levels in [[0], [3], [2, 3]]]
    if "full+ag" in set(main["system"]):
        tests.append(analysis.wilcoxon_table(main, "full+ag", ["autogluon"], [2, 3]))
    lines += ["", "Wilcoxon signed-rank tests (pairs = dataset x level cells; Holm over the baselines of one row group):", "", markdown(pd.concat(tests, ignore_index=True)[["system", "baseline", "levels", "pairs", "mean_difference", "wins", "ties", "losses", "p_value", "p_holm"]], index=False), ""]
    rows = []
    for system, baseline in [("full", "autogluon"), ("full", "generic"), ("full+ag", "autogluon")]:
        for levels in [[0], [2, 3]]:
            value = analysis.bootstrap_difference(main, system, baseline, levels)
            rows.append({"comparison": f"{system} vs {baseline}", "levels": ",".join(map(str, levels)), "mean": value["mean"], "low": value["low"], "high": value["high"], "datasets": value["datasets"]})
    lines += ["Bootstrap 95% intervals of the mean normalized-score difference (resampling datasets):", "", markdown(pd.DataFrame(rows), index=False), ""]
    return lines

# the RQ3 section from pilot-leaks
def leak_section(leaks):
    lines = ["## RQ3: leakage (`pilot-leaks`, Group D)", ""]
    if leaks.empty:
        return lines + ["(no results)", ""]
    detection = analysis.leak_metrics(leaks, figures.cell_truth)
    lines += ["Leak detection (a leak column dropped by the leak guard; false drops = other non-id columns dropped as leaks):", "", markdown(detection), ""]
    summary = figures.leak_summary(leaks)
    if not summary.empty:
        lines += ["Naive score (leak values present in the test file) vs deployment score (leak masked), mean normalized:", "", markdown(summary), ""]
    return lines

# the RQ4 section: reliability and cost over all pilot grids
def reliability_section(frames):
    data = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    lines = ["## RQ4: reliability and cost (all pilot grids)", ""]
    return lines + [markdown(analysis.reliability(data)), ""] if not data.empty else lines + ["(no results)", ""]

# the RQ5 section from pilot-ablations (with generic, rules and full of...
# pilot-main on the same datasets at level 3)
def ablation_section(main, ablation):
    lines = ["## RQ5: ablations (`pilot-ablations`, level 3)", ""]
    if ablation.empty:
        return lines + ["(no results)", ""]
    datasets = sorted(ablation["dataset"].unique())
    data = analysis.normalized(pd.concat([main, ablation], ignore_index=True).query("level == 3 and dataset in @datasets"))
    table = data.groupby(["system", "dataset"])["normalized"].mean().unstack("dataset")
    table["mean"] = table.mean(axis=1)
    return lines + [f"Normalized score per dataset ({', '.join(datasets)}):", "", markdown(table.sort_values("mean", ascending=False)), ""]

# the E6 section: repair accuracy per mess type and gain over generic
def types_section(types):
    lines = ["## E6: mess types (`pilot-types` and the cleaning-only checks)", ""]
    repairs_file = results_dir / "repairs.csv"
    if repairs_file.exists():
        repairs = pd.read_csv(repairs_file)
        repairs = repairs[repairs["cells"] > 0]
        summary = repairs.groupby(["variant", "mess"]).apply(lambda part: pd.Series({"datasets": part["dataset"].nunique(), "cells": part["cells"].sum(), "accuracy": part["repaired"].sum() / part["cells"].sum(), "false_nan_rate": part["false_nan_rate"].mean()}), include_groups=False)
        lines += ["Repair accuracy of the cleaning stage (share of corrupted training cells restored):", "", markdown(summary), ""]
    if not types.empty:
        data = analysis.normalized(types)
        table = data.groupby(["mess_types", "system"])["normalized"].mean().unstack("system")
        lines += ["Mean normalized score per single mess type (level 3):", "", markdown(table), ""]
    frame = analysis.load_results(["reading"])
    if not frame.empty:
        table = frame.groupby("system").agg(runs=("status", "size"), success=("status", lambda values: (values == "ok").mean()))
        lines += ["E6b reading robustness (file layout: BOM, title rows, footer, `;`, decimal commas, cp1252):", "", markdown(table), ""]
    return lines

# the E7 and E10 sections
def other_sections(natural):
    lines = ["## E7: framer accuracy (40 OpenML datasets)", ""]
    framer_file = results_dir / "framer.csv"
    if framer_file.exists():
        framer = pd.read_csv(framer_file)
        counts = framer[["rules_target_ok", "rules_task_ok", "llm_target_ok", "llm_task_ok"]].sum().to_frame(f"correct of {len(framer)}")
        lines += [markdown(counts), ""]
    lines += ["## E10: natural mess (`pilot-natural`, Group C)", ""]
    if natural.empty:
        return lines + ["(no results)", ""]
    return lines + [markdown(natural.pivot_table(index="dataset", columns="system", values="score", aggfunc="first")), ""]

# writes docs/results.md from the pilot result files
def build(out=None):
    load = lambda name: analysis.load_results([name])
    main, seed, leaks, ablation, types, natural, budget = (load(name) for name in ["pilot-main", "pilot-seed", "pilot-leaks", "pilot-ablations", "pilot-types", "pilot-natural", "pilot-budget"])
    lines = [
        "# Pilot results", "",
        "Generated by `python -m benchmark.resultsdoc` from `benchmark/results/`. **Pilot results on a small sample** (8 Group A datasets, one seed for the LLM systems, 120 s per run). They are not the final results of the study; `docs/projection.md` estimates what the full benchmark would show.", "",
        f"Runs: {sum(len(frame) for frame in [main, seed, leaks, ablation, types, natural, budget])} in the pilot grids. Figures: `benchmark/results/figures/`; tables: `benchmark/results/tables.md`.", "",
    ]
    if not main.empty:
        lines += main_section(main)
    lines += leak_section(leaks)
    lines += reliability_section([main, seed, leaks, ablation, types, natural, budget])
    lines += ablation_section(main, ablation)
    lines += types_section(types)
    lines += other_sections(natural)
    if not budget.empty:
        lines += ["## E9: budget check (`pilot-budget`, 300 s, level 3)", "", markdown(budget.pivot_table(index="dataset", columns="system", values="score", aggfunc="first")), ""]
    text = "\n".join(lines).rstrip("\n") + "\n"
    (out or docs_dir / "results.md").write_text(text, encoding="utf-8", newline="\n")
    return text

# command line: python -m benchmark.resultsdoc (after python -m benchmark.figures)
def main():
    build()
    print(f"Written {docs_dir / 'results.md'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
