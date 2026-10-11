import json
import time
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from edabench.injector import inject
from paths import edabench_results_dir, edabench_runs_dir
from edabench.baselines import cleanlab_recall, cleanlab_rows, legacy_issues, ydata_issues
from edabench.datasets import clean_datasets, load_openml, local_dataset_path, local_datasets
from edabench.evaluate import accepted_codes, categories, issue_labels, load_labelled, matches, run_ledger, score_run, summarize

# list prices per million tokens, the LLM budget of the experiments and the...
# file that tracks the spend
prices = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40)}
budget = 3.0
spend_file = edabench_results_dir / "spend.json"
    
# reads the recorded LLM spend
def read_spend():
    if spend_file.exists():
        return json.loads(spend_file.read_text(encoding="utf-8"))
    return {"total": 0.0, "runs": []}

# adds the cost of one agent run to spend.json
def add_spend(label, model_id, tokens):
    spend = read_spend()
    price_in, price_out = prices[model_id]
    cost = tokens["input"] / 1e6 * price_in + tokens["output"] / 1e6 * price_out
    spend["total"] = round(spend["total"] + cost, 4)
    spend["runs"].append({"label": label, "model": model_id, "tokens": tokens, "cost": round(cost, 4)})
    edabench_results_dir.mkdir(parents=True, exist_ok=True)
    spend_file.write_text(json.dumps(spend, indent=2) + "\n", encoding="utf-8")
    return cost

# checks if the next run fits in the budget (estimate: twice the most...
# expensive earlier run)
def can_spend(model_id):
    spend = read_spend()
    same = [run["cost"] for run in spend["runs"] if run["model"] == model_id and run["cost"] > 0]
    estimate = 2 * max(same) if same else 0.15
    return spend["total"] + estimate <= budget, spend["total"], estimate

# scores the issues of a baseline tool, which have no row sets
def score_simple(issues, baseline, ground_truth):
    new_issues = [issue for issue in issues if issue.fingerprint not in baseline]
    records = []
    for item in ground_truth:
        related = [issue for issue in issues if matches(issue, item)]
        strict = any(issue.code == item["code"] for issue in related)
        records.append({"kind": item["kind"], "code": item["code"], "found": bool(related), "strict": strict})
    true_positive = [issue for issue in new_issues if any(matches(issue, item) for item in ground_truth)]
    false_positive = [{"code": issue.code, "columns": issue.columns} for issue in new_issues if issue not in true_positive]
    return records, len(true_positive), false_positive

# compares edatool with ydata-profiling, the old eda.py and cleanlab on...
# medium strength
def run_baselines(strength="medium"):
    tools = {"edatool": None, "ydata-profiling": ydata_issues, "legacy eda.py": None, "cleanlab": None}
    results = {name: [] for name in tools}
    for entry in clean_datasets:
        raw, target, task = load_openml(entry)
        dirty, ground_truth = inject(raw, target, strength)
        clean_context, _ = run_ledger(raw, target, task)
        context, _ = run_ledger(dirty, target, task)
        records, true_positive, false_positive = score_run(context, {issue.fingerprint for issue in clean_context.ledger.issue_list()}, ground_truth)
        results["edatool"].append({"records": records, "true_positive": true_positive, "false_positive": false_positive})
        for name, function in [("ydata-profiling", lambda frame: ydata_issues(frame, target, task)), ("legacy eda.py", lambda frame: legacy_issues(frame, target, task, entry["name"]))]:
            start = time.perf_counter()
            baseline = {issue.fingerprint for issue in function(raw)}
            records, true_positive, false_positive = score_simple(function(dirty), baseline, ground_truth)
            results[name].append({"records": records, "true_positive": true_positive, "false_positive": false_positive, "seconds": time.perf_counter() - start})
        records = cleanlab_recall(cleanlab_rows(dirty, target, task), ground_truth)
        results["cleanlab"].append({"records": records, "true_positive": 0, "false_positive": []})
        print(f"baselines done: {entry['name']}")
    kinds = list(accepted_codes)
    table = []
    for kind in kinds:
        row = [kind]
        for name in tools:
            hits = [record["found"] for result in results[name] for record in result["records"] if record["kind"] == kind]
            row.append(np.mean(hits) if hits else None)
        table.append(row)
    overall = ["all injected"]
    precision = ["precision (column-level)"]
    for name in tools:
        hits = [record["found"] for result in results[name] for record in result["records"]]
        overall.append(np.mean(hits))
        true_positive = sum(result["true_positive"] for result in results[name])
        false_positive = sum(len(result["false_positive"]) for result in results[name])
        precision.append(true_positive / (true_positive + false_positive) if name != "cleanlab" and true_positive + false_positive else None)
    table += [overall, precision]
    lines = [
        f"# Baselines on the injection benchmark ({strength} strength, {len(clean_datasets)} datasets)",
        "",
        "Recall per injected issue kind. cleanlab works on rows: an injected row issue counts as found when at least half of its rows are flagged with the matching issue type (outlier, near_duplicate, label, null); its column-level precision is not defined.",
        "",
        tabulate(table, headers=["Injected kind"] + list(tools), tablefmt="pipe", floatfmt=".3f", missingval="-"),
    ]
    write_result("baselines", lines, results)

# writes an experiment table (md) and its data (json) into benchmark/results
def write_result(name, lines, data):
    edabench_results_dir.mkdir(parents=True, exist_ok=True)
    (edabench_results_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (edabench_results_dir / f"{name}.json").write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    print("\n".join(lines))

# agent settings of the experiments: model, gate flag and cache folder
def agent_settings(model_id, use_gate):
    cache = edabench_runs_dir / "agentcache-bench"
    return {"agent": {"model_id": model_id, "use_gate": use_gate, "cache_dir": str(cache)}}

# runs the agent on one dataset if the budget allows and records the cost
def run_agent_dataset(raw, target, task, model_id, use_gate, label):
    from edatool.agent.session import review
    allowed, spent, estimate = can_spend(model_id)
    if not allowed:
        print(f"skipped {label}: spent ${spent:.2f} + estimate ${estimate:.2f} would pass the ${budget:.2f} cap")
        return None
    context, _ = run_ledger(raw, target, task, agent_settings(model_id, use_gate))
    start = time.perf_counter()
    summary, _ = review(context, context.settings)
    seconds = time.perf_counter() - start
    fresh = {
        "input": sum(session["tokens"]["input"] for session in summary["sessions"] if not session["cached"]),
        "output": sum(session["tokens"]["output"] for session in summary["sessions"] if not session["cached"]),
    }
    cost = add_spend(label, model_id, fresh)
    return context, summary, seconds, cost

# compares the agent with and without the done gate and with two models
def run_ablations(strength="medium"):
    settings_list = [("gpt-4.1-mini", True), ("gpt-4.1-nano", True), ("gpt-4.1-mini", False)]
    rows = []
    data = {}
    deterministic = []
    for entry in clean_datasets:
        raw, target, task = load_openml(entry)
        dirty, ground_truth = inject(raw, target, strength)
        clean_context, _ = run_ledger(raw, target, task)
        baseline = {issue.fingerprint for issue in clean_context.ledger.issue_list()}
        context, _ = run_ledger(dirty, target, task)
        records, true_positive, false_positive = score_run(context, baseline, ground_truth)
        deterministic.append({"records": records, "true_positive": true_positive, "false_positive": false_positive})
        for model_id, use_gate in settings_list:
            label = f"ablation {entry['name']} {model_id} gate={use_gate}"
            outcome = run_agent_dataset(dirty, target, task, model_id, use_gate, label)
            if outcome is None:
                continue
            context, summary, seconds, cost = outcome
            issues = [issue for issue in context.ledger.issue_list() if issue.status != "dismissed"]
            dismissed = [issue for issue in context.ledger.issue_list() if issue.status == "dismissed"]
            records, true_positive, false_positive = score_run(context, baseline, ground_truth, issues)
            false_dismissals = [issue.code for issue in dismissed if any(matches(issue, item) for item in ground_truth)]
            reviewed = [issue for issue in context.ledger.issue_list() if issue.needs_review]
            key = f"{model_id} gate={use_gate}"
            data.setdefault(key, []).append(
                {
                    "dataset": entry["name"],
                    "records": records,
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "dismissed": len(dismissed),
                    "false_dismissals": false_dismissals,
                    "reviewed": len(reviewed),
                    "closed": sum(issue.status != "open" for issue in reviewed),
                    "complete": summary["complete"],
                    "tokens": summary["tokens"],
                    "seconds": seconds,
                    "cost": cost,
                    "token_cost": summary["tokens"]["input"] / 1e6 * prices[model_id][0] + summary["tokens"]["output"] / 1e6 * prices[model_id][1],
                }
            )
            print(f"{label}: complete={summary['complete']} dismissed={len(dismissed)} false_dismissals={len(false_dismissals)} cost=${cost:.3f}")
    data["deterministic"] = deterministic
    headers = ["Setting", "Runs", "Recall", "Precision", "Review items closed", "Dismissed", "False dismissals", "Complete runs", "Tokens/run", "Cost/run ($)"]
    for key, runs in data.items():
        _, overall = summarize(runs)
        if key == "deterministic":
            rows.append([key, len(runs), overall["recall"], overall["precision"], None, None, None, None, None, None])
            continue
        reviewed = sum(run["reviewed"] for run in runs)
        rows.append(
            [
                key,
                len(runs),
                overall["recall"],
                overall["precision"],
                sum(run["closed"] for run in runs) / reviewed if reviewed else None,
                sum(run["dismissed"] for run in runs),
                sum(len(run["false_dismissals"]) for run in runs),
                sum(run["complete"] for run in runs),
                np.mean([run["tokens"]["input"] + run["tokens"]["output"] for run in runs]),
                np.mean([run["token_cost"] for run in runs]),
            ]
        )
    lines = [
        f"# Agent ablations ({strength} strength)",
        "",
        "Precision counts dismissed issues as removed. A false dismissal is a dismissed issue that matches an injected issue. 'gate=False' is free stopping: the done gate is not enforced. Costs use the list prices per million tokens: " + ", ".join(f"{name} ${p[0]} in / ${p[1]} out" for name, p in prices.items()) + ". Cost/run is computed from the tokens of the review (also when the sessions were read from the cache).",
        "",
        tabulate(rows, headers=headers, tablefmt="pipe", floatfmt=".3f", missingval="-"),
        "",
        f"Total LLM spend recorded for M9 so far: ${read_spend()['total']:.2f} (cap ${budget:.2f}).",
    ]
    write_result("ablations", lines, data)

# runs the agent on the dirty real datasets and compares the result with the...
# labels
def run_real_agent():
    lines = ["# Real dirty datasets: deterministic vs agent review (gpt-4.1-mini, gate on)", ""]
    data = []
    for label in issue_labels(include_drafts=True):
        if label["source"] != "openml":
            continue
        raw, target, task = load_labelled(label)
        outcome = run_agent_dataset(raw, target, task, "gpt-4.1-mini", True, f"real {label['dataset']}")
        if outcome is None:
            continue
        context, summary, seconds, cost = outcome
        issues = context.ledger.issue_list()
        kept = [issue for issue in issues if issue.status != "dismissed"]
        found_before = found_after = 0
        for item in label["issues"]:
            hit = lambda pool: any(issue.code in item["codes"] and (not item["columns"] or set(issue.columns) & set(item["columns"])) for issue in pool)
            found_before += hit(issues)
            found_after += hit(kept)
        added = [f"{issue.code} {issue.columns}" for issue in issues if issue.source == "agent"]
        dismissed = [f"{issue.id} {issue.code} {issue.columns}: {issue.diagnosis}" for issue in issues if issue.status == "dismissed"]
        data.append({"dataset": label["dataset"], "labels": len(label["issues"]), "found_before": found_before, "found_after": found_after, "issues": len(issues), "kept": len(kept), "added": added, "dismissed": dismissed, "complete": summary["complete"], "tokens": summary["tokens"], "cost": cost})
        lines += [
            f"## {label['dataset']}",
            f"- labelled issues found: {found_before}/{len(label['issues'])} before review, {found_after}/{len(label['issues'])} after review",
            f"- issues: {len(issues)} reported, {len(kept)} kept after review; complete: {summary['complete']}; tokens: {summary['tokens']}; cost ${cost:.3f}",
            f"- added by the agent: {', '.join(added) or 'none'}",
            "- dismissed: " + ("; ".join(dismissed) if dismissed else "none"),
            "",
        ]
    write_result("realagent", lines, data)

# thresholds and values tested in the sensitivity analysis
sensitivity_grid = [
    ("D02", "min_robust_z", [5, 7.5, 10, 15, 20]),
    ("E09", "min_abs_skew", [1, 2, 3, 5]),
    ("C03", "min_gap_iqr", [1, 3, 10, 30]),
    ("B01", "min_parse_share", [0.8, 0.9, 0.95, 0.99]),
]

# changes one threshold at a time and draws recall and precision
def run_sensitivity(strength="medium"):
    prepared = []
    for entry in clean_datasets:
        raw, target, task = load_openml(entry)
        dirty, ground_truth = inject(raw, target, strength)
        prepared.append((raw, dirty, target, task, ground_truth))
    rows = []
    figure, axes = plt.subplots(1, len(sensitivity_grid), figsize=(4 * len(sensitivity_grid), 3.6))
    for axis, (code, name, values) in zip(axes, sensitivity_grid):
        recalls, precisions = [], []
        for value in values:
            settings = {"thresholds": {code: {name: {"value": value}}}}
            runs = []
            for raw, dirty, target, task, ground_truth in prepared:
                clean_context, _ = run_ledger(raw, target, task, settings)
                context, _ = run_ledger(dirty, target, task, settings)
                records, true_positive, false_positive = score_run(context, {issue.fingerprint for issue in clean_context.ledger.issue_list()}, ground_truth)
                runs.append({"records": records, "true_positive": true_positive, "false_positive": false_positive})
            _, overall = summarize(runs)
            recalls.append(overall["recall"])
            precisions.append(overall["precision"])
            rows.append([f"{code}.{name}", value, overall["recall"], overall["precision"], overall["false_positive"]])
            print(f"{code}.{name}={value}: recall {overall['recall']:.3f} precision {overall['precision']:.3f}")
        axis.plot(values, recalls, marker="o", label="recall", color="#2563eb")
        axis.plot(values, precisions, marker="s", label="precision", color="#d97706")
        axis.set_title(f"{code}.{name}")
        axis.set_xlabel("threshold")
        axis.set_ylim(0.6, 1.02)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(edabench_results_dir / "sensitivity.png", dpi=150)
    lines = [
        f"# Threshold sensitivity ({strength} strength, {len(clean_datasets)} datasets, deterministic)",
        "",
        "One threshold changes at a time; all others keep their defaults. Figure: sensitivity.png.",
        "",
        tabulate(rows, headers=["Threshold", "Value", "Recall", "Precision", "False positives"], tablefmt="pipe", floatfmt=".3f"),
    ]
    write_result("sensitivity", lines, rows)

# draws the recall per category and injection strength
def run_figures():
    data = json.loads((edabench_results_dir / "benchmark-deterministic.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(
        [
            {"strength": run["strength"], "category": categories[record["code"][0]], "found": record["found"]}
            for run in data["runs"]
            for record in run["records"]
        ]
    )
    table = frame.groupby(["category", "strength"])["found"].mean().unstack()[["low", "medium", "high"]]
    axis = table.plot(kind="bar", figsize=(7, 3.6), color=["#93c5fd", "#3b82f6", "#1e3a8a"], rot=0)
    axis.set_ylabel("recall")
    axis.set_ylim(0, 1.05)
    axis.set_title("Deterministic recall by category and injection strength")
    axis.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(edabench_results_dir / "recall.png", dpi=150)
    print(table.round(3))

# measures the stage times on the local datasets and the agent cost of...
# earlier runs
def run_timings():
    from edatool import audit
    rows = []
    slowest = []
    for entry in local_datasets:
        folder = edabench_runs_dir / "timings"
        audit(local_dataset_path(entry["file"]), target=entry["target"], task=entry["task"], use_agent=False, run_dir=folder)
        latest = max(folder.iterdir(), key=lambda path: path.stat().st_mtime)
        timings = json.loads((latest / "timings.json").read_text(encoding="utf-8"))
        report = json.loads((latest / "report.json").read_text(encoding="utf-8"))
        meta = report["meta"]
        rows.append([entry["file"], meta["rows"], meta["columns"], timings["load"], timings["types"], timings["meta"], timings["detectors"], timings["enrichment"], timings["total"], len(report["issues"])])
        top = sorted(timings["per_detector"].items(), key=lambda item: -item[1])[:3]
        slowest.append(f"- {entry['file']}: " + ", ".join(f"{name} {seconds:.2f} s" for name, seconds in top))
    latest_runs = {}
    for folder in sorted((edabench_runs_dir / "agent").glob("2*")):
        latest_runs[folder.name.split("-", 2)[-1]] = folder
    agent_rows = []
    for folder in latest_runs.values():
        timings = json.loads((folder / "timings.json").read_text(encoding="utf-8"))
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if "agent_tokens" in timings:
            tokens = timings["agent_tokens"]
            cost = tokens["input"] / 1e6 * prices["gpt-4.1-mini"][0] + tokens["output"] / 1e6 * prices["gpt-4.1-mini"][1]
            agent_rows.append([report["meta"]["dataset"], len(timings["agent_sessions"]), tokens["input"], tokens["output"], timings["agent"], cost, report["meta"]["complete"]])
    lines = [
        "# Timings and cost (local datasets)",
        "",
        "Deterministic stages, seconds (one run on this laptop):",
        "",
        tabulate(rows, headers=["Dataset", "Rows", "Columns", "Load", "Types", "Meta", "Detectors", "Enrichment", "Total", "Issues"], tablefmt="pipe", floatfmt=".2f"),
        "",
        "Slowest detectors:",
        *slowest,
        "",
        "Agent review (gpt-4.1-mini, M7 runs; later runs of the same dataset reuse cached sessions):",
        "",
        tabulate(agent_rows, headers=["Dataset", "Sessions", "Input tokens", "Output tokens", "Agent s", "Cost ($)", "Complete"], tablefmt="pipe", floatfmt=".3f"),
    ]
    write_result("timings", lines, {"deterministic": rows, "agent": agent_rows})

# command line entry point of the experiments
def main():
    parser = argparse.ArgumentParser(description="M9 experiments.")
    parser.add_argument("experiment", choices=["baselines", "ablations", "realagent", "sensitivity", "figures", "timings"])
    args = parser.parse_args()
    experiments = {
        "baselines": run_baselines,
        "ablations": run_ablations,
        "realagent": run_real_agent,
        "sensitivity": run_sensitivity,
        "figures": run_figures,
        "timings": run_timings,
    }
    experiments[args.experiment]()

if __name__ == "__main__":
    main()
