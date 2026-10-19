import json
import time
import argparse
import numpy as np
import pandas as pd
from edatool import load_raw
from tabulate import tabulate
from edatool.api import build_context
from edatool.core.enrichment import enrich
from edatool.settings import load_settings
from edatool.core.registry import run_detectors
from edabench.injector import inject, strengths
from edatool.core.typeinference import infer_schema
from paths import data_dir, edabench_labels_dir, edabench_results_dir
from edabench.datasets import clean_datasets, dirty_datasets, load_openml, local_dataset_path, local_datasets

# issue codes that count as finding each injected kind, and the category...
# names of the code letters
accepted_codes = {
    "missing": {"C01", "C04", "C06", "G04"},
    "hidden_missing": {"C02"},
    "hidden_missing_type": {"B01"},
    "sentinel": {"C03", "D01", "D02"},
    "outliers": {"D02", "F07"},
    "negatives": {"D01"},
    "label_variants": {"B05", "D05"},
    "number_format": {"B01", "B03"},
    "duplicates": {"A04"},
    "conflicts": {"A05"},
    "id_column": {"E03"},
    "constant_column": {"E01"},
    "duplicate_column": {"A03", "E06", "E07", "E08"},
    "linear_relation": {"E07", "E08", "D04"},
    "one_to_one": {"E06"},
    "leakage": {"G01", "G02", "G03"},
    "units": {"B07"},
    "mixed_units": {"B07"},
    "packed": {"B08"},
    "mixed_dates": {"B09"},
}
categories = {
    "A": "structure",
    "B": "types",
    "C": "missing",
    "D": "validity",
    "E": "columns",
    "F": "target",
    "G": "leakage",
}

# compares the inferred column types with the type labels
def type_accuracy(include_drafts=False):
    targets = {dataset["file"]: dataset["target"] for dataset in local_datasets}
    label_files = sorted(edabench_labels_dir.glob("types-*.json"))
    results = []
    for label_file in label_files:
        label = json.loads(label_file.read_text(encoding="utf-8"))
        if not label["reviewed"] and not include_drafts:
            continue
        raw = load_raw(data_dir / label["dataset"])
        schema, _, _ = infer_schema(raw, targets[label["dataset"]], load_settings())
        predicted = {column.name: column.type for column in schema}
        errors = {
            name: {"label": expected, "predicted": predicted.get(name)}
            for name, expected in label["types"].items()
            if predicted.get(name) != expected
        }
        results.append(
            {
                "dataset": label["dataset"],
                "reviewed": label["reviewed"],
                "correct": len(label["types"]) - len(errors),
                "total": len(label["types"]),
                "errors": errors,
            }
        )
    return results, len(label_files)

# runs the detectors and enrichment without the agent and returns the context
def run_ledger(raw, target, task, settings=None):
    context = build_context(raw, target, task, settings, None, {})
    _, _, complete = run_detectors(context, "audit")
    enrich(context)
    return context, complete

# checks if an issue matches an injected issue (accepted code and the same...
# columns)
def matches(issue, item):
    if issue.code not in accepted_codes[item["kind"]]:
        return False
    if not item["columns"]:
        return True
    return bool(set(issue.columns) & set(item["columns"]))

# F1 score between the found rows and the injected rows
def row_f1(found_rows, true_rows):
    found_rows = set(int(row) for row in found_rows)
    true_rows = set(true_rows)
    both = len(found_rows & true_rows)
    if not both:
        return 0.0
    precision = both / len(found_rows)
    recall = both / len(true_rows)
    return 2 * precision * recall / (precision + recall)

# scores one run: found injections, row F1, fix and link checks, true and...
# false positives (issues already in the clean data are ignored)
def score_run(context, baseline, ground_truth, issues=None):
    issues = issues if issues is not None else context.ledger.issue_list()
    new_issues = [issue for issue in issues if issue.fingerprint not in baseline]
    records = []
    for item in ground_truth:
        related = [issue for issue in issues if matches(issue, item)]
        primary = [issue for issue in related if issue.code == item["code"]] or related
        strict = any(issue.code == item["code"] for issue in related)
        record = {"kind": item["kind"], "code": item["code"], "found": bool(related), "strict": strict}
        if primary and item["rows"] is not None:
            record["row_f1"] = row_f1(context.ledger.rows.get(primary[0].id, []), item["rows"])
        if primary and item["fix"]:
            record["fix_ok"] = any(issue.fix is not None and issue.fix.action == item["fix"] for issue in primary)
        if item["kind"] == "hidden_missing_type" and primary:
            roots = {issue.id for issue in issues if issue.code == "C02" and set(issue.columns) == set(item["columns"])}
            record["link_ok"] = primary[0].caused_by in roots
        records.append(record)
    true_positive = [issue for issue in new_issues if any(matches(issue, item) for item in ground_truth)]
    false_positive = [issue for issue in new_issues if issue not in true_positive]
    return records, len(true_positive), [{"code": issue.code, "columns": issue.columns} for issue in false_positive]

# injects issues into every clean dataset at each strength and scores the...
# deterministic audit
def run_benchmark(names=None, strength_names=None, seed=0):
    results = []
    for entry in clean_datasets:
        if names and entry["name"] not in names:
            continue
        raw, target, task = load_openml(entry)
        clean_context, _ = run_ledger(raw, target, task)
        baseline = {issue.fingerprint for issue in clean_context.ledger.issue_list()}
        for strength in strength_names or list(strengths):
            dirty, ground_truth = inject(raw, target, strength, seed)
            start = time.perf_counter()
            context, complete = run_ledger(dirty, target, task)
            seconds = time.perf_counter() - start
            records, true_positive, false_positive = score_run(context, baseline, ground_truth)
            results.append(
                {
                    "dataset": entry["name"],
                    "task": task,
                    "strength": strength,
                    "rows": len(dirty),
                    "seconds": round(seconds, 2),
                    "complete": complete,
                    "records": records,
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                }
            )
            print(f"{entry['name']} {strength}: found {sum(r['found'] for r in records)}/{len(records)}, fp {len(false_positive)}, {seconds:.1f}s")
    return results

# turns run scores into a table per category and the overall recall and...
# precision
def summarize(results):
    rows = {}
    for result in results:
        for record in result["records"]:
            category = categories[record["code"][0]]
            row = rows.setdefault(category, {"injected": 0, "found": 0, "strict": 0, "f1": [], "fix": [], "link": []})
            row["injected"] += 1
            row["found"] += record["found"]
            row["strict"] += record["strict"]
            if "row_f1" in record:
                row["f1"].append(record["row_f1"])
            if "fix_ok" in record:
                row["fix"].append(record["fix_ok"])
            if "link_ok" in record:
                row["link"].append(record["link_ok"])
    true_positive = sum(result["true_positive"] for result in results)
    false_positive = sum(len(result["false_positive"]) for result in results)
    table = []
    for category, row in sorted(rows.items()):
        table.append(
            [
                category,
                row["injected"],
                row["found"] / row["injected"],
                row["strict"] / row["injected"],
                np.mean(row["f1"]) if row["f1"] else None,
                np.mean(row["fix"]) if row["fix"] else None,
                np.mean(row["link"]) if row["link"] else None,
            ]
        )
    injected = sum(row["injected"] for row in rows.values())
    found = sum(row["found"] for row in rows.values())
    strict = sum(row["strict"] for row in rows.values())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    return table, {"injected": injected, "recall": found / injected if injected else None, "strict_recall": strict / injected if injected else None, "precision": precision, "true_positive": true_positive, "false_positive": false_positive}

# writes the benchmark table and data into benchmark/results
def write_benchmark(results, name):
    edabench_results_dir.mkdir(parents=True, exist_ok=True)
    table, overall = summarize(results)
    headers = ["Category", "Injected", "Recall", "Strict recall", "Row F1", "Fix action OK", "Cause link OK"]
    false_codes = pd.Series([item["code"] for result in results for item in result["false_positive"]]).value_counts()
    lines = [
        f"# Injection benchmark: {name}",
        "",
        f"Runs: {len(results)} · injected issues: {overall['injected']} · recall: {overall['recall']:.3f} (strict {overall['strict_recall']:.3f}) · precision: {overall['precision']:.3f} (TP {overall['true_positive']}, FP {overall['false_positive']})",
        "",
        tabulate(table, headers=headers, tablefmt="pipe", floatfmt=".3f", missingval="-"),
        "",
        "False positives by code: " + (", ".join(f"{code} {count}" for code, count in false_codes.items()) or "none"),
        "",
        f"Mean run time: {np.mean([result['seconds'] for result in results]):.2f} s",
    ]
    (edabench_results_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (edabench_results_dir / f"{name}.json").write_text(json.dumps({"overall": overall, "runs": results}, indent=2, default=str) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return overall

# reads the issue label files (only reviewed ones, unless drafts are allowed)
def issue_labels(include_drafts=False):
    found = []
    for label_file in sorted(edabench_labels_dir.glob("issues-*.json")):
        label = json.loads(label_file.read_text(encoding="utf-8"))
        if label["reviewed"] or include_drafts:
            found.append(label)
    return found

# loads the dataset of a label file (local or OpenML)
def load_labelled(label):
    if label["source"] == "local":
        entry = next(item for item in local_datasets if item["file"] == label["dataset"])
        return load_raw(local_dataset_path(entry["file"])), entry["target"], entry["task"]
    entry = next(item for item in dirty_datasets if item["name"] == label["dataset"])
    return load_openml(entry)

# checks the labelled real datasets and writes which labelled issues were...
# found
def run_real(include_drafts=False):
    labels = issue_labels(include_drafts)
    lines = ["# Real datasets: labelled issues", "", f"Label files used: {len(labels)} (reviewed: {sum(label['reviewed'] for label in labels)})", ""]
    totals = {"labels": 0, "found": 0}
    for label in labels:
        raw, target, task = load_labelled(label)
        context, _ = run_ledger(raw, target, task)
        run_detectors(context, "presplit", codes={"H01", "H02", "H03", "H04", "H05"})
        issues = context.ledger.issue_list()
        lines.append(f"## {label['dataset']} ({'reviewed' if label['reviewed'] else 'draft'})")
        for item in label["issues"]:
            hit = [issue.id for issue in issues if issue.code in item["codes"] and (not item["columns"] or set(issue.columns) & set(item["columns"]))]
            totals["labels"] += 1
            totals["found"] += bool(hit)
            status = f"found ({', '.join(hit)})" if hit else "MISSED"
            lines.append(f"- {'/'.join(item['codes'])} {', '.join(item['columns']) or '(dataset)'}: {status}. {item['note']}")
        lines.append(f"- issues reported by the tool: {len(issues)}")
        lines.append("")
    if totals["labels"]:
        lines.append(f"Overall: {totals['found']}/{totals['labels']} labelled issues found ({totals['found'] / totals['labels']:.1%}).")
    edabench_results_dir.mkdir(parents=True, exist_ok=True)
    (edabench_results_dir / "realdatasets.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

# command line entry point of the evaluation
def main():
    parser = argparse.ArgumentParser(description="Evaluate edatool.")
    commands = parser.add_subparsers(dest="command", required=True)
    types_command = commands.add_parser("types", help="type inference accuracy")
    types_command.add_argument("--drafts", action="store_true", help="also use labels that are not reviewed")
    bench_command = commands.add_parser("benchmark", help="injection benchmark, deterministic only")
    bench_command.add_argument("--datasets", nargs="*")
    bench_command.add_argument("--strengths", nargs="*")
    bench_command.add_argument("--name", default="benchmark-deterministic")
    real_command = commands.add_parser("real", help="labelled real datasets")
    real_command.add_argument("--drafts", action="store_true")
    args = parser.parse_args()
    if args.command == "types":
        results, file_count = type_accuracy(args.drafts)
        reviewed = sum(result["reviewed"] for result in results)
        print(f"Label files: {file_count}, reviewed: {reviewed}, used: {len(results)}")
        for result in results:
            status = "reviewed" if result["reviewed"] else "draft"
            print(f"{result['dataset']} ({status}): {result['correct']}/{result['total']} = {result['correct'] / result['total']:.1%}")
            for name, error in result["errors"].items():
                print(f"  {name}: label {error['label']}, predicted {error['predicted']}")
    elif args.command == "benchmark":
        write_benchmark(run_benchmark(args.datasets, args.strengths), args.name)
    else:
        run_real(args.drafts)

if __name__ == "__main__":
    main()
