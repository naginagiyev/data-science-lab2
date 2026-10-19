import sys
import json
import hashlib
import argparse
from benchmark.leaks import add_leak
from benchmark.corruptor import corrupt
from tools.metrics import primary_metric
from paths import benchmark_files_dir, results_dir
from benchmark.splits import dataset_split, fixed_sample
from benchmark.datasets import DEV_POOL, dataset_spec, load_dataset

# returns the sha256 hash of a file
def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

# the folder of one benchmark cell: group, dataset, level and seed
def cell_dir(group, name, level, seed):
    return benchmark_files_dir / group / name / f"level{level}" / f"seed{seed}"

# writes a frame as text: empty cells stay empty, UTF-8 without BOM, LF endings
def write_text_csv(frame, path):
    frame.to_csv(path, index=False, na_rep="", encoding="utf-8", lineterminator="\n")

# returns the clean raw table of a dataset: rows with an empty target removed...
# and the fixed sample applied (the same for every seed and level); sample_rows...
# gives a smaller fixed sample (dev iteration)
def clean_table(name, sample_rows=None):
    spec = dataset_spec(name)
    raw = load_dataset(name)
    raw = raw[raw[spec["target"]].str.strip() != ""].reset_index(drop=True)
    rows = min(value for value in [spec["sample_rows"], sample_rows, len(raw)] if value)
    return fixed_sample(raw, spec["target"], spec["task"], rows, seed=0), spec

# writes train.csv, test_features.csv, test_labels.csv, truth.json and...
# cells.parquet of one dataset, level and seed; the mess is applied to the full...
# table before the fixed split
def materialize(name, level, seed, overwrite=False, group=None, sample_rows=None):
    raw, spec = clean_table(name, sample_rows)
    group = group or spec["group"]
    folder = cell_dir(group, name, level, seed)
    if (folder / "truth.json").exists() and not overwrite:
        return folder
    target = spec["target"]
    split_name = f"{name}-n{sample_rows}" if sample_rows else name
    train_rows, test_rows = dataset_split(split_name, raw, target, spec["task"], seed)
    leak = None
    if group == "D":
        raw, leak = add_leak(raw, name, target, spec["task"], seed)
    table, corruption, cells = corrupt(raw, target, name, seed, level)
    folder.mkdir(parents=True, exist_ok=True)
    train = table.iloc[train_rows]
    test = table.iloc[test_rows]
    if leak is not None:
        write_text_csv(train, folder / "train.csv")
        write_text_csv(test.drop(columns=[target]).assign(**{leak["column"]: ""}), folder / "test_features.csv")
        write_text_csv(test.drop(columns=[target]), folder / "test_features_naive.csv")
    else:
        write_text_csv(train, folder / "train.csv")
        write_text_csv(test.drop(columns=[target]), folder / "test_features.csv")
    write_text_csv(test[[target]], folder / "test_labels.csv")
    positions = {"train": [int(row) for row in train_rows], "test": [int(row) for row in test_rows]}
    if len(cells):
        cells.to_parquet(folder / "cells.parquet", index=False)
    truth = {
        "dataset": name, "group": group, "level": level, "seed": seed, "target": target,
        "task": spec["task"], "metric": primary_metric(spec["task"]), "rows": {"train": len(train), "test": len(test)},
        "mess": corruption["items"], "leak": leak, "positions": positions,
        "files": {file.name: file_hash(file) for file in sorted(folder.glob("*.csv"))},
    }
    (folder / "truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n")
    return folder

# lists every materialized cell of the benchmark groups (A, C, D) with its...
# rows and file hashes in benchmark/results/files.json
def write_manifest(groups=("A", "C", "D")):
    cells = []
    for truth_file in sorted(path for group in groups for path in (benchmark_files_dir / group).glob("*/*/*/truth.json")):
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        cells.append({
            "folder": truth_file.parent.relative_to(benchmark_files_dir).as_posix(), "dataset": truth["dataset"], "group": truth["group"],
            "level": truth["level"], "seed": truth["seed"], "rows": truth["rows"], "files": truth["files"],
        })
    (results_dir / "files.json").write_text(json.dumps(cells, indent=1) + "\n", encoding="utf-8", newline="\n")
    return cells

# command line: writes the files of the chosen datasets, levels and seeds
def main():
    parser = argparse.ArgumentParser(description="Write benchmark files.")
    parser.add_argument("--datasets", nargs="*", default=DEV_POOL)
    parser.add_argument("--levels", nargs="*", type=int, default=[0])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--sample-rows", type=int)
    parser.add_argument("--manifest", action="store_true", help="only write benchmark/results/files.json")
    arguments = parser.parse_args()
    if arguments.manifest:
        print(f"{len(write_manifest())} cells in the manifest")
        return 0
    for name in arguments.datasets:
        for level in arguments.levels:
            for seed in arguments.seeds:
                folder = materialize(name, level, seed, arguments.overwrite, arguments.group, arguments.sample_rows)
                print(f"{name} level {level} seed {seed}: {folder}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
