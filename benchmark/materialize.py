import sys
import json
import hashlib
import argparse
from paths import benchmark_files_dir
from tools.metrics import primary_metric
from benchmark.corruptor import corrupt, write_layout
from benchmark.splits import dataset_split, fixed_sample
from benchmark.datasets import DEV_POOL, dataset_spec, load_dataset

# returns the sha256 hash of a file
def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

# the folder of one benchmark cell: group, dataset, level (with the mess type...
# when only one type is used) and seed
def cell_dir(group, name, level, seed, mess="mixed"):
    folder = f"level{level}" if mess in {"mixed", None} else f"level{level}-{mess}"
    return benchmark_files_dir / group / name / folder / f"seed{seed}"

# writes a frame as text: empty cells stay empty, UTF-8 without BOM, LF endings
def write_text_csv(frame, path):
    frame.to_csv(path, index=False, na_rep="", encoding="utf-8", lineterminator="\n")

# returns the clean raw table of a dataset: rows with an empty target removed...
# and the fixed sample applied (the same for every seed and level)
def clean_table(name):
    spec = dataset_spec(name)
    raw = load_dataset(name)
    raw = raw[raw[spec["target"]].str.strip() != ""].reset_index(drop=True)
    return fixed_sample(raw, spec["target"], spec["task"], spec["sample_rows"], seed=0), spec

# writes train.csv, test_features.csv, test_labels.csv, truth.json and...
# cells.parquet of one dataset, level, seed and mess (mixed or one type); the...
# mess is applied to the full table before the fixed split; the layout type...
# changes only how the feature files are written
def materialize(name, level, seed, overwrite=False, mess="mixed", group=None):
    raw, spec = clean_table(name)
    group = group or spec["group"]
    folder = cell_dir(group, name, level, seed, mess)
    if (folder / "truth.json").exists() and not overwrite:
        return folder
    target = spec["target"]
    train_rows, test_rows = dataset_split(name, raw, target, spec["task"], seed)
    types = None if mess in {"mixed", "layout"} else [mess]
    table, corruption, cells = corrupt(raw, target, name, seed, level if mess != "layout" else 0, types)
    folder.mkdir(parents=True, exist_ok=True)
    train = table.iloc[train_rows]
    test = table.iloc[test_rows]
    if mess == "layout":
        write_layout(train, folder / "train.csv", level)
        write_layout(test.drop(columns=[target]), folder / "test_features.csv", level)
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
        "mess": corruption["items"], "mess_kind": mess, "positions": positions,
        "files": {file.name: file_hash(file) for file in sorted(folder.glob("*.csv"))},
    }
    (folder / "truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n")
    return folder

# command line: writes the files of the chosen datasets, levels and seeds
def main():
    parser = argparse.ArgumentParser(description="Write benchmark files.")
    parser.add_argument("--datasets", nargs="*", default=DEV_POOL)
    parser.add_argument("--levels", nargs="*", type=int, default=[0])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mess", default="mixed")
    arguments = parser.parse_args()
    for name in arguments.datasets:
        for level in arguments.levels:
            for seed in arguments.seeds:
                folder = materialize(name, level, seed, arguments.overwrite, arguments.mess)
                print(f"{name} level {level} seed {seed}: {folder}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
