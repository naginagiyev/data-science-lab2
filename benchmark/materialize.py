import sys
import json
import hashlib
import argparse
from paths import benchmark_files_dir
from tools.metrics import primary_metric
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
# and the fixed sample applied (the same for every seed and level)
def clean_table(name):
    spec = dataset_spec(name)
    raw = load_dataset(name)
    raw = raw[raw[spec["target"]].str.strip() != ""].reset_index(drop=True)
    return fixed_sample(raw, spec["target"], spec["task"], spec["sample_rows"], seed=0), spec

# writes train.csv, test_features.csv, test_labels.csv and truth.json of one...
# dataset, level and seed; the mess is applied to the full table before the split
def materialize(name, level, seed, overwrite=False):
    raw, spec = clean_table(name)
    folder = cell_dir(spec["group"], name, level, seed)
    if (folder / "truth.json").exists() and not overwrite:
        return folder
    target = spec["target"]
    train_rows, test_rows = dataset_split(name, raw, target, spec["task"], seed)
    if level != 0:
        raise NotImplementedError("Mess levels above 0 need the corruptor (P6.2).")
    table, mess = raw, []
    folder.mkdir(parents=True, exist_ok=True)
    train = table.iloc[train_rows]
    test = table.iloc[test_rows]
    write_text_csv(train, folder / "train.csv")
    write_text_csv(test.drop(columns=[target]), folder / "test_features.csv")
    write_text_csv(test[[target]], folder / "test_labels.csv")
    truth = {
        "dataset": name, "group": spec["group"], "level": level, "seed": seed, "target": target,
        "task": spec["task"], "metric": primary_metric(spec["task"]), "rows": {"train": len(train), "test": len(test)},
        "mess": mess,
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
    arguments = parser.parse_args()
    for name in arguments.datasets:
        for level in arguments.levels:
            for seed in arguments.seeds:
                folder = materialize(name, level, seed, arguments.overwrite)
                print(f"{name} level {level} seed {seed}: {folder}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
