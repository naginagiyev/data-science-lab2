import json
import hashlib
import numpy as np
from paths import benchmark_splits_dir
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# returns a short hash of a raw frame (column names and values), used to check...
# that a saved split belongs to the same data
def frame_hash(raw):
    digest = hashlib.sha256("\x1f".join(raw.columns).encode("utf-8"))
    for column in raw.columns:
        digest.update("\x1e".join(raw[column].astype(str)).encode("utf-8"))
    return digest.hexdigest()[:16]

# returns the stratify labels of a classification target, or None when a class...
# has fewer than 2 rows (then the split is random) or for regression
def stratify_labels(labels, task):
    if task == "regression":
        return None
    counts = labels.value_counts()
    return labels if counts.min() >= 2 else None

# draws a fixed sample of rows (stratified for classification), keeping the...
# original row order
def fixed_sample(raw, target, task, rows, seed=0):
    if rows is None or len(raw) <= rows:
        return raw.reset_index(drop=True)
    positions = np.arange(len(raw))
    chosen, _ = train_test_split(positions, train_size=rows, random_state=seed, stratify=stratify_labels(raw[target], task))
    return raw.iloc[np.sort(chosen)].reset_index(drop=True)

# the fixed outer split of one dataset and seed: 80/20, stratified for...
# classification, or by groups (all rows of a group on one side) when a group...
# column is given; returns sorted train and test row positions
def outer_split(raw, target, task, seed, test_size=0.2, group_column=None):
    positions = np.arange(len(raw))
    if group_column:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train, test = next(splitter.split(positions, groups=raw[group_column]))
    else:
        train, test = train_test_split(positions, test_size=test_size, random_state=seed, stratify=stratify_labels(raw[target], task))
    return sorted(int(value) for value in train), sorted(int(value) for value in test)

# loads the saved outer split of a dataset and seed, or makes and saves it; a...
# saved split of different data is an error
def dataset_split(name, raw, target, task, seed, test_size=0.2, group_column=None):
    path = benchmark_splits_dir / name / f"seed{seed}.json"
    data_hash = frame_hash(raw)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["data_hash"] != data_hash:
            raise ValueError(f"The saved split {path} belongs to other data; delete it to remake it.")
        return saved["train"], saved["test"]
    train, test = outer_split(raw, target, task, seed, test_size, group_column)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"dataset": name, "seed": seed, "data_hash": data_hash, "test_size": test_size, "train": train, "test": test}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8", newline="\n")
    return train, test
