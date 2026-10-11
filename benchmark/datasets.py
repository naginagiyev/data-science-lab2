import pandas as pd
from paths import data_dir

# the local datasets of the dev pool: file, target, task and an optional fixed...
# stratified sample size (the OpenML pools are added in P6.1)
LOCAL_DATASETS = {
    "ibmchurn": {"file": "ibmchurn.csv", "target": "Churn", "task": "binary-classification", "sample_rows": None},
    "kinghousedata": {"file": "kinghousedata.csv", "target": "price", "task": "regression", "sample_rows": None},
    "winequality": {"file": "winequality.csv", "target": "quality", "task": "multi-class-classification", "sample_rows": None},
    "creditcardfraud": {"file": "creditcardfraud.csv", "target": "Class", "task": "binary-classification", "sample_rows": 50000},
}
DEV_POOL = list(LOCAL_DATASETS)

# reads a CSV file as a raw text view: every value a string, empty cells ""
def read_raw_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")

# returns the spec of a dataset (target, task, sample size) by name
def dataset_spec(name):
    if name in LOCAL_DATASETS:
        return dict(LOCAL_DATASETS[name], name=name, group="dev")
    raise KeyError(f"Unknown dataset '{name}'.")

# loads a dataset as raw text, before any sampling
def load_dataset(name):
    spec = dataset_spec(name)
    return read_raw_csv(data_dir / spec["file"])
