import re
import sys
import json
import openml
import hashlib
import argparse
import numpy as np
import pandas as pd
from paths import data_dir, openml_cache_dir, results_dir

ANONYMIZED = re.compile(r"^(V|X|f|col|feature|attr|var)_?\d+$", re.IGNORECASE)
MANIFEST = results_dir / "datasets.json"
BINARY, MULTI, REGRESSION = "binary-classification", "multi-class-classification", "regression"

# the local datasets of the dev pool: file, target, task and an optional fixed...
# stratified sample size
LOCAL_DATASETS = {
    "ibmchurn": {"file": "ibmchurn.csv", "target": "Churn", "task": BINARY, "sample_rows": None},
    "kinghousedata": {"file": "kinghousedata.csv", "target": "price", "task": REGRESSION, "sample_rows": None},
    "winequality": {"file": "winequality.csv", "target": "quality", "task": MULTI, "sample_rows": None},
    "creditcardfraud": {"file": "creditcardfraud.csv", "target": "Class", "task": BINARY, "sample_rows": 50000},
}

# the OpenML datasets (TabArena-v0.1 ids, Appendix D): pool, task, pilot flag...
# and a fixed sample size for the two large ones
OPENML_DATASETS = {
    "houses": {"did": 46934, "pool": "dev", "task": REGRESSION},
    "Is-this-a-good-customer": {"did": 46938, "pool": "dev", "task": BINARY},
    "credit-g": {"did": 46918, "pool": "A", "task": BINARY, "pilot": True},
    "Fitness_Club": {"did": 46927, "pool": "A", "task": BINARY},
    "Marketing_Campaign": {"did": 46940, "pool": "A", "task": BINARY},
    "Bank_Customer_Churn": {"did": 46911, "pool": "A", "task": BINARY, "pilot": True},
    "E-CommereShippingData": {"did": 46924, "pool": "A", "task": BINARY, "pilot": True},
    "online_shoppers_intention": {"did": 46947, "pool": "A", "task": BINARY},
    "maternal_health_risk": {"did": 46941, "pool": "A", "task": MULTI, "pilot": True},
    "students_dropout_and_academic_success": {"did": 46960, "pool": "A", "task": MULTI, "pilot": True},
    "SDSS17": {"did": 46955, "pool": "A", "task": MULTI, "sample_rows": 30000},
    "concrete_compressive_strength": {"did": 46917, "pool": "A", "task": REGRESSION, "pilot": True},
    "healthcare_insurance_expenses": {"did": 46931, "pool": "A", "task": REGRESSION},
    "airfoil_self_noise": {"did": 46904, "pool": "A", "task": REGRESSION},
    "Another-Dataset-on-used-Fiat-500": {"did": 46907, "pool": "A", "task": REGRESSION, "pilot": True},
    "miami_housing": {"did": 46942, "pool": "A", "task": REGRESSION},
    "diamonds": {"did": 46923, "pool": "A", "task": REGRESSION, "pilot": True, "sample_rows": 30000},
    "churn": {"did": 46915, "pool": "spare-A", "task": BINARY},
    "credit_card_clients_default": {"did": 46919, "pool": "spare-A", "task": BINARY},
    "blood-transfusion-service-center": {"did": 46913, "pool": "spare-A", "task": BINARY},
    "wine_quality": {"did": 46964, "pool": "spare-A", "task": REGRESSION},
    "HR_Analytics_Job_Change_of_Data_Scientists": {"did": 46935, "pool": "C", "task": BINARY, "pilot": True},
    "in_vehicle_coupon_recommendation": {"did": 46937, "pool": "C", "task": BINARY, "pilot": True},
    "heloc": {"did": 46932, "pool": "C", "task": BINARY, "pilot": True},
    "bank-marketing": {"did": 46910, "pool": "C", "task": BINARY},
    "Food_Delivery_Time": {"did": 46928, "pool": "spare-C", "task": REGRESSION, "sample_rows": 30000},
    "adult": {"did": 1590, "pool": "spare-C", "task": BINARY, "sample_rows": 30000},
    "Titanic": {"did": 40945, "pool": "spare-C", "task": BINARY},
    "employee_salaries": {"did": 42125, "pool": "C", "task": REGRESSION},
    "house_prices_nominal": {"did": 42563, "pool": "spare-C", "task": REGRESSION},
    "credit-approval": {"did": 29, "pool": "spare-C", "task": BINARY},
}
DEV_POOL = list(LOCAL_DATASETS) + [name for name, spec in OPENML_DATASETS.items() if spec["pool"] == "dev"]
GROUP_A = [name for name, spec in OPENML_DATASETS.items() if spec["pool"] == "A"]
GROUP_C = [name for name, spec in OPENML_DATASETS.items() if spec["pool"] == "C"]
GROUP_D = [
    "credit-g", "Fitness_Club", "Marketing_Campaign", "Bank_Customer_Churn", "E-CommereShippingData", "maternal_health_risk",
    "concrete_compressive_strength", "healthcare_insurance_expenses", "Another-Dataset-on-used-Fiat-500", "diamonds",
]

# reads a CSV file as a raw text view: every value a string, empty cells ""
def read_raw_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")

# the file group of a dataset: dev, A (and B with levels > 0), C
def group_of(name):
    if name in LOCAL_DATASETS:
        return "dev"
    pool = OPENML_DATASETS[name]["pool"]
    return pool.split("-")[-1] if pool.startswith("spare") else pool

# the spec of a dataset: target, task, sample size, group
def dataset_spec(name):
    if name in LOCAL_DATASETS:
        return dict(LOCAL_DATASETS[name], name=name, group="dev")
    if name not in OPENML_DATASETS:
        raise KeyError(f"Unknown dataset '{name}'.")
    spec = OPENML_DATASETS[name]
    target = OPENML_DATASETS[name].get("target") or openml_meta(name)["target"]
    return {"name": name, "group": group_of(name), "target": target, "task": spec["task"], "sample_rows": spec.get("sample_rows"), "did": spec["did"]}

# the OpenML dataset object (from the local cache after the first download)
def openml_dataset(name, download=True):
    openml.config.set_root_cache_directory(str(openml_cache_dir))
    return openml.datasets.get_dataset(OPENML_DATASETS[name]["did"], download_data=download, download_qualities=True, download_features_meta_data=True)

# name, version, licence, default target, rows and columns of an OpenML dataset
def openml_meta(name):
    dataset = openml_dataset(name, download=False)
    return {
        "did": OPENML_DATASETS[name]["did"], "openml_name": dataset.name, "version": dataset.version, "licence": dataset.licence,
        "target": dataset.default_target_attribute.split(",")[0], "url": f"https://www.openml.org/d/{OPENML_DATASETS[name]['did']}",
    }

# turns one OpenML column into text, the way a CSV file would store it
def column_as_text(values):
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        numbers = values.astype(float)
        whole = numbers.dropna()
        text = numbers.astype("Int64").astype(str) if len(whole) and (whole == np.floor(whole)).all() else numbers.astype(str)
        return text.where(numbers.notna(), "")
    return values.astype(object).where(values.notna(), "").astype(str)

# loads a dataset as raw text (the way a CSV file stores it), before sampling
def load_dataset(name):
    if name in LOCAL_DATASETS:
        return read_raw_csv(data_dir / LOCAL_DATASETS[name]["file"])
    frame, _, _, _ = openml_dataset(name).get_data(dataset_format="dataframe")
    return pd.DataFrame({column: column_as_text(frame[column]) for column in frame.columns}).reset_index(drop=True)

# the share of column names that read like words (not V1, X3, col_7, ...)
def readable_share(columns):
    readable = [name for name in columns if not ANONYMIZED.match(str(name)) and len(re.findall(r"[A-Za-z]", str(name))) >= 2]
    return len(readable) / max(1, len(columns))

# a short hash of the raw text table (column names and every value)
def table_hash(raw):
    digest = hashlib.sha256("\x1f".join(raw.columns).encode("utf-8"))
    for column in raw.columns:
        digest.update("\x1e".join(raw[column]).encode("utf-8"))
    return digest.hexdigest()[:16]

# downloads every OpenML dataset and writes the manifest: version, licence,...
# target, task, rows, columns, readable-name share and table hash
def build_manifest(names=None):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    for name in names or list(OPENML_DATASETS):
        spec = dataset_spec(name)
        raw = load_dataset(name)
        target_values = raw[spec["target"]]
        manifest[name] = {
            **openml_meta(name), "pool": OPENML_DATASETS[name]["pool"], "task": spec["task"], "rows": len(raw),
            "columns": raw.shape[1], "classes": int(target_values.nunique()) if spec["task"] != REGRESSION else None,
            "readable_share": round(readable_share([column for column in raw.columns if column != spec["target"]]), 3),
            "sample_rows": spec["sample_rows"], "table_hash": table_hash(raw),
        }
        print(f"{name}: {len(raw)} rows, {raw.shape[1]} columns, target {spec['target']}, readable {manifest[name]['readable_share']}", flush=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return manifest

# command line: python -m benchmark.datasets [names...] downloads and checks
def main():
    parser = argparse.ArgumentParser(description="Download the benchmark datasets and write the manifest.")
    parser.add_argument("names", nargs="*")
    build_manifest(parser.parse_args().names or None)
    return 0

if __name__ == "__main__":
    sys.exit(main())
