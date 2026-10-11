import openml
import numpy as np
import pandas as pd
from paths import data_dir, openml_cache_dir

# the local datasets, the clean OpenML datasets for injection and the dirty...
# OpenML datasets
local_datasets = [
    {"file": "ibmchurn.csv", "target": "Churn", "task": "binary-classification"},
    {"file": "kinghousedata.csv", "target": "price", "task": "regression"},
    {"file": "winequality.csv", "target": "quality", "task": "multi-class-classification"},
    {"file": "creditcardfraud.csv", "target": "Class", "task": "binary-classification"},
]

clean_datasets = [
    {"name": "credit-g", "id": 31},
    {"name": "bank-marketing", "id": 1461},
    {"name": "australian", "id": 40981},
    {"name": "churn", "id": 40701},
    {"name": "electricity", "id": 151},
    {"name": "segment", "id": 40984},
    {"name": "abalone", "id": 44956},
    {"name": "cpuact", "id": 197},
    {"name": "boston", "id": 531},
    {"name": "airfoil", "id": 44957},
]

dirty_datasets = [
    {"name": "adult", "id": 1590},
    {"name": "creditapproval", "id": 29},
    {"name": "titanic", "id": 40945},
    {"name": "hepatitis", "id": 55},
    {"name": "houseprices", "id": 42563},
]

# returns the path of a local dataset in the project data folder
def local_dataset_path(file_name):
    return data_dir / file_name

# turns one OpenML column into text, the way a CSV file would store it
def column_as_text(values):
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        numbers = values.astype(float)
        whole = numbers.dropna()
        if len(whole) and (whole == np.floor(whole)).all():
            text = numbers.astype("Int64").astype(str)
        else:
            text = numbers.astype(str)
        return text.where(numbers.notna(), "")
    return values.astype(object).where(values.notna(), "").astype(str)

# turns an OpenML DataFrame into a raw text view
def to_raw(frame, name):
    raw = pd.DataFrame({column: column_as_text(frame[column]) for column in frame.columns}).reset_index(drop=True)
    raw.attrs["edatool"] = {"source": name, "format": "openml", "raw_text": True}
    return raw

# decides the task type from the target values
def openml_task(dataset, target_values):
    if pd.api.types.is_numeric_dtype(target_values) and target_values.nunique() > 10:
        return "regression"
    classes = dataset.qualities.get("NumberOfClasses") or target_values.nunique()
    return "binary-classification" if int(classes) == 2 else "multi-class-classification"

# downloads (or reads from the cache) an OpenML dataset and returns the raw...
# view, target and task
def load_openml(entry):
    openml.config.set_root_cache_directory(str(openml_cache_dir))
    dataset = openml.datasets.get_dataset(entry["id"], download_data=True, download_qualities=True, download_features_meta_data=True)
    frame, _, _, _ = dataset.get_data(dataset_format="dataframe")
    target = dataset.default_target_attribute.split(",")[0]
    task = openml_task(dataset, frame[target])
    return to_raw(frame, entry["name"]), target, task
