import io
import re
import warnings
import contextlib
import numpy as np
import pandas as pd
from types import SimpleNamespace
from edabench.legacy.eda import EDA
from paths import edabench_runs_dir
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# maps ydata-profiling alerts and old eda.py warnings to our issue codes
ydata_codes = {
    "MISSING": "C01",
    "CONSTANT": "E01",
    "UNIQUE": "E03",
    "HIGH_CORRELATION": "E08",
    "DUPLICATES": "A04",
    "HIGH_CARDINALITY": "E04",
    "SKEWED": "E09",
    "IMBALANCE": "D05",
    "ZEROS": "C03",
    "INFINITE": "D01",
    "EMPTY": "C06",
    "CONSTANT_LENGTH": "B03",
}
legacy_patterns = [
    (r"Column '(.+)' has high missingness", "C01"),
    (r"Column '(.+)' is fully missing", "C06"),
    (r"Column '(.+)' is 100% missing", "C06"),
    (r"Mixed types detected in column '(.+)'", "B03"),
    (r"Column '(.+)' appears numeric-like in string format", "B01"),
    (r"skew in '(.+)'", "E09"),
    (r"Outliers above 5% in '(.+)'", "D02"),
    (r"High cardinality in '(.+)'", "E04"),
    (r"Column '(.+)' appears identifier-like", "E03"),
    (r"Potential multicollinearity: '(.+)' vs '(.+)'", "E08"),
    (r"Class imbalance in target '(.+)'", "F04"),
]

# reads a raw view the way pd.read_csv would load the file
def as_csv_frame(raw):
    buffer = io.StringIO()
    raw.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer, low_memory=False)

# a minimal issue-like object for the results of a baseline tool
def pseudo(code, columns):
    return SimpleNamespace(code=code, columns=list(columns), fingerprint=f"{code}|{'|'.join(sorted(columns))}")

# runs ydata-profiling and turns its alerts into issue-like objects
def ydata_issues(raw, target, task):
    from ydata_profiling import ProfileReport
    frame = as_csv_frame(raw)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        report = ProfileReport(frame, progress_bar=False)
        alerts = report.description_set.alerts
    found = []
    for alert in alerts:
        name = alert.alert_type.name
        code = ydata_codes.get(name)
        if code is None:
            continue
        column = alert.column_name
        if name == "DUPLICATES":
            found.append(pseudo(code, []))
        elif name == "HIGH_CORRELATION":
            fields = alert.values.get("fields", []) if isinstance(alert.values, dict) else []
            others = [field for field in fields if field != column]
            if target in others and column != target:
                found.append(pseudo("G01", [column]))
            found.append(pseudo(code, [column] + others))
        elif column is not None:
            found.append(pseudo(code, [column]))
    return found

# runs the old eda.py and turns its warnings into issue-like objects
def legacy_issues(raw, target, task, name):
    folder = edabench_runs_dir / "baselines"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    raw.to_csv(path, index=False)
    eda = EDA(input_path=str(path), target_col=target, task_type=task)
    eda.output_path = str(folder / f"{name}eda.md")
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        eda.run()
    found = []
    for message in eda.warnings:
        for pattern, code in legacy_patterns:
            match = re.search(pattern, message)
            if match:
                found.append(pseudo(code, list(match.groups())))
                break
    return found

# numeric features for cleanlab (one-hot encoding for small categories)
def encoded_features(frame, target):
    features = frame.drop(columns=[target])
    numeric = features.select_dtypes(include="number")
    numeric = numeric.fillna(numeric.median())
    categories = features.select_dtypes(exclude="number")
    small = [column for column in categories.columns if categories[column].nunique() <= 30]
    encoded = pd.get_dummies(categories[small].astype(str), dtype=float)
    return pd.concat([numeric, encoded], axis=1).fillna(0).to_numpy(dtype=float)

# runs cleanlab Datalab and returns the flagged rows per issue type
def cleanlab_rows(raw, target, task):
    from cleanlab import Datalab
    frame = as_csv_frame(raw)
    frame = frame[frame[target].notna()].reset_index(drop=True)
    features = encoded_features(frame, target)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        if task == "regression":
            lab = Datalab(data={"y": frame[target].astype(float).to_numpy()}, label_name="y", task="regression")
            lab.find_issues(features=features, issue_types={"outlier": {}, "near_duplicate": {}, "null": {}})
        else:
            labels = frame[target].astype(str).to_numpy()
            classes, codes = np.unique(labels, return_inverse=True)
            model = LogisticRegression(max_iter=300)
            probabilities = cross_val_predict(model, features, codes, cv=3, method="predict_proba")
            lab = Datalab(data={"y": labels}, label_name="y")
            lab.find_issues(features=features, pred_probs=probabilities, issue_types={"label": {}, "outlier": {}, "near_duplicate": {}, "null": {}})
        issues = lab.get_issues()
    flagged = {}
    for name in ["label", "outlier", "near_duplicate", "null"]:
        column = f"is_{name}_issue"
        if column in issues.columns:
            flagged[name] = set(np.where(issues[column].to_numpy())[0].tolist())
    return flagged

# cleanlab issue types that match each injected kind
cleanlab_kinds = {
    "outliers": "outlier",
    "duplicates": "near_duplicate",
    "conflicts": "label",
    "missing": "null",
    "hidden_missing": "null",
}

# an injected row issue counts as found when cleanlab flags at least half of...
# its rows
def cleanlab_recall(flagged, ground_truth):
    records = []
    for item in ground_truth:
        issue_type = cleanlab_kinds.get(item["kind"])
        found = False
        if issue_type and item["rows"]:
            rows = set(item["rows"])
            found = len(rows & flagged.get(issue_type, set())) >= 0.5 * len(rows)
        records.append({"kind": item["kind"], "code": item["code"], "found": found, "strict": found})
    return records
