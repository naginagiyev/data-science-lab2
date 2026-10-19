import numpy as np
import pandas as pd
from schemas import Issue
from scipy.stats import skew
from tools.columntypes import CATEGORY_TYPES, NUMERIC_TYPES

ALL_TYPES = NUMERIC_TYPES | CATEGORY_TYPES | {"numeric_code", "boolean", "datetime", "text", "id", "constant", "empty"}

# everything a check needs: the raw and typed views, the column types and...
# details, the target, the task, the catalog and thresholds, and the found issues
class Context:
    def __init__(self, raw, typed, types, details, target, task, config, sample_rows):
        self.raw = raw
        self.typed = typed
        self.types = types
        self.details = details
        self.target = target
        self.task = task
        self.config = config
        self.max_sample = sample_rows
        self.n_rows = len(raw)
        self.is_classification = task != "regression"
        self.issues = []
        self.sample = None

    # one threshold of a check from configuration/audit.json
    def value(self, code, name):
        return self.config["thresholds"][code][name]

    # the feature columns of the given types (never the target)
    def columns_of(self, types):
        return [column for column in self.raw.columns if self.types[column] in types and column != self.target]

    # a column as text, with missing values as ''
    def text(self, column):
        return self.raw[column].astype(str).where(self.raw[column].notna(), "")

    # a row sample for the slower checks; for classification it keeps the...
    # class shares
    def sample_rows(self):
        if self.sample is None:
            index = self.raw.index
            if len(index) <= self.max_sample:
                self.sample = index
            elif not self.is_classification:
                self.sample = index[np.sort(np.random.default_rng(0).choice(len(index), self.max_sample, replace=False))]
            else:
                groups = pd.DataFrame({"group": self.typed[self.target].astype(str).to_numpy()}, index=index)
                self.sample = groups.groupby("group", group_keys=False).sample(frac=self.max_sample / len(index), random_state=0).index.sort_values()
        return self.sample

    # adds one issue with the catalog title, priority, diagnosis and default fix;...
    # the fix parameters of the check are added to the defaults
    def add(self, code, columns, rule, evidence, rows=None, params=None):
        entry = self.config["catalog"][code]
        columns = [str(column) for column in columns]
        fix = dict(entry["params"])
        if len(columns) == 1:
            fix["column"] = columns[0]
        elif columns:
            fix["columns"] = columns
        fix.update(params or {})
        number = sum(issue.code == code for issue in self.issues) + 1
        count = int(len(rows)) if rows is not None else 0
        self.issues.append(Issue(id=f"{code}-{number}", code=code, title=entry["title"], columns=columns, rows=count, evidence=evidence, priority=entry["priority"], rule=rule, action=entry["action"], params=fix, diagnosis=entry["diagnosis"]))

# the row ids where a mask is True
def mask_index(mask):
    return mask.index[np.asarray(mask, dtype=bool)]

# interquartile range (q3 - q1)
def iqr(values):
    first, third = np.percentile(values, [25, 75])
    return float(third - first)

# robust z-scores from the median and MAD; None when the MAD is 0
def robust_z(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return None if mad == 0 else (values - median) / (1.4826 * mad)

# skewness, or 0 for constant or very short data
def skewness(values):
    return 0.0 if len(values) < 3 or np.std(values) == 0 else float(skew(values))

# lowercases labels and removes punctuation and extra spaces
def normalize_labels(values):
    return values.astype(str).str.lower().str.replace(r"[\W_]+", " ", regex=True).str.strip()

# labels that are equal after normalization, each variant mapped to the most...
# common spelling
def label_variants(values):
    counts = values.astype(str).value_counts()
    normalized = normalize_labels(pd.Series(counts.index, index=counts.index))
    variants = {}
    for key, group in counts.groupby(normalized.to_numpy()):
        if len(group) > 1:
            variants[key] = {label: group.idxmax() for label in group.index if label != group.idxmax()}
    return variants
