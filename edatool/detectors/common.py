import numpy as np
import pandas as pd
from scipy.stats import skew

# groups of semantic types used to select columns
numeric_types = {"numeric_continuous", "numeric_discrete"}
category_types = {"categorical_nominal", "categorical_ordinal"}
all_types = {
    "numeric_continuous",
    "numeric_discrete",
    "numeric_code",
    "categorical_nominal",
    "categorical_ordinal",
    "boolean",
    "datetime",
    "text",
    "id",
    "constant",
    "empty",
}

# returns the row ids where a mask is True
def mask_index(mask):
    return mask.index[np.asarray(mask, dtype=bool)]

# returns a column as text, with missing values as ''
def text_values(context, column):
    raw = context.raw[column]
    return raw.astype(str).where(raw.notna(), "")

# returns the parsed numbers of a column without missing values
def numeric_values(context, column):
    return context.typed[column].dropna().astype(float)

# interquartile range (q3 - q1)
def iqr(values):
    first, third = np.percentile(values, [25, 75])
    return float(third - first)

# robust z-scores based on the median and MAD; None when MAD is 0
def robust_z(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return None
    return (values - median) / (1.4826 * mad)

# skewness, or 0 for constant or very short data
def skewness(values):
    if len(values) < 3 or np.std(values) == 0:
        return 0.0
    return float(skew(values))

# lowercases labels and removes punctuation and extra spaces
def normalize_labels(values):
    lowered = values.astype(str).str.lower()
    return lowered.str.replace(r"[\W_]+", " ", regex=True).str.strip()

# finds labels that are equal after normalization and maps each variant to...
# the most common one
def label_variants(values):
    counts = values.astype(str).value_counts()
    normalized = normalize_labels(pd.Series(counts.index, index=counts.index))
    variants = {}
    for key, group in counts.groupby(normalized.to_numpy()):
        if len(group) > 1:
            canonical = group.idxmax()
            variants[key] = {label: canonical for label in group.index if label != canonical}
    return variants

# returns the target as plain objects for class counting
def class_target(context):
    return context.typed[context.target].astype(object)
