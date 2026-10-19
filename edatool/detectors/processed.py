import numpy as np
import pandas as pd
from edatool.core.registry import register
from edatool.detectors.common import all_types, numeric_types

# P01: non-numeric columns left in the processed data
@register("P01", applies_to=all_types, mode="verify")
def non_numeric_left(context, columns):
    for column in columns:
        if context.types[column] not in numeric_types | {"boolean", "constant", "empty"}:
            evidence = {"type": context.types[column]}
            context.ledger.add("P01", [column], "non_numeric", evidence)

# P02: missing or infinite values left in the processed data
@register("P02", applies_to=all_types, mode="verify")
def nan_or_infinite(context, columns):
    for column in columns:
        values = context.typed[column]
        missing = int(values.isna().sum())
        infinite = int(np.isinf(values.astype(float)).sum()) if context.types[column] in numeric_types else 0
        if missing or infinite:
            evidence = {"missing": missing, "infinite": infinite}
            context.ledger.add("P02", [column], "nan_or_inf", evidence)

# P03: columns that became constant during processing
@register("P03", applies_to={"constant", "empty"}, mode="verify")
def new_constant_columns(context, columns):
    before = context.baseline_types or {}
    for column in columns:
        original = context.lineage.get(column, column)
        if before.get(original) not in {"constant", "empty"}:
            evidence = {"original_column": original, "original_type": before.get(original)}
            context.ledger.add("P03", [column], "became_constant", evidence)

# P04: processing created far more columns than the original data had
@register("P04", mode="verify")
def column_explosion(context, columns):
    before = len(context.baseline_types or {}) or context.raw.shape[1]
    growth = context.raw.shape[1] / max(before, 1)
    if growth > context.value("P04", "max_column_growth"):
        evidence = {"columns_before": before, "columns_after": context.raw.shape[1], "growth": round(growth, 2)}
        context.ledger.add("P04", [], "growth", evidence)

# P05: train and test have different columns, order or types
@register("P05", mode="verify")
def schema_mismatch(context, columns):
    test = context.test_frame
    if test is None:
        return
    train = context.raw
    evidence = {}
    missing = [str(column) for column in train.columns if column not in test.columns]
    extra = [str(column) for column in test.columns if column not in train.columns]
    if missing or extra:
        evidence["missing_in_test"] = missing[:20]
        evidence["extra_in_test"] = extra[:20]
    elif list(train.columns) != list(test.columns):
        evidence["order_differs"] = True
    shared = [column for column in train.columns if column in test.columns]
    dtypes = {str(column): [str(train[column].dtype), str(test[column].dtype)] for column in shared if pd.api.types.is_numeric_dtype(train[column]) != pd.api.types.is_numeric_dtype(test[column])}
    if dtypes:
        evidence["dtype_differs"] = dict(list(dtypes.items())[:20])
    if evidence:
        context.ledger.add("P05", [], "train_test", evidence)
