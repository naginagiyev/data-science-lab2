import numpy as np
import pandas as pd
from operations.rules import evaluate_rule
from operations.base import Operation, Params, is_missing

# parameters of the row operations without options
class NoParams(Params):
    pass

# parameters of drop_rows_where
class DropRowsWhereParams(Params):
    condition: str

# a typed copy of a raw frame for rule conditions: columns that are at least...
# 90% numbers become numbers, others stay text; empty cells are missing
def typed_view(frame):
    typed = {}
    for column in frame.columns:
        values = frame[column]
        numbers = pd.to_numeric(values.astype(str).str.strip(), errors="coerce")
        present = ~is_missing(values)
        typed[column] = numbers if present.sum() and numbers[present].notna().mean() >= 0.9 else values.where(present)
    return pd.DataFrame(typed, index=frame.index)

# base of the row operations: they only remove training rows (inside each CV...
# fold and in the final refit); they never run at prediction time and never...
# touch validation or test rows
class RowOperation(Operation):
    kind = "row"
    Parameters = NoParams

    def mask(self, frame, y):
        raise NotImplementedError

    def transform(self, frame):
        raise RuntimeError(f"{self.name} is a training-only row filter; it never runs at prediction time")

# drops exact copies of training rows (features and target), keeping the first
class DropDuplicateRows(RowOperation):
    name = "drop_duplicate_rows"
    purpose = "drops exact copies of training rows (features and target); training rows only"
    example = {}
    actions = ["drop_rows"]

    def mask(self, frame, y):
        joined = frame.astype(str).assign(__target__=pd.Series(y, index=frame.index).astype(str))
        return ~joined.duplicated().to_numpy()

# drops all training rows whose features repeat with a different target (label...
# noise)
class DropConflictingRows(RowOperation):
    name = "drop_conflicting_rows"
    purpose = "drops training rows that have the same features but different targets (label noise); training rows only"
    example = {}
    actions = ["drop_rows"]

    def mask(self, frame, y):
        keys = pd.util.hash_pandas_object(frame.astype(str), index=False)
        targets = pd.Series(y, index=frame.index).astype(str)
        conflicting = targets.groupby(keys.to_numpy()).transform("nunique").to_numpy() > 1
        return ~conflicting

# drops training rows where a condition in the rule language is true...
# (never eval)
class DropRowsWhere(RowOperation):
    name = "drop_rows_where"
    Parameters = DropRowsWhereParams
    purpose = "drops training rows where a condition is true; condition language: `column` names, numbers, 'text', + - * /, < <= > >= == !=, and/or/not, missing(`column`)"
    example = {"condition": "`sqft_living` > 13000"}
    actions = ["drop_rows"]

    def mask(self, frame, y):
        hit, _ = evaluate_rule(typed_view(frame), self.config.condition, frame)
        return ~hit.to_numpy()

# all row steps of a plan as one filter; it returns which training rows to keep
class RowFilter:
    def __init__(self, operations):
        self.operations = operations

    def mask(self, frame, y):
        keep = np.ones(len(frame), dtype=bool)
        for operation in self.operations:
            keep &= operation.mask(frame, y)
        return keep
