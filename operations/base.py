import time
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator, TransformerMixin

# base class of the parameter models: unknown parameters are errors
class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

# true for cells that hold no value: NaN, None or text that is empty after strip
def is_missing(values):
    series = pd.Series(values)
    text = series.astype(str).str.strip()
    return series.isna() | (text == "")

# parses text to numbers after stripping spaces; failures become NaN
def to_numbers(values):
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce").astype(float)

# counts the present input values that became missing (parse failures) and...
# keeps up to 10 example values
def failures(inputs, outputs):
    present = ~is_missing(inputs)
    failed = present & pd.Series(outputs, index=inputs.index).isna()
    examples = inputs[failed].astype(str).value_counts().head(10)
    return {"present": int(present.sum()), "failures": int(failed.sum()), "examples": {str(key): int(value) for key, value in examples.items()}}

# the base of every operation: a scikit-learn transformer with validated...
# parameters; fit learns state from training rows only (most operations learn...
# nothing); transform returns a frame with the same index and number of rows,...
# never touches columns it does not own, and records stats_ (changed values,...
# parse failures with examples, seconds)
class Operation(BaseEstimator, TransformerMixin):
    name = "operation"
    kind = "column"
    Parameters = Params
    purpose = ""
    example = {}
    actions = []

    def __init__(self, params=None):
        self.params = params

    @property
    def config(self):
        return self.Parameters.model_validate(self.params or {})

    def inputs(self):
        return []

    def outputs(self):
        return []

    def removes(self):
        return []

    def learn(self, frame, y):
        return None

    def apply(self, frame):
        raise NotImplementedError

    def fit(self, frame, y=None):
        config = self.config
        missing = [column for column in self.inputs() if column not in frame.columns]
        if missing:
            raise KeyError(f"{self.name}: unknown column(s) {missing}")
        self.config_ = config
        self.learn(frame, y)
        return self

    def transform(self, frame):
        start = time.perf_counter()
        missing = [column for column in self.inputs() if column not in frame.columns]
        if missing:
            raise KeyError(f"{self.name}: unknown column(s) {missing}")
        result, stats = self.apply(frame.copy())
        if len(result) != len(frame) or not result.index.equals(frame.index):
            raise ValueError(f"{self.name} changed the number of rows or the index")
        stats["seconds"] = round(time.perf_counter() - start, 4)
        self.stats_ = stats
        return result

    def describe(self):
        return {"op": self.name, "params": self.config.model_dump()}

# returns a float column with NaN where the parse failed, and the parse stats
def numeric_result(inputs, outputs):
    outputs = pd.Series(np.asarray(outputs, dtype=float), index=inputs.index)
    return outputs, failures(inputs, outputs)
