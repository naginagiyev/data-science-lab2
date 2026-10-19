import numpy as np
import pandas as pd
from operations.base import Operation, Params, is_missing, numeric_result, to_numbers

# one replacement: a raw value and its new value (None means missing)
class Replacement(Params):
    value: str
    new: str | None = None

# parameters of replace_values
class ReplaceValuesParams(Params):
    column: str
    items: list[Replacement]

# replaces exact raw values (spaces around them ignored) by new values or by...
# missing, for example " " or "?" -> missing
class ReplaceValues(Operation):
    name = "replace_values"
    Parameters = ReplaceValuesParams
    purpose = "replaces exact raw values (outer spaces ignored) by a new value, or by missing when new is null"
    example = {"column": "TotalCharges", "items": [{"value": " ", "new": None}, {"value": "?", "new": None}]}
    actions = ["replace_values"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return [self.config.column]

    def apply(self, frame):
        config = self.config_
        inputs = frame[config.column]
        stripped = inputs.astype(str).str.strip()
        outputs = inputs.astype(object).copy()
        changed = pd.Series(False, index=inputs.index)
        for item in config.items:
            hit = (stripped == item.value.strip()) & ~changed & inputs.notna()
            outputs[hit] = np.nan if item.new is None else item.new
            changed |= hit
        frame[config.column] = outputs
        return frame, {"changed": int(changed.sum()), "present": int((~is_missing(inputs)).sum()), "failures": 0, "examples": {}}

# parameters of set_missing_where
class SetMissingWhereParams(Params):
    column: str
    below: float | None = None
    above: float | None = None
    values: list[str] = []

# sets impossible or placeholder values to missing: numbers below or above a...
# limit, or listed raw values; other values stay as they are
class SetMissingWhere(Operation):
    name = "set_missing_where"
    Parameters = SetMissingWhereParams
    purpose = "sets values to missing: numbers below `below` or above `above`, and listed raw values"
    example = {"column": "bedrooms", "below": 0, "above": 20, "values": ["0"]}
    actions = ["replace_values"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return [self.config.column]

    def apply(self, frame):
        config = self.config_
        inputs = frame[config.column]
        numbers = to_numbers(inputs)
        hit = pd.Series(False, index=inputs.index)
        if config.below is not None:
            hit |= numbers < config.below
        if config.above is not None:
            hit |= numbers > config.above
        if config.values:
            hit |= inputs.astype(str).str.strip().isin([value.strip() for value in config.values]) & inputs.notna()
        outputs = inputs.astype(object).where(~hit, np.nan) if not pd.api.types.is_numeric_dtype(inputs) else inputs.where(~hit, np.nan)
        frame[config.column] = outputs
        return frame, {"changed": int(hit.sum()), "present": int((~is_missing(inputs)).sum()), "failures": 0, "examples": {}}

# parameters of clip_values
class ClipValuesParams(Params):
    column: str
    lower_quantile: float | None = None
    upper_quantile: float | None = 0.99

# winsorizes a numeric feature at quantiles learned on the training rows (never...
# the target)
class ClipValues(Operation):
    name = "clip_values"
    Parameters = ClipValuesParams
    purpose = "clips a numeric feature at quantiles learned on training rows (never the target)"
    example = {"column": "sqft_lot", "lower_quantile": None, "upper_quantile": 0.99}
    actions = ["cap_values"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return [self.config.column]

    def learn(self, frame, y):
        numbers = to_numbers(frame[self.config_.column]).dropna()
        config = self.config_
        self.lower_ = float(numbers.quantile(config.lower_quantile)) if config.lower_quantile is not None and len(numbers) else None
        self.upper_ = float(numbers.quantile(config.upper_quantile)) if config.upper_quantile is not None and len(numbers) else None

    def apply(self, frame):
        inputs = frame[self.config_.column]
        numbers = to_numbers(inputs)
        clipped = numbers.clip(lower=self.lower_, upper=self.upper_)
        outputs, stats = numeric_result(inputs, clipped)
        stats["changed"] = int((clipped != numbers).sum() - (clipped.isna() & numbers.isna()).sum())
        frame[self.config_.column] = outputs
        return frame, stats

# parameters of log_transform
class LogTransformParams(Params):
    column: str
    offset: float = 1.0

# log(value + offset) of a numeric feature; values with value + offset <= 0...
# become missing
class LogTransform(Operation):
    name = "log_transform"
    Parameters = LogTransformParams
    purpose = "log(value + offset) of a numeric feature (offset 1 gives log1p)"
    example = {"column": "sqft_lot", "offset": 1.0}
    actions = ["log_transform"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return [self.config.column]

    def apply(self, frame):
        inputs = frame[self.config_.column]
        shifted = to_numbers(inputs) + self.config_.offset
        outputs, stats = numeric_result(inputs, np.log(shifted.where(shifted > 0)))
        frame[self.config_.column] = outputs
        return frame, stats

# parameters of impute_constant
class ImputeConstantParams(Params):
    column: str
    value: str | float

# fills missing cells with a constant that the meaning gives (for example 0...
# when tenure is 0)
class ImputeConstant(Operation):
    name = "impute_constant"
    Parameters = ImputeConstantParams
    purpose = "fills missing cells with a constant given by the meaning (use only when the meaning gives the value)"
    example = {"column": "TotalCharges", "value": 0}
    actions = ["impute"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return [self.config.column]

    def apply(self, frame):
        inputs = frame[self.config_.column]
        missing = is_missing(inputs)
        value = self.config_.value
        if pd.api.types.is_numeric_dtype(inputs):
            outputs = inputs.where(~missing, float(value))
        else:
            outputs = inputs.astype(object).where(~missing, str(value) if not isinstance(value, str) else value)
        frame[self.config_.column] = outputs
        return frame, {"changed": int(missing.sum()), "present": int((~missing).sum()), "failures": 0, "examples": {}}
