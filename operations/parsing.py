import re
import warnings
import numpy as np
import pandas as pd
from typing import Literal
from operations.base import Operation, Params, failures, is_missing, numeric_result

PARTS = ["year", "month", "day", "weekday", "hour", "epoch_days"]
NUMBER = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$")

# one suffix and its multiplication factor ("k" -> 1000)
class Suffix(Params):
    suffix: str
    factor: float

# one unit and its factor to the common unit ("g" -> 0.001 for kg)
class Unit(Params):
    unit: str
    factor: float

# parameters of parse_number
class ParseNumberParams(Params):
    column: str
    remove: list[str] = []
    decimal: Literal[".", ","] = "."
    thousands: str | None = None
    percent: bool = False
    suffixes: list[Suffix] = []

# text to number: removes symbols, thousand separators and suffixes (k, M); with...
# percent the "%" is removed and the value divided by 100
class ParseNumber(Operation):
    name = "parse_number"
    Parameters = ParseNumberParams
    purpose = "text to number: removes symbols, thousand separators and suffixes (k, M); percent=true divides values with % by 100"
    example = {"column": "Price", "remove": ["$"], "thousands": ",", "decimal": ".", "percent": False, "suffixes": [{"suffix": "k", "factor": 1000}]}
    actions = ["convert_type"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return [self.config.column]

    def parse(self, value):
        config = self.config_
        text = str(value).strip()
        if text == "" or value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        factor = 1.0
        if config.percent and text.endswith("%"):
            text, factor = text[:-1].strip(), 0.01
        for item in sorted(config.suffixes, key=lambda item: -len(item.suffix)):
            if item.suffix and text.lower().endswith(item.suffix.lower()):
                text, factor = text[: -len(item.suffix)].strip(), factor * item.factor
                break
        for token in config.remove:
            text = text.replace(token, "")
        if config.thousands:
            text = text.replace(config.thousands, "")
        if config.decimal == ",":
            text = text.replace(",", ".")
        text = text.replace(" ", "")
        try:
            return float(text) * factor
        except ValueError:
            return np.nan

    def apply(self, frame):
        column = self.config_.column
        inputs = frame[column]
        outputs, stats = numeric_result(inputs, [self.parse(value) for value in inputs])
        frame[column] = outputs
        return frame, stats

# parameters of parse_units
class ParseUnitsParams(Params):
    column: str
    units: list[Unit]
    default_factor: float | None = None

# number with a unit to one common unit ("1.37kg" -> 1.37, "800 g" -> 0.8);...
# a value without a unit uses default_factor (or fails)
class ParseUnits(Operation):
    name = "parse_units"
    Parameters = ParseUnitsParams
    purpose = "number with a unit to one common unit; units maps each unit to its factor; values without a unit use default_factor"
    example = {"column": "Weight", "units": [{"unit": "kg", "factor": 1.0}, {"unit": "g", "factor": 0.001}], "default_factor": None}
    actions = ["convert_type"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return [self.config.column]

    def parse(self, value, factors):
        match = NUMBER.match(str(value).replace(",", ""))
        if match is None:
            return np.nan
        unit = match.group(2).lower().rstrip(".")
        if unit == "":
            factor = self.config_.default_factor
        else:
            factor = factors.get(unit)
        return float(match.group(1)) * factor if factor is not None else np.nan

    def apply(self, frame):
        column = self.config_.column
        factors = {item.unit.lower().strip().rstrip("."): item.factor for item in self.config_.units}
        inputs = frame[column]
        outputs = [np.nan if missing else self.parse(value, factors) for value, missing in zip(inputs, is_missing(inputs))]
        outputs, stats = numeric_result(inputs, outputs)
        frame[column] = outputs
        return frame, stats

# parameters of parse_datetime
class ParseDatetimeParams(Params):
    column: str
    formats: list[str] = ["mixed"]
    dayfirst: bool = False
    parts: list[Literal["year", "month", "day", "weekday", "hour", "epoch_days"]] = ["year", "month", "day", "weekday"]
    keep_original: bool = False

# parses text dates with the given formats (tried in order per value, or...
# "mixed") and returns them as datetimes
def parse_dates(values, formats, dayfirst):
    text = pd.Series(values).astype(str).str.strip()
    result = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    for date_format in formats:
        todo = result.isna() & ~is_missing(values)
        if not todo.any():
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if date_format == "mixed":
                parsed = pd.to_datetime(text[todo], errors="coerce", format="mixed", dayfirst=dayfirst)
            else:
                parsed = pd.to_datetime(text[todo], errors="coerce", format=date_format)
        result[todo] = parsed
    return result

# text to date parts (year, month, day, weekday, hour, days since 1970); the...
# original column is removed unless keep_original
class ParseDatetime(Operation):
    name = "parse_datetime"
    Parameters = ParseDatetimeParams
    purpose = "text dates to numeric parts; formats are strptime formats tried in order (or [\"mixed\"]); creates <column>_<part> columns"
    example = {"column": "date", "formats": ["%Y%m%dT%H%M%S"], "dayfirst": False, "parts": ["year", "month", "day", "weekday"], "keep_original": False}
    actions = ["convert_type", "extract_datetime"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return [f"{self.config.column}_{part}" for part in self.config.parts]

    def removes(self):
        return [] if self.config.keep_original else [self.config.column]

    def apply(self, frame):
        config = self.config_
        inputs = frame[config.column]
        dates = parse_dates(inputs, config.formats, config.dayfirst)
        values = {
            "year": dates.dt.year, "month": dates.dt.month, "day": dates.dt.day, "weekday": dates.dt.weekday,
            "hour": dates.dt.hour, "epoch_days": (dates - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400,
        }
        for part in config.parts:
            frame[f"{config.column}_{part}"] = values[part].astype(float)
        stats = failures(inputs, dates.where(dates.notna(), np.nan))
        if not config.keep_original:
            frame = frame.drop(columns=[config.column])
        return frame, stats

# parameters of parse_boolean
class ParseBooleanParams(Params):
    column: str
    true_values: list[str]
    false_values: list[str]

# many spellings of yes and no to 1 and 0 (case and spaces ignored)
class ParseBoolean(Operation):
    name = "parse_boolean"
    Parameters = ParseBooleanParams
    purpose = "spellings of yes/no to 1/0 (case and spaces ignored); other values become missing"
    example = {"column": "Partner", "true_values": ["yes", "y", "true", "1"], "false_values": ["no", "n", "false", "0"]}
    actions = ["normalize_labels", "convert_type"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return [self.config.column]

    def apply(self, frame):
        config = self.config_
        mapping = {value.strip().lower(): 0.0 for value in config.false_values}
        mapping.update({value.strip().lower(): 1.0 for value in config.true_values})
        inputs = frame[config.column]
        outputs = inputs.astype(str).str.strip().str.lower().map(mapping)
        outputs, stats = numeric_result(inputs, outputs.where(~is_missing(inputs)))
        frame[config.column] = outputs
        return frame, stats
