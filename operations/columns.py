import re
from typing import Literal
from operations.parsing import parse_dates
from operations.base import Operation, Params, failures, is_missing, to_numbers

# parameters of drop_columns
class DropColumnsParams(Params):
    columns: list[str]

# removes columns (ids, leaks, constant or useless columns)
class DropColumns(Operation):
    name = "drop_columns"
    Parameters = DropColumnsParams
    purpose = "removes columns (ids, leaks, constant or useless columns)"
    example = {"columns": ["customerID"]}
    actions = ["drop_column", "set_role"]

    def inputs(self):
        return list(self.config.columns)

    def removes(self):
        return list(self.config.columns)

    def apply(self, frame):
        return frame.drop(columns=self.config_.columns), {"changed": len(self.config_.columns), "present": 0, "failures": 0, "examples": {}}

# parameters of extract_pattern
class ExtractPatternParams(Params):
    column: str
    pattern: str
    into: list[str]
    types: list[Literal["number", "text"]] = []
    keep_original: bool = False

# regex groups to new columns ("2560x1600" with (\d+)x(\d+) -> width, height);...
# a value without a match gives missing values
class ExtractPattern(Operation):
    name = "extract_pattern"
    Parameters = ExtractPatternParams
    purpose = "regex groups to new columns (one name in `into` and one type per group); the original is removed unless keep_original"
    example = {"column": "ScreenResolution", "pattern": "(\\d+)x(\\d+)", "into": ["screen_width", "screen_height"], "types": ["number", "number"], "keep_original": False}
    actions = ["extract_pattern"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return list(self.config.into)

    def removes(self):
        return [] if self.config.keep_original else [self.config.column]

    def fit(self, frame, y=None):
        config = self.config
        groups = re.compile(config.pattern).groups
        if groups != len(config.into):
            raise ValueError(f"extract_pattern: the pattern has {groups} groups but into has {len(config.into)} names")
        if config.types and len(config.types) != len(config.into):
            raise ValueError("extract_pattern: types must have one entry per name in into")
        return super().fit(frame, y)

    def apply(self, frame):
        config = self.config_
        inputs = frame[config.column]
        extracted = inputs.astype(str).str.extract(config.pattern, expand=True)
        extracted.loc[is_missing(inputs).to_numpy()] = None
        types = config.types or ["number"] * len(config.into)
        for position, name in enumerate(config.into):
            values = extracted.iloc[:, position]
            frame[name] = to_numbers(values.str.replace(",", "", regex=False)) if types[position] == "number" else values
        stats = failures(inputs, extracted.iloc[:, 0] if len(config.into) else inputs)
        if not config.keep_original:
            frame = frame.drop(columns=[config.column])
        return frame, stats

# parameters of date_difference
class DateDifferenceParams(Params):
    start: str
    end: str
    unit: Literal["days", "years"] = "years"
    name: str

# turns a column into times: 4-digit years stay years, other values are...
# parsed as dates; returns fractional years
def as_years(values):
    numbers = to_numbers(values)
    years = numbers.where((numbers >= 1000) & (numbers <= 3000))
    dates = parse_dates(values, ["mixed"], False)
    from_dates = dates.dt.year + (dates.dt.dayofyear - 1) / 365.25
    return years.fillna(from_dates.astype(float))

# the time between two date or year columns (end - start) in days or years,...
# for example the house age at sale
class DateDifference(Operation):
    name = "date_difference"
    Parameters = DateDifferenceParams
    purpose = "end minus start in days or years; start and end are date columns or 4-digit year columns; creates column `name`"
    example = {"start": "yr_built", "end": "date", "unit": "years", "name": "age_at_sale"}
    actions = ["extract_datetime"]

    def inputs(self):
        return [self.config.start, self.config.end]

    def outputs(self):
        return [self.config.name]

    def apply(self, frame):
        config = self.config_
        difference = as_years(frame[config.end]) - as_years(frame[config.start])
        frame[config.name] = difference * 365.25 if config.unit == "days" else difference
        stats = failures(frame[config.end], difference)
        return frame, stats

# parameters of as_category
class AsCategoryParams(Params):
    column: str

# treats a column as a category even if its values look like numbers (codes...
# such as zipcode); unseen values stay as their own level
class AsCategory(Operation):
    name = "as_category"
    Parameters = AsCategoryParams
    purpose = "treats a column as a category even if it looks numeric (codes such as zipcode)"
    example = {"column": "zipcode"}
    actions = ["encode"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return [self.config.column]

    def apply(self, frame):
        inputs = frame[self.config_.column]
        text = inputs.astype(str).str.strip().where(~is_missing(inputs))
        frame[self.config_.column] = text.astype("category")
        return frame, {"changed": 0, "present": int(text.notna().sum()), "failures": 0, "examples": {}}

# parameters of add_missing_indicator
class AddMissingIndicatorParams(Params):
    columns: list[str]

# adds <column>_missing = 1 where the column is missing, else 0
class AddMissingIndicator(Operation):
    name = "add_missing_indicator"
    Parameters = AddMissingIndicatorParams
    purpose = "adds <column>_missing = 1 where the column is missing (use when missingness carries meaning)"
    example = {"columns": ["yr_renovated"]}
    actions = ["add_indicator"]

    def inputs(self):
        return list(self.config.columns)

    def outputs(self):
        return [f"{column}_missing" for column in self.config.columns]

    def apply(self, frame):
        for column in self.config_.columns:
            frame[f"{column}_missing"] = is_missing(frame[column]).astype(float).to_numpy()
        return frame, {"changed": 0, "present": 0, "failures": 0, "examples": {}}
