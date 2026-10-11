import re
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from pandas._libs.parsers import STR_NA_VALUES
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from skrub import DatetimeEncoder, StringEncoder, TableVectorizer

BOOLEAN_TEXT = {"True": 1.0, "TRUE": 1.0, "true": 1.0, "False": 0.0, "FALSE": 0.0, "false": 0.0}
LOW_CARDINALITY = 30
STRING_COMPONENTS = 30
FORBIDDEN_NAME = re.compile(r"[\[\]{}\":,]")

# turns pandas' default missing tokens into NaN (exact match, like read_csv;...
# "?" and " " are not among them)
def pandas_missing(values):
    return values.where(~values.isin(STR_NA_VALUES), np.nan)

# makes column names safe for LightGBM (no [ ] { } " : , and no duplicates);...
# returns the new names
def sanitize_names(names):
    used = {}
    result = []
    for name in names:
        clean = FORBIDDEN_NAME.sub("_", str(name)).strip() or "column"
        count = used.get(clean, 0)
        used[clean] = count + 1
        result.append(clean if count == 0 else f"{clean}_{count + 1}")
    return result

# mimics pd.read_csv defaults on text columns that no operation produced: the...
# pandas NA tokens become missing, a column whose other values all parse as...
# numbers becomes numeric, True/False columns become 0/1; fit decides on the...
# training rows and transform applies the same decision to any rows
class TypingStep(BaseEstimator, TransformerMixin):
    def fit(self, frame, y=None):
        self.numeric_ = []
        self.boolean_ = []
        for column in frame.columns:
            if not pd.api.types.is_object_dtype(frame[column]):
                continue
            values = pandas_missing(frame[column].astype(object))
            present = values.dropna().astype(str)
            if present.empty:
                continue
            if present.isin(BOOLEAN_TEXT).all() and values.notna().all():
                self.boolean_.append(column)
            elif pd.to_numeric(present, errors="coerce").notna().all():
                self.numeric_.append(column)
        return self

    def transform(self, frame):
        frame = frame.copy()
        for column in frame.columns:
            if not pd.api.types.is_object_dtype(frame[column]):
                continue
            values = pandas_missing(frame[column].astype(object))
            if column in self.boolean_:
                frame[column] = values.map(BOOLEAN_TEXT).astype(float)
            elif column in self.numeric_:
                frame[column] = pd.to_numeric(values, errors="coerce").astype(float)
            else:
                frame[column] = values
        return frame

# the generic preparation every variant shares (skrub's TableVectorizer):...
# numbers pass, dates become parts, strings with at most 30 levels become...
# ordinal codes (trees) or one-hot columns (linear), other strings become 30...
# StringEncoder features; linear models also get median imputation and scaling;...
# output names are safe for LightGBM and the map to the original names is kept
class GenericStage(BaseEstimator, TransformerMixin):
    def __init__(self, kind="tree", seed=0):
        self.kind = kind
        self.seed = seed

    def make_vectorizer(self):
        options = {
            "cardinality_threshold": LOW_CARDINALITY + 1,
            "high_cardinality": StringEncoder(n_components=STRING_COMPONENTS, random_state=self.seed),
            "datetime": DatetimeEncoder(resolution="day", add_weekday=True, add_total_seconds=True),
        }
        if self.kind == "tree":
            options["low_cardinality"] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=np.nan)
        return TableVectorizer(**options)

    def fit(self, frame, y=None):
        self.vectorizer_ = self.make_vectorizer()
        output = self.vectorizer_.fit_transform(self.clean_input(frame))
        output = self.to_frame(output)
        if self.kind == "linear":
            self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(output)
            self.scaler_ = StandardScaler().fit(self.imputer_.transform(output))
        self.names_ = sanitize_names(output.columns) if len(output.columns) else ["no_features"]
        self.name_map_ = dict(zip(self.names_, [str(name) for name in output.columns] or ["no_features"]))
        return self

    def clean_input(self, frame):
        frame = frame.copy()
        frame.columns = [str(column) for column in frame.columns]
        numbers = frame.select_dtypes("number").columns
        frame[numbers] = frame[numbers].replace([np.inf, -np.inf], np.nan)
        return frame

    def to_frame(self, output):
        if isinstance(output, pd.DataFrame):
            return output.reset_index(drop=True)
        return pd.DataFrame(np.asarray(output))

    def transform(self, frame):
        output = self.to_frame(self.vectorizer_.transform(self.clean_input(frame)))
        if output.shape[1] == 0:
            return pd.DataFrame({"no_features": np.zeros(len(frame))})
        values = output.to_numpy(dtype=float)
        if self.kind == "linear":
            values = self.scaler_.transform(self.imputer_.transform(output)).astype(float)
        return pd.DataFrame(values, columns=self.names_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.names_, dtype=object)
