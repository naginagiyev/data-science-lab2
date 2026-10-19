import sys
import joblib
import logging
import sklearn
import catboost
import lightgbm
import numpy as np
import pandas as pd
from tools.reader import read_table, to_raw_view

logger = logging.getLogger("automl.model")

# the saved artifact of a run: the trained model (preparation pipelines and...
# models), the label encoder, the raw input columns, the task, the metric,...
# the plan records, versions and the settings hash; it predicts from raw input
class FittedModel:
    def __init__(self, trained, encoder, input_columns, target, task, metric, settings_hash, plan=None, target_transform=None):
        self.trained = trained
        self.encoder = encoder
        self.input_columns = list(input_columns)
        self.target = target
        self.task = task
        self.metric = metric
        self.settings_hash = settings_hash
        self.plan = plan
        self.target_transform = target_transform
        self.versions = {
            "python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        }

    @property
    def classes(self):
        return [] if self.encoder is None else [str(label) for label in self.encoder.classes_]

    def prepare(self, source):
        raw = read_table(source)[0] if not isinstance(source, pd.DataFrame) else to_raw_view(source)
        missing = [column for column in self.input_columns if column not in raw.columns]
        extra = [column for column in raw.columns if column not in self.input_columns]
        if missing:
            logger.warning(f"{len(missing)} input columns are missing and treated as empty: {missing[:10]}")
        if extra:
            logger.warning(f"{len(extra)} extra columns are ignored: {extra[:10]}")
        for column in missing:
            raw[column] = ""
        return raw[self.input_columns].reset_index(drop=True)

    def raw_values(self, source):
        values = self.trained.predict_values(self.prepare(source))
        if self.task == "regression" and self.target_transform is not None:
            values = self.target_transform.inverse(values)
        return values

    def predict_proba(self, source):
        if self.task == "regression":
            raise ValueError("predict_proba is only defined for classification.")
        return pd.DataFrame(self.raw_values(source), columns=self.classes)

    def predict(self, source):
        values = self.raw_values(source)
        if self.task == "regression":
            return values
        return np.asarray(self.classes, dtype=object)[values.argmax(axis=1)]

    def predict_frame(self, source):
        values = self.raw_values(source)
        if self.task == "regression":
            return pd.DataFrame({"prediction": values})
        frame = pd.DataFrame(values, columns=self.classes)
        frame["label"] = np.asarray(self.classes, dtype=object)[values.argmax(axis=1)]
        return frame

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
