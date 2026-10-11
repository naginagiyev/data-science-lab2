import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from operations.base import Operation, Params, is_missing
from sklearn.feature_extraction.text import TfidfVectorizer

# parameters of text_features
class TextFeaturesParams(Params):
    column: str
    max_features: int = 2000
    components: int = 10

# long free text to a few numeric columns (TF-IDF + SVD fitted on the training...
# rows); creates <column>_text_<i> and removes the original column
class TextFeatures(Operation):
    name = "text_features"
    Parameters = TextFeaturesParams
    purpose = "long free text to `components` numeric columns (TF-IDF + SVD learned on training rows); removes the original"
    example = {"column": "description", "max_features": 2000, "components": 10}
    actions = ["drop_column"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return [f"{self.config.column}_text_{index}" for index in range(self.config.components)]

    def removes(self):
        return [self.config.column]

    def texts(self, frame):
        values = frame[self.config_.column]
        return values.astype(str).where(~is_missing(values), "").tolist()

    def learn(self, frame, y):
        config = self.config_
        self.vectorizer_ = TfidfVectorizer(max_features=config.max_features, min_df=1)
        matrix = self.vectorizer_.fit_transform(self.texts(frame))
        components = max(1, min(config.components, matrix.shape[1] - 1)) if matrix.shape[1] > 1 else 1
        self.svd_ = TruncatedSVD(n_components=components, random_state=0) if matrix.shape[1] > 1 else None
        if self.svd_ is not None:
            self.svd_.fit(matrix)

    def apply(self, frame):
        config = self.config_
        present = int((~is_missing(frame[config.column])).sum())
        matrix = self.vectorizer_.transform(self.texts(frame))
        values = self.svd_.transform(matrix) if self.svd_ is not None else np.zeros((len(frame), 1))
        for index in range(config.components):
            column = values[:, index] if index < values.shape[1] else np.zeros(len(frame))
            frame[f"{config.column}_text_{index}"] = pd.Series(column, index=frame.index)
        frame = frame.drop(columns=[config.column])
        return frame, {"changed": 0, "present": present, "failures": 0, "examples": {}}
