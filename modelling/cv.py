import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tools.metrics import align_probabilities, compute_scores

# turns the raw target text into model labels: integer codes and a fitted...
# label encoder for classification, floats for regression
def encode_target(values, task):
    text = pd.Series(values).astype(str).str.strip()
    if task == "regression":
        return pd.to_numeric(text, errors="coerce").to_numpy(dtype=float), None
    encoder = LabelEncoder().fit(text)
    return encoder.transform(text), encoder

# the training and validation matrices of one fold after the preparation...
# pipeline was fitted on the (row-filtered) training part only
class FoldData:
    def __init__(self, train_x, train_y, valid_x, valid_y, valid_rows, train_rows):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.valid_rows = valid_rows
        self.train_rows = train_rows

# fits the preparation on the training part of every fold and transforms both...
# parts; row filters touch the training part only, never the validation part
def prepare_folds(frame, y, folds, make_prep, row_filter=None):
    data = []
    everything = np.arange(len(frame))
    for valid in folds:
        valid = np.asarray(valid, dtype=int)
        train = np.setdiff1d(everything, valid)
        train_frame = frame.iloc[train]
        train_y = y[train]
        if row_filter is not None:
            keep = row_filter.mask(train_frame, train_y)
            train_frame, train_y, train = train_frame[keep], train_y[keep], train[keep]
        prep = make_prep()
        train_x = prep.fit_transform(train_frame, train_y)
        valid_x = prep.transform(frame.iloc[valid])
        data.append(FoldData(train_x, train_y, valid_x, y[valid], valid, train))
    return data

# the predictions of a fitted model: probabilities aligned to all classes for...
# classification, values for regression
def model_predictions(model, x, task, n_classes):
    if task == "regression":
        return np.asarray(model.predict(x), dtype=float).ravel()
    return align_probabilities(model.predict_proba(x), list(model.classes_), list(range(n_classes)))

# the primary score of out-of-fold predictions (integer class labels)
def oof_score(task, y, oof, n_classes):
    classes = None if task == "regression" else list(range(n_classes))
    return compute_scores(task, y, oof, classes)["score"]

# an empty out-of-fold array of the right shape
def empty_oof(rows, task, n_classes):
    return np.full(rows, np.nan) if task == "regression" else np.full((rows, n_classes), np.nan)

# fits a fresh model per fold and returns the out-of-fold predictions and the...
# seconds used; make_model() returns an unfitted estimator
def cross_val_oof(fold_data, make_model, task, n_classes, rows):
    start = time.perf_counter()
    oof = empty_oof(rows, task, n_classes)
    for fold in fold_data:
        model = make_model().fit(fold.train_x, fold.train_y)
        oof[fold.valid_rows] = model_predictions(model, fold.valid_x, task, n_classes)
    return oof, time.perf_counter() - start
