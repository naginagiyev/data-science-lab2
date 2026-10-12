import re
import numpy as np
import pandas as pd
from operations.categories import Mapping
from operations.base import Operation, Params

# parameters of map_target_labels
class MapTargetLabelsParams(Params):
    items: list[Mapping]

# parameters of log_target
class LogTargetParams(Params):
    offset: float = 1.0

# the spelling key of a label: lowercase, no outer or repeated spaces, no...
# punctuation; two labels with the same key are spellings of one class
def spelling_key(value):
    text = re.sub(r"[^\w\s]", "", str(value).casefold())
    return re.sub(r"\s+", " ", text).strip()

# base of the target operations: they change how the target is read, never...
# the label space (no class merging, no capping of an evaluated target)
class TargetOperation(Operation):
    kind = "target"

    def transform(self, frame):
        raise RuntimeError(f"{self.name} works on the target, not on the features")

# normalizes spellings of existing classes ("yes " -> "Yes"); a mapping between...
# two different classes is refused, so classes are never merged
class MapTargetLabels(TargetOperation):
    name = "map_target_labels"
    Parameters = MapTargetLabelsParams
    purpose = "classification only: maps spellings of an existing class to that class (\"yes \" -> \"Yes\"); never merges different classes"
    example = {"items": [{"value": "yes ", "canonical": "Yes"}]}
    actions = ["normalize_labels"]

    def check(self, labels):
        known = set(pd.Series(labels).astype(str))
        problems = []
        for item in self.config.items:
            if item.value not in known:
                problems.append(f"'{item.value}' does not occur in the target; map only values that occur (or leave out map_target_labels)")
            elif item.canonical not in known:
                problems.append(f"canonical label '{item.canonical}' is not an existing class")
            elif spelling_key(item.value) != spelling_key(item.canonical):
                problems.append(f"'{item.value}' and '{item.canonical}' are different classes; classes are never merged")
        return problems

    def map(self, labels):
        mapping = {item.value: item.canonical for item in self.config.items}
        return pd.Series(labels).astype(str).map(lambda value: mapping.get(value, value)).to_numpy()

# trains regression models on log(target + offset); predictions are turned back...
# with exp(prediction) - offset, so scores stay on the original scale
class LogTarget(TargetOperation):
    name = "log_target"
    Parameters = LogTargetParams
    purpose = "regression only: train on log(target + offset); predictions and scores are on the original scale"
    example = {"offset": 1.0}
    actions = ["log_transform"]

    def check(self, values):
        values = np.asarray(values, dtype=float)
        return [] if (values + self.config.offset > 0).all() else ["log_target needs target + offset > 0 for every training row"]

    def forward(self, values):
        return np.log(np.asarray(values, dtype=float) + self.config.offset)

    def inverse(self, values):
        return np.exp(np.asarray(values, dtype=float)) - self.config.offset
