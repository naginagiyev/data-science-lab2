import pandas as pd
from operations.base import Operation, Params, is_missing

# one spelling and the canonical value it maps to
class Mapping(Params):
    value: str
    canonical: str

# parameters of map_categories
class MapCategoriesParams(Params):
    column: str
    items: list[Mapping]
    case_fold: bool = True
    strip: bool = True

# merges spellings of the same thing ("NY", "new york" -> "New York"); values...
# without a mapping stay as they are (stripped when strip is true)
class MapCategories(Operation):
    name = "map_categories"
    Parameters = MapCategoriesParams
    purpose = "maps spellings of the same category to one canonical value (case and spaces ignored by default)"
    example = {"column": "City", "items": [{"value": "NY", "canonical": "New York"}, {"value": "new york", "canonical": "New York"}], "case_fold": True, "strip": True}
    actions = ["normalize_labels"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return [self.config.column]

    def key(self, value):
        text = str(value)
        text = text.strip() if self.config_.strip else text
        return text.casefold() if self.config_.case_fold else text

    def apply(self, frame):
        config = self.config_
        mapping = {self.key(item.value): item.canonical for item in config.items}
        inputs = frame[config.column]
        missing = is_missing(inputs)
        cleaned = inputs.astype(str).str.strip() if config.strip else inputs.astype(str)
        outputs = pd.Series([mapping.get(self.key(value), clean) for value, clean in zip(inputs, cleaned)], index=inputs.index, dtype=object)
        outputs = outputs.where(~missing, inputs.where(~missing))
        changed = int(((outputs != inputs.astype(str)) & ~missing).sum())
        frame[config.column] = outputs
        return frame, {"changed": changed, "present": int((~missing).sum()), "failures": 0, "examples": {}}

# parameters of group_rare
class GroupRareParams(Params):
    column: str
    min_count: int = 10
    label: str = "Other"

# levels with fewer than min_count training rows (and unseen levels) become...
# one label; the frequent levels are learned on the training rows
class GroupRare(Operation):
    name = "group_rare"
    Parameters = GroupRareParams
    purpose = "levels with fewer than min_count training rows (and unseen levels) become `label`"
    example = {"column": "Cpu", "min_count": 10, "label": "Other"}
    actions = ["group_rare"]

    def inputs(self):
        return [self.config.column]

    def outputs(self):
        return [self.config.column]

    def learn(self, frame, y):
        values = frame[self.config_.column]
        counts = values[~is_missing(values)].astype(str).str.strip().value_counts()
        self.frequent_ = set(counts[counts >= self.config_.min_count].index)

    def apply(self, frame):
        inputs = frame[self.config_.column]
        missing = is_missing(inputs)
        text = inputs.astype(str).str.strip()
        rare = ~missing & ~text.isin(self.frequent_)
        frame[self.config_.column] = text.where(~rare, self.config_.label).where(~missing, inputs.where(~missing))
        return frame, {"changed": int(rare.sum()), "present": int((~missing).sum()), "failures": 0, "examples": {}}
