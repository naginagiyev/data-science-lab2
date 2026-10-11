import numpy as np
import pandas as pd
from tools.sandbox import run_code
from operations.base import Operation, Params, failures

CLASS_NAME = "CustomStep"
CACHE = {}

# runs generated code through the sandbox once per (code, training rows, input...
# rows): the CV folds and the engine candidates repeat the same transforms
def cached_run(code, class_name, train, frame):
    key = (hash(code), class_name, frame_key(train), frame_key(frame))
    if key not in CACHE:
        if len(CACHE) >= 256:
            CACHE.pop(next(iter(CACHE)))
        CACHE[key] = run_code(code, class_name, train, [frame])[0]
    return CACHE[key].copy()

# a hash of a frame's columns and values
def frame_key(frame):
    return (tuple(frame.columns), int(pd.util.hash_pandas_object(frame, index=False).sum()), len(frame))

# parameters of custom: the input columns, the new column names, the...
# instructions for the code generator and the generated code
class CustomParams(Params):
    columns: list[str]
    into: list[str]
    instructions: str
    code: str | None = None

# the escape hatch: generated code (prompts/customop.md) turns some columns...
# into new numeric columns, for example "128GB SSD + 1TB HDD" -> ssd_gb, hdd_gb;...
# the code runs only in the sandbox; the training values are kept, so every...
# transform (also at prediction time) fits the code on the same training rows
class CustomOperation(Operation):
    name = "custom"
    Parameters = CustomParams
    purpose = "escape hatch: generated code turns `columns` into new columns `into` following `instructions` (use only when no catalog operation fits); the originals are removed"
    example = {"columns": ["Memory"], "into": ["ssd_gb", "hdd_gb"], "instructions": "sum the SSD and HDD sizes in GB; 1TB = 1000GB"}
    actions = ["extract_pattern", "convert_type"]

    def inputs(self):
        return list(self.config.columns)

    def outputs(self):
        return list(self.config.into)

    def removes(self):
        return list(self.config.columns)

    def learn(self, frame, y):
        if not self.config_.code:
            raise ValueError("custom: no generated code; the code is generated before the checks")
        self.train_ = frame[self.config_.columns].astype(str).reset_index(drop=True)

    def apply(self, frame):
        config = self.config_
        part = frame[config.columns].astype(str)
        output = cached_run(config.code, CLASS_NAME, self.train_, part.reset_index(drop=True))
        missing = [name for name in config.into if name not in output.columns]
        if missing:
            raise ValueError(f"custom: the generated code did not create {missing}")
        for name in config.into:
            values = pd.to_numeric(output[name], errors="coerce").to_numpy(dtype=float)
            frame[name] = pd.Series(np.where(np.isfinite(values), values, np.nan), index=frame.index)
        stats = failures(part.iloc[:, 0], frame[config.into[0]])
        return frame.drop(columns=config.columns), stats

FREE_CLASS = "CleaningStep"

# parameters of freecode: the generated cleaning class
class FreeCodeParams(Params):
    code: str

# the free-code ablation: one generated class cleans the whole feature table...
# (any columns in, any columns out); it runs only in the sandbox and refits on...
# the kept training rows at every transform
class FreeCodeOperation(Operation):
    name = "freecode"
    Parameters = FreeCodeParams
    purpose = "free-code ablation only: a generated class cleans the whole table"
    example = {"code": "class CleaningStep: ..."}
    actions = []

    def learn(self, frame, y):
        self.train_ = frame.astype(str).reset_index(drop=True)

    def apply(self, frame):
        output = cached_run(self.config_.code, FREE_CLASS, self.train_, frame.astype(str).reset_index(drop=True))
        output.index = frame.index
        return output, {"changed": 0, "present": int(frame.size), "failures": 0, "examples": {}}
