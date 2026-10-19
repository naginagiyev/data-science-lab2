import pandas as pd
from tools.sandbox import run_code
from operations.base import Operation, Params

FREE_CLASS = "CleaningStep"
CACHE = {}

# a hash of a frame's columns and values
def frame_key(frame):
    return (tuple(frame.columns), int(pd.util.hash_pandas_object(frame, index=False).sum()), len(frame))

# runs generated code through the sandbox once per (code, training rows, input...
# rows): the CV folds and the engine candidates repeat the same transforms
def cached_run(code, class_name, train, frame):
    key = (hash(code), class_name, frame_key(train), frame_key(frame))
    if key not in CACHE:
        if len(CACHE) >= 256:
            CACHE.pop(next(iter(CACHE)))
        CACHE[key] = run_code(code, class_name, train, [frame])[0]
    return CACHE[key].copy()

# parameters of freecode: the generated cleaning class
class FreeCodeParams(Params):
    code: str

# the free-code variant: one generated class cleans the whole feature table...
# (any columns in, any columns out); it runs only in the sandbox and refits on...
# the kept training rows at every transform
class FreeCodeOperation(Operation):
    name = "freecode"
    Parameters = FreeCodeParams
    purpose = "free-code variant only: a generated class cleans the whole table"
    example = {"code": "class CleaningStep: ..."}
    actions = []

    def learn(self, frame, y):
        self.train_ = frame.astype(str).reset_index(drop=True)

    def apply(self, frame):
        output = cached_run(self.config_.code, FREE_CLASS, self.train_, frame.astype(str).reset_index(drop=True))
        output.index = frame.index
        return output, {"changed": 0, "present": int(frame.size), "failures": 0, "examples": {}}
