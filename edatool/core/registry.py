import time
import hashlib
import logging
import numpy as np
import pandas as pd
from edatool.schemas import CoverageEntry
from edatool.core.progress import issue_text
from edatool.core.sampling import sample_index

logger = logging.getLogger("edatool")
# list of all registered detectors, filled by the register decorator
detectors = []

# decorator that registers a detector with its code, column types, cost,...
# target need and mode
def register(code, applies_to=None, cost="C", needs_target=False, version=1, mode="audit"):
    def wrap(function):
        detectors.append(
            {
                "code": code,
                "name": function.__name__,
                "function": function,
                "applies_to": applies_to,
                "cost": cost,
                "needs_target": needs_target,
                "version": version,
                "mode": mode,
            }
        )
        return function
    return wrap

# everything a detector needs: raw view, typed view, schema, target, task,...
# settings and the ledger
class Context:
    def __init__(self, raw, typed, schema, details, target, task, settings, ledger):
        self.raw = raw
        self.typed = typed
        self.schema = schema
        self.details = details
        self.target = target
        self.task = task
        self.settings = settings
        self.ledger = ledger
        self.types = {column.name: column.type for column in schema}
        self.confidence = {column.name: column.confidence for column in schema}
        self.n_rows = len(raw)
        self.is_classification = task != "regression"
        self.hashes = {}
        self.sample = None
        self.baseline_types = None
        self.lineage = {}
        self.test_frame = None

    # reads one threshold value from the settings
    def value(self, code, name):
        return self.settings["thresholds"][code][name]["value"]

    # returns the columns of the given types, without the target
    def columns_of(self, types):
        return [column for column in self.raw.columns if self.types[column] in types and column != self.target]

    # returns a cached hash per row of one column (numeric columns use their...
    # parsed values)
    def column_hash(self, column):
        if column not in self.hashes:
            numeric = self.types[column] in {"numeric_continuous", "numeric_discrete"}
            values = self.typed[column] if numeric else self.raw[column]
            self.hashes[column] = pd.util.hash_pandas_object(values, index=False).to_numpy()
        return self.hashes[column]

    # returns one hash of the whole dataset, used in the report meta and in...
    # the agent cache
    def dataset_hash(self):
        digest = hashlib.sha256("".join(str(name) for name in self.raw.columns).encode("utf-8"))
        for column in self.raw.columns:
            digest.update(self.column_hash(column).tobytes())
        return digest.hexdigest()

    # combines the column hashes into one hash per row, used to find...
    # duplicate rows
    def row_hash(self, columns):
        combined = np.zeros(self.n_rows, dtype=np.uint64)
        for column in columns:
            combined = combined * np.uint64(1000003) ^ self.column_hash(column)
        return combined

    # returns the cached row sample used by the slower checks
    def sample_rows(self):
        if self.sample is None:
            strata = self.typed[self.target] if self.is_classification else None
            self.sample = sample_index(self.raw.index, self.value("sampling", "sample_rows"), strata)
        return self.sample

# runs every detector of the mode, records coverage and timings and logs what...
# each one found; a crash is recorded, never silent
def run_detectors(context, mode, codes=None, log=True):
    coverage = []
    timings = {}
    complete = True
    for detector in detectors:
        modes = {"audit", "verify"} if mode == "verify" else {mode}
        if detector["mode"] not in modes | {"both"}:
            continue
        if codes is not None and detector["code"] not in codes:
            continue
        start = time.perf_counter()
        columns = context.columns_of(detector["applies_to"]) if detector["applies_to"] else []
        context.ledger.detector = f"{detector['name']} v{detector['version']}"
        skipped = {}
        error = None
        before = set(context.ledger.issues)
        try:
            skipped = detector["function"](context, columns) or {}
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
            complete = False
        seconds = round(time.perf_counter() - start, 4)
        timings[detector["name"]] = seconds
        found = [issue for key, issue in context.ledger.issues.items() if key not in before]
        if log and error:
            logger.info(f"  {detector['code']} {detector['name']}: FAILED ({error})")
        elif log and found:
            logger.debug(f"  {detector['code']} {detector['name']}: {len(found)} issue(s)")
            for issue in found:
                logger.debug(f"      {issue_text(issue)}")
        elif log:
            reason = f" (skipped: {skipped['*']})" if "*" in skipped else ""
            logger.debug(f"  {detector['code']} {detector['name']}: nothing found{reason}")
        checked = 0 if "*" in skipped else len(columns) - len(skipped) if detector["applies_to"] else context.raw.shape[1]
        coverage.append(
            CoverageEntry(
                detector=detector["name"],
                code=detector["code"],
                checked=checked,
                skipped=skipped,
                error=error,
                seconds=seconds,
            )
        )
    return coverage, timings, complete
