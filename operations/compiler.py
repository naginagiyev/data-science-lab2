import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.pipeline import Pipeline
from operations.rows import RowFilter
from operations.generic import GenericStage, TypingStep
from operations.target import LogTarget, MapTargetLabels
from sklearn.base import BaseEstimator, TransformerMixin
from operations.registry import OPERATIONS, build, operation_class

# a readable name of a plan step for logs and checks
def step_name(position, step):
    column = step.params.get("column") or ", ".join(step.params.get("columns", [])[:2]) or step.params.get("name", "")
    return f"step {position + 1}: {step.op}" + (f" {column}" if column else "")

# checks a plan in order and tracks the columns that exist after each step:...
# unknown operations, bad parameters, unknown columns, steps on the target and...
# operations outside their kind are errors; returns a list of error texts
def validate_plan(plan, columns, target, task):
    errors = []
    present = [column for column in columns if column != target]
    for group, kind in [(plan.steps, "column"), (plan.row_steps, "row"), (plan.target_steps, "target")]:
        for position, step in enumerate(group):
            label = f"{kind} step {position + 1} ({step.op})"
            if step.op not in OPERATIONS:
                errors.append(f"{label}: unknown operation; use one of the catalog")
                continue
            if operation_class(step.op).kind != kind:
                errors.append(f"{label}: {step.op} is a {operation_class(step.op).kind} operation, not a {kind} operation")
                continue
            try:
                operation = build(step)
            except ValidationError as error:
                problems = "; ".join(f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in error.errors()[:5])
                errors.append(f"{label}: bad parameters: {problems}")
                continue
            if kind == "target":
                if step.op == "map_target_labels" and task == "regression":
                    errors.append(f"{label}: map_target_labels is for classification only")
                if step.op == "log_target" and task != "regression":
                    errors.append(f"{label}: log_target is for regression only")
                continue
            if kind == "row":
                continue
            used = operation.inputs()
            if target in used or target in operation.outputs():
                errors.append(f"{label}: never touch the target column `{target}` with a column operation")
                continue
            unknown = [column for column in used if column not in present]
            if unknown:
                errors.append(f"{label}: unknown column(s) {unknown} at this point (dropped or never created)")
                continue
            clash = [column for column in operation.outputs() if column in present and column not in used]
            if clash:
                errors.append(f"{label}: output column(s) {clash} already exist; choose new names")
                continue
            present = [column for column in present if column not in operation.removes()]
            present += [column for column in operation.outputs() if column not in present]
    return errors

# the agent steps of a plan as one transformer: the operations run in order,...
# each fitted on the output of the previous one (training rows only)
class AgentSteps(BaseEstimator, TransformerMixin):
    def __init__(self, steps=None):
        self.steps = steps

    def fit(self, frame, y=None):
        self.operations_ = []
        current = frame
        for step in self.steps or []:
            operation = build(step)
            current = operation.fit(current, y).transform(current)
            self.operations_.append(operation)
        return self

    def transform(self, frame):
        current = frame
        for operation in self.operations_:
            current = operation.transform(current)
        return current

    def snapshots(self, frame):
        current = frame
        result = []
        for position, (step, operation) in enumerate(zip(self.steps or [], self.operations_)):
            current = operation.transform(current)
            result.append((step_name(position, step), current))
        return result

# the target operations of a plan: label spellings for classification and a...
# log transform for regression (applied by the engine, scored on the original...
# scale)
class TargetSteps:
    def __init__(self, steps):
        self.mapping = None
        self.transform = None
        for step in steps:
            operation = build(step)
            if isinstance(operation, MapTargetLabels):
                self.mapping = operation
            elif isinstance(operation, LogTarget):
                self.transform = operation

    def labels(self, values):
        return self.mapping.map(values) if self.mapping is not None else np.asarray(pd.Series(values).astype(str))

# the feature pipeline of a plan and a model kind: agent steps, typing step...
# and generic stage
def feature_pipeline(plan, kind, seed=0):
    steps = plan.steps if plan is not None else []
    return Pipeline([("agent", AgentSteps(list(steps))), ("typing", TypingStep()), ("generic", GenericStage(kind, seed))])

# the training-only row filter of a plan, or None
def row_filter(plan):
    if plan is None or not plan.row_steps:
        return None
    return RowFilter([build(step) for step in plan.row_steps])

# the target steps of a plan
def target_steps(plan):
    return TargetSteps(plan.target_steps if plan is not None else [])
