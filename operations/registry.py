import json
from operations.text import TextFeatures
from operations.custom import CustomOperation
from operations.target import LogTarget, MapTargetLabels
from operations.categories import GroupRare, MapCategories
from operations.rows import DropConflictingRows, DropDuplicateRows, DropRowsWhere
from operations.parsing import ParseBoolean, ParseDatetime, ParseNumber, ParseUnits
from operations.values import ClipValues, ImputeConstant, LogTransform, ReplaceValues, SetMissingWhere
from operations.columns import AddMissingIndicator, AsCategory, DateDifference, DropColumns, ExtractPattern

COLUMN_OPERATIONS = [
    DropColumns, ReplaceValues, SetMissingWhere, ParseNumber, ParseUnits, ParseDatetime, DateDifference, ParseBoolean,
    MapCategories, GroupRare, AsCategory, ExtractPattern, AddMissingIndicator, ImputeConstant, ClipValues, LogTransform,
    TextFeatures,
]
ROW_OPERATIONS = [DropDuplicateRows, DropConflictingRows, DropRowsWhere]
TARGET_OPERATIONS = [MapTargetLabels, LogTarget]
OPERATIONS = {operation.name: operation for operation in COLUMN_OPERATIONS + ROW_OPERATIONS + TARGET_OPERATIONS}
OPERATIONS[CustomOperation.name] = CustomOperation

# registers more operations (row, target and custom operations add themselves)
def register(*operations):
    for operation in operations:
        OPERATIONS[operation.name] = operation

# the class of an operation by name, or a KeyError with the known names
def operation_class(name):
    if name not in OPERATIONS:
        raise KeyError(f"Unknown operation '{name}'. Known operations: {', '.join(OPERATIONS)}")
    return OPERATIONS[name]

# builds an unfitted operation from a plan step; the parameters are validated...
# with the operation's own parameter model
def build(step):
    operation = operation_class(step.op)(dict(step.params))
    operation.config
    return operation

# one catalog entry for prompts: name, kind, purpose, a JSON example and the...
# edatool fix actions it implements
def entry_text(operation):
    example = json.dumps(operation.example, ensure_ascii=False)
    actions = ", ".join(operation.actions) or "-"
    return f"- {operation.name} ({operation.kind}): {operation.purpose}. params_json example: {example}. edatool fixes: {actions}."

# the catalog text injected into prompts (prompts never list operations by hand)
def catalog_text(names=None):
    chosen = [OPERATIONS[name] for name in (names or OPERATIONS) if name in OPERATIONS]
    return "\n".join(entry_text(operation) for operation in chosen)
