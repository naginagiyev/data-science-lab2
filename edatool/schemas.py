from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

# allowed values of the main fields; the fix actions are the contract with the fixing step of a pipeline
FixAction = Literal[
    "drop_column",
    "drop_rows",
    "convert_type",
    "replace_values",
    "impute",
    "add_indicator",
    "normalize_labels",
    "group_rare",
    "cap_values",
    "log_transform",
    "encode",
    "extract_datetime",
    "extract_pattern",
    "set_role",
    "split_strategy",
    "no_action",
]
SemanticType = Literal[
    "numeric_continuous",
    "numeric_discrete",
    "numeric_code",
    "categorical_nominal",
    "categorical_ordinal",
    "boolean",
    "datetime",
    "text",
    "id",
    "constant",
    "empty",
]
Role = Literal["feature", "target", "id", "group_key", "time_key", "drop"]
Task = Literal["binary-classification", "multi-class-classification", "regression"]
Priority = Literal["must_fix", "should_fix", "consider"]
Status = Literal["open", "confirmed", "dismissed", "accepted", "resolved", "still_open"]

# a recommended fix: one action from the fix vocabulary and its parameters
class Fix(BaseModel):
    action: FixAction
    params: dict[str, Any] = {}

# the rows of an issue: how many, which share of the data, and up to 5...
# example row ids
class RowSet(BaseModel):
    count: int
    share: float
    examples: list[int] = Field(default_factory=list, max_length=5)

# one data problem with its evidence, priority, review status, diagnosis, fix...
# and links
class Issue(BaseModel):
    id: str
    fingerprint: str
    code: str
    title: str
    columns: list[str] = []
    rows: RowSet | None = None
    evidence: dict[str, Any] = {}
    detector: str
    priority: Priority
    needs_review: bool = False
    status: Status = "open"
    diagnosis: str | None = None
    fix: Fix | None = None
    alternatives: list[Fix] = []
    caused_by: str | None = None
    children: list[str] = []
    created_by_step: str | None = None
    level: int = 0
    source: Literal["detector", "agent"] = "detector"
    tool_calls: list[str] = []
    unresolved: str | None = None

# a link between two issues: caused_by, same_rows or group
class Edge(BaseModel):
    source: str
    target: str
    kind: Literal["caused_by", "same_rows", "group"]
    reason: str | None = None

# the semantic type, role and type confidence of one column
class ColumnInfo(BaseModel):
    name: str
    type: SemanticType
    role: Role = "feature"
    confidence: Literal["high", "low"] = "high"

# what one detector did: how many columns it checked, what it skipped and if...
# it failed
class CoverageEntry(BaseModel):
    detector: str
    code: str
    checked: int = 0
    skipped: dict[str, str] = {}
    error: str | None = None
    seconds: float = 0.0

# information about the run: versions, hashes, data size, agent model and if...
# the run is complete
class Meta(BaseModel):
    schema_version: str = "1.0"
    tool_version: str
    settings_hash: str
    dataset_hash: str
    dataset: str
    rows: int
    columns: int
    target: str
    task: Task
    sample_rows: int | None = None
    raw_input: bool
    llm_model: str | None = None
    llm_tokens: dict[str, int] | None = None
    agent_sessions: int | None = None
    smolagents_version: str | None = None
    prompt_hash: str | None = None
    run_seconds: float = 0.0
    complete: bool = False
    created_at: str
    verify_level: int | None = None
    loop_finished: bool | None = None

# the recommended train/test split strategy, with the split-risk issues...
# behind it
class SplitAdvice(BaseModel):
    strategy: Literal["random", "stratified", "group", "time"]
    key: str | None = None
    shuffle: bool = True
    reason: str
    issues: list[Issue] = []

# the full result of a run; the schema_ field is saved as 'schema' in JSON
class Report(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    meta: Meta
    schema_: list[ColumnInfo] = Field(default_factory=list, alias="schema")
    issues: list[Issue] = []
    edges: list[Edge] = []
    coverage: list[CoverageEntry] = []
