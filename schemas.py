from pydantic import BaseModel
from typing import Any, Literal

Task = Literal["binary-classification", "multi-class-classification", "regression"]

# short task names that the command line accepts, mapped once to the full names
TASK_ALIASES = {
    "binary": "binary-classification",
    "multiclass": "multi-class-classification",
    "regression": "regression",
}

# raised for the few hard stops of §3.1 (unreadable file, no rows, one class...
# ...); the message is shown to the user and the run ends without a model
class StopRun(Exception):
    pass

# what to predict: target, task, metric, id and time columns, and who decided
class TaskSpec(BaseModel):
    target: str
    task: Task
    metric: Literal["roc_auc", "log_loss", "rmse"]
    id_columns: list[str] = []
    time_column: str | None = None
    confidence: float = 1.0
    reason: str = ""
    source: Literal["flags", "rules", "llm"] = "rules"

# how the rows were split: strategy, key, test row positions and CV folds
class SplitInfo(BaseModel):
    strategy: Literal["random", "stratified", "group", "time"]
    key: str | None = None
    test_rows: list[int] = []
    folds: list[list[int]] = []
    reason: str = ""
    warnings: list[str] = []

# one operation of a plan with its parameters, the issues it fixes and its origin
class Step(BaseModel):
    op: str
    params: dict[str, Any]
    issues: list[str] = []
    reason: str = ""
    protected: bool = False
    source: Literal["leakguard", "translator", "llm", "repair", "refine", "custom"] = "translator"

# a data-preparation plan: column steps, training-only row steps and target steps
class Plan(BaseModel):
    version: int = 0
    steps: list[Step] = []
    row_steps: list[Step] = []
    target_steps: list[Step] = []
    notes: str = ""

# one decision of a run: what was proposed, the evidence, and if it was accepted
class Decision(BaseModel):
    stage: str
    kind: str
    subject: str
    proposal: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    accepted: bool
    reason: str
    source: Literal["rules", "llm", "edatool", "gate", "checks", "user"]

# the outcome of one stage: success, time, fallback and error
class StageStatus(BaseModel):
    name: str
    ok: bool
    seconds: float
    fallback: str | None = None
    error: str | None = None

# the state of one run, passed between the stages and saved after each stage
class RunContext(BaseModel):
    run_id: str
    mode: Literal["demo", "benchmark"]
    variant: str
    data_path: str
    settings_hash: str
    task: TaskSpec | None = None
    split: SplitInfo | None = None
    audit_dir: str | None = None
    leaks: list[dict[str, Any]] = []
    plans: list[Plan] = []
    final_plan: Plan | None = None
    gate: dict[str, Any] = {}
    model: dict[str, Any] = {}
    scores: dict[str, Any] = {}
    stages: list[StageStatus] = []
    llm: dict[str, float] = {}
