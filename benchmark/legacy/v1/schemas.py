from typing import Literal
from pydantic import BaseModel

class ProjectConfig(BaseModel):
    desiredMetric: str | None = None
    minScoreRequirement: float | None = None
    explainableModel: bool | None = None

class TargetDetectionResult(BaseModel):
    targetCol: str

class ReasoningResult(BaseModel):
    condition: Literal["fail", "pass"]
    prompt: str | None = None
