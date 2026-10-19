import json
import hashlib
from pathlib import Path
from paths import settings_file
from pydantic import BaseModel, ConfigDict

# base class of every settings section: unknown keys are errors, so a typo...
# in an override never passes silently
class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")

# test share of the demo split, number of CV folds and the minimum number of...
# labeled rows
class SplitSettings(Section):
    test_size: float
    cv_folds: int
    min_rows: int

# time budget of one run: total, the minimum left for the model and the...
# limit of each stage
class BudgetSettings(Section):
    total_seconds: float
    min_model_seconds: float
    stage_seconds: dict[str, float]

# model, sampling, retries, cache and per-run limits of the LLM client
class LlmSettings(Section):
    provider: str
    model: str
    pilot_model: str
    temperature: float
    seed: int
    timeout_seconds: float
    api_retries: int
    schema_retries: int
    cache: bool
    max_calls_per_run: int
    max_cost_per_run_usd: float
    prices_per_million_tokens: dict[str, dict[str, float]]

# global LLM spend caps of the whole plan and the category of this process
class SpendSettings(Section):
    category: str
    dev_cap_usd: float
    experiments_cap_usd: float
    stop_optional_usd: float
    total_cap_usd: float

# options of the framer: number of target candidates shown to the LLM, the...
# confidence under which --ask asks the user, and the target name hints
class FramerSettings(Section):
    candidates: int
    min_confidence: float
    max_target_levels: int
    name_hints: list[str]

# options of the audit stage: the row sample of the slower checks and the...
# parallel jobs of the leak check
class AuditSettings(Section):
    sample_rows: int
    n_jobs: int

# options of the leak guard
class LeakageSettings(Section):
    drop_semantic_only: bool

# options of the LLM cleaner: on or off, value samples, repair and refine...
# rounds
class CleanerSettings(Section):
    enabled: bool
    show_values: bool
    max_values_per_column: int
    max_repairs: int
    max_refines: int

# limits of the deterministic data checks of compiled operations
class ChecksSettings(Section):
    max_parse_failure: float
    max_column_growth: float
    op_seconds: float

# options of the validation gate: folds, sample size, per-operation tests...
# and the tolerance of each metric
class GateSettings(Section):
    enabled: bool
    folds: int
    max_rows: int
    per_op: bool
    max_per_op_tests: int
    n_jobs: int
    tolerance: dict[str, dict[str, float]]

# options of the own model engine
class EngineSettings(Section):
    candidates: list[str]
    optuna: bool
    ensemble: bool
    n_jobs: int
    patience_trials: int
    max_search_rows: int

# options of the report stage
class ReportSettings(Section):
    llm_summary: bool

# all settings of one run (configuration/settings.json plus overrides)
class Settings(Section):
    seed: int
    split: SplitSettings
    budget: BudgetSettings
    llm: LlmSettings
    spend: SpendSettings
    framer: FramerSettings
    audit: AuditSettings
    leakage: LeakageSettings
    cleaner: CleanerSettings
    checks: ChecksSettings
    gate: GateSettings
    engine: EngineSettings
    backend: str
    report: ReportSettings

# reads a JSON file into a python object
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

# merges the overrides into the base settings, also inside nested...
# dictionaries
def deep_merge(base, overrides):
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

# turns the text of a --set value into a python value: JSON when it parses...
# (numbers, true, false, null, lists), otherwise the plain text
def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text

# applies one "a.b.c=value" override to a settings dictionary; the keys must...
# already exist, so a typo is an error
def apply_dotted(values, assignment):
    if "=" not in assignment:
        raise ValueError(f"Override '{assignment}' must look like key.subkey=value.")
    key_text, value_text = assignment.split("=", 1)
    keys = key_text.strip().split(".")
    node = values
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            raise ValueError(f"Unknown settings key '{key_text}'.")
        node = node[key]
    if keys[-1] not in node:
        raise ValueError(f"Unknown settings key '{key_text}'.")
    node[keys[-1]] = parse_value(value_text.strip())
    return values

# loads the default settings, merges an optional JSON file and the dotted...
# overrides, and validates the result
def load_settings(overrides_file=None, assignments=None, overrides=None):
    values = read_json(settings_file)
    if overrides_file is not None:
        values = deep_merge(values, read_json(overrides_file))
    if overrides:
        values = deep_merge(values, overrides)
    for assignment in assignments or []:
        values = apply_dotted(values, assignment)
    return Settings.model_validate(values)

# returns a hash of the final settings, so every run folder shows which...
# settings produced it
def settings_hash(settings):
    canonical = json.dumps(settings.model_dump(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
