import json
import pytest
from pydantic import ValidationError
from settings import apply_dotted, load_settings, settings_hash

# the default settings file loads and has the values of the build plan
def test_defaults_load():
    settings = load_settings()
    assert settings.llm.model == "gpt-4.1-nano"
    assert settings.spend.total_cap_usd == 5.0
    assert settings.gate.tolerance["roc_auc"]["absolute"] == 0.002
    assert settings.budget.stage_seconds["audit"] == 90

# dotted overrides change booleans, numbers, text and nested values
def test_dotted_overrides():
    settings = load_settings(assignments=["gate.enabled=false", "budget.total_seconds=120", "llm.model=gpt-4.1-mini", "budget.stage_seconds.audit=30"])
    assert settings.gate.enabled is False
    assert settings.budget.total_seconds == 120
    assert settings.llm.model == "gpt-4.1-mini"
    assert settings.budget.stage_seconds["audit"] == 30

# a JSON file is deep merged: keys it does not name keep their default
def test_file_override_deep_merge(tmp_path):
    override_file = tmp_path / "override.json"
    override_file.write_text(json.dumps({"gate": {"folds": 5}, "seed": 3}), encoding="utf-8")
    settings = load_settings(overrides_file=override_file)
    assert settings.gate.folds == 5
    assert settings.gate.enabled is True
    assert settings.seed == 3

# unknown keys are errors, both in dotted overrides and in override files
def test_unknown_keys_fail(tmp_path):
    with pytest.raises(ValueError):
        load_settings(assignments=["gate.enabeld=false"])
    with pytest.raises(ValueError):
        apply_dotted({"gate": {"enabled": True}}, "gate.enabled")
    override_file = tmp_path / "override.json"
    override_file.write_text(json.dumps({"gate": {"unknown": 1}}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(overrides_file=override_file)

# the hash is stable for equal settings and changes when a value changes
def test_settings_hash():
    first = settings_hash(load_settings())
    assert first == settings_hash(load_settings())
    assert first != settings_hash(load_settings(assignments=["seed=1"]))
