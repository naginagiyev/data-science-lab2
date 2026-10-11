import re
import pandas as pd
from schemas import TaskSpec
from tools.metrics import primary_metric

EXCLUDED_TYPES = {"id", "constant", "empty", "text"}

# splits a column name into lowercase words (camelCase and separators)
def name_words(name):
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [word for word in re.split(r"[^a-z0-9]+", spaced.lower()) if word]

# proposes the task from the target values: 2 values -> binary; text with 3+...
# values -> multi-class; numbers with decimals or more than max_levels values ->...
# regression; integers with 3 to max_levels levels -> ambiguous (multi-class...
# by default, the LLM decides by meaning); returns (task, ambiguous, reason)
def rule_task(values, max_levels=20):
    text = pd.Series(values).astype(str).str.strip()
    text = text[text != ""]
    levels = text.nunique()
    if levels <= 2:
        return "binary-classification", False, f"{levels} distinct values"
    numbers = pd.to_numeric(text, errors="coerce")
    if numbers.isna().mean() > 0.05:
        return "multi-class-classification", False, f"text with {levels} distinct values"
    numbers = numbers.dropna()
    if (numbers != numbers.round()).any():
        return "regression", False, "numbers with decimals"
    if levels > max_levels:
        return "regression", False, f"integers with {levels} distinct values"
    return "multi-class-classification", True, f"integers with only {levels} levels (an ordered score or a category?)"

# ranks the target candidates from the glance: name hints, last-column bonus,...
# never an id, constant, empty or long-text column; returns (name, score, reason)
def rank_candidates(glance, hints):
    columns = glance["columns"]
    hint_set = set(hints)
    ranked = []
    for position, entry in enumerate(columns):
        if entry.get("type") in EXCLUDED_TYPES:
            continue
        words = name_words(entry["name"])
        matches = [word for word in words if word in hint_set]
        score = 3.0 * bool(matches) + (1.0 if position == len(columns) - 1 else 0.0)
        score += 0.5 * (entry.get("unique", 0) <= 20)
        reason = []
        if matches:
            reason.append(f"name hint '{matches[0]}'")
        if position == len(columns) - 1:
            reason.append("last column")
        ranked.append((entry["name"], score, ", ".join(reason) or "no hint"))
    return sorted(ranked, key=lambda item: -item[1])

# the columns the glance marks as ids
def id_columns(glance, target):
    return [entry["name"] for entry in glance["columns"] if entry.get("type") == "id" and entry["name"] != target]

# frames the task by rules only: the given target or the best candidate, the...
# given task or the rule task; the metric always comes from the task
def frame_by_rules(raw, glance, settings, target=None, task=None):
    source = "flags" if target and task else "rules"
    reason = []
    if not target:
        ranked = rank_candidates(glance, settings.framer.name_hints)
        target = ranked[0][0] if ranked else raw.columns[-1]
        reason.append(f"target '{target}' from rules ({ranked[0][2] if ranked else 'last column'})")
    ambiguous = False
    if not task:
        task, ambiguous, why = rule_task(raw[target], settings.framer.max_target_levels)
        reason.append(f"task {task} from rules ({why})")
    confidence = 1.0 if source == "flags" else (0.5 if ambiguous else 0.8)
    return TaskSpec(
        target=target, task=task, metric=primary_metric(task), id_columns=id_columns(glance, target),
        confidence=confidence, reason="; ".join(reason) or "given by flags", source=source,
    )
