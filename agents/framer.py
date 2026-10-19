import re
import questionary
import pandas as pd
from typing import Literal
from schemas import TaskSpec
from pydantic import BaseModel
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

# ranks the target candidates from the glance: name hints (earlier hints in the...
# list weigh more), last-column bonus, a small bonus for few levels; never an...
# id, constant, empty or long-text column; returns (name, score, reason)
def rank_candidates(glance, hints):
    columns = glance["columns"]
    hint_set = set(hints)
    ranked = []
    for position, entry in enumerate(columns):
        if entry.get("type") in EXCLUDED_TYPES:
            continue
        words = name_words(entry["name"])
        matches = [word for word in words if word in hint_set]
        strength = max((len(hints) - hints.index(word)) / len(hints) for word in matches) if matches else 0.0
        score = (3.0 + strength) * bool(matches) + (1.0 if position == len(columns) - 1 else 0.0)
        score += 0.2 * (entry.get("unique", 0) <= 20)
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

# the framer's answer (strict structured output)
class FramerAnswer(BaseModel):
    target: str
    task: Literal["binary-classification", "multi-class-classification", "regression"]
    confidence: float
    reason: str
    id_columns: list[str]
    time_column: str | None

ID_WORDS = {"id", "key", "code", "number", "no", "uuid", "guid", "index", "nr"}

# true when a column can be an id: typed as id by the glance, or an id-like...
# name with at least 90% unique values
def id_like(raw, glance, column):
    entry = next((item for item in glance["columns"] if item["name"] == column), {})
    if entry.get("type") == "id":
        return True
    values = raw[column].astype(str)
    return bool(ID_WORDS & set(name_words(column))) and values.nunique() >= 0.9 * len(values)

# the checks of a framer answer against the table; returns error texts
def answer_problems(answer, raw, candidates, glance=None):
    problems = []
    if answer.target not in candidates:
        problems.append(f"target '{answer.target}' is not in the candidate list {candidates}")
    unknown = [column for column in answer.id_columns + ([answer.time_column] if answer.time_column else []) if column not in raw.columns]
    if unknown:
        problems.append(f"unknown columns {unknown}; use exact column names")
    if answer.target in answer.id_columns:
        problems.append("the target cannot be an id column")
    not_ids = [column for column in answer.id_columns if column in raw.columns and glance is not None and not id_like(raw, glance, column)]
    if not_ids:
        problems.append(f"{not_ids} are not ids (their values repeat or the name is not an identifier); list only columns that identify rows")
    values = raw[answer.target].astype(str).str.strip() if answer.target in raw.columns else pd.Series(dtype=str)
    if answer.task != "binary-classification" and values[values != ""].nunique() == 2:
        problems.append("the target has exactly two values, so the task is binary-classification")
    if not 0 <= answer.confidence <= 1:
        problems.append("confidence must be between 0 and 1")
    return problems

# frames the task: flags win; otherwise rules propose and, when an LLM client...
# is given and the target is unknown or the rule task is ambiguous, the LLM...
# decides among the rule candidates; an invalid or missing answer leaves the...
# rule result; with ask, a low confidence is confirmed by the user
def frame_task(raw, glance, settings, target=None, task=None, client=None, context=None, ask=False, glance_text=""):
    rules = frame_by_rules(raw, glance, settings, target, task)
    if target and task:
        return rules
    ambiguous = task is None and rule_task(raw[rules.target], settings.framer.max_target_levels)[1]
    if client is None or (target and not ambiguous):
        return confirm(rules, raw, settings, ask)
    ranked = [name for name, _, _ in rank_candidates(glance, settings.framer.name_hints)][: settings.framer.candidates]
    candidates = [target] if target else ranked or [rules.target]
    lines = [f"{position + 1}. {name}" for position, name in enumerate(candidates)]
    inputs = {"profile": glance_text, "candidates": "\n".join(lines), "rule_task": f"{rules.task} for `{rules.target}` ({rules.reason})"}
    answer = client.ask("framer", inputs, FramerAnswer, "frame", context, check=lambda answer: answer_problems(answer, raw, candidates, glance))
    if answer is None:
        return confirm(rules, raw, settings, ask)
    ids = sorted(set(answer.id_columns) | set(id_columns(glance, answer.target)))
    framed = TaskSpec(
        target=answer.target, task=task or answer.task, metric=primary_metric(task or answer.task), id_columns=[column for column in ids if column != answer.target],
        time_column=answer.time_column, confidence=answer.confidence, reason=answer.reason, source="llm",
    )
    return confirm(framed, raw, settings, ask)

# asks the user to confirm target and task when the confidence is low (only...
# in the interactive --ask mode)
def confirm(spec, raw, settings, ask):
    if not ask or spec.confidence >= settings.framer.min_confidence:
        return spec
    target = questionary.select(f"Target column (framer suggests {spec.target}):", choices=list(map(str, raw.columns)), default=spec.target).ask() or spec.target
    choices = ["binary-classification", "multi-class-classification", "regression"]
    chosen = questionary.select("Task:", choices=choices, default=spec.task if target == spec.target else None).ask() or spec.task
    return spec.model_copy(update={"target": target, "task": chosen, "metric": primary_metric(chosen), "confidence": 1.0, "source": "flags", "reason": "confirmed by the user"})
