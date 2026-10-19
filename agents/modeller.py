from typing import Literal
from pydantic import BaseModel
from collections import Counter
from tools.columntypes import CATEGORY_TYPES

# one model family and its share of the search time
class FamilyShare(BaseModel):
    family: Literal["linear", "lightgbm", "catboost"]
    share: float

# the modeller's answer: the families to search with their time shares
class ModelAnswer(BaseModel):
    families: list[FamilyShare]
    reason: str

# the facts of the prompt: task, metric, size, column types, wide categories,...
# classes and the run budget (fixed, so the answer can be cached)
def profile_text(glance, train, task, budget):
    columns = [entry for entry in glance.get("columns", []) if entry["name"] != task.target]
    types = Counter(entry.get("type", "?") for entry in columns)
    wide = sum(entry.get("type") in CATEGORY_TYPES and entry.get("unique", 0) > 50 for entry in columns)
    lines = [
        f"Task {task.task}, metric {task.metric}.",
        f"{len(train)} training rows, {len(columns)} feature columns before preparation.",
        "Column types: " + ", ".join(f"{name} {count}" for name, count in types.most_common()) + ".",
        f"Categorical columns with more than 50 levels: {wide}.",
        f"Time budget of the whole run: {budget:.0f} seconds.",
    ]
    if task.task != "regression":
        shares = train[task.target].astype(str).str.strip().value_counts(normalize=True)
        lines.append(f"{len(shares)} classes; the smallest has {shares.iloc[-1]:.1%} of the rows.")
    return "\n".join(lines)

# the problems of an answer: no boosting family, a family twice, a share of 0...
# or less
def answer_problems(answer):
    names = [item.family for item in answer.families]
    problems = []
    if not {"lightgbm", "catboost"} & set(names):
        problems.append("choose at least one of lightgbm and catboost")
    if len(set(names)) != len(names):
        problems.append("name each family only once")
    if any(item.share <= 0 for item in answer.families):
        problems.append("every share must be above 0")
    return problems

# asks the LLM which model families to search and how to split the search...
# time (prompts/modeller.md); returns the normalized shares and the reason,...
# or None (the engine defaults stay)
def choose_models(client, context, glance, train, task, budget):
    answer = client.ask("modeller", {"profile": profile_text(glance, train, task, budget)}, ModelAnswer, "model", context, check=answer_problems)
    if answer is None:
        return None
    total = sum(item.share for item in answer.families)
    shares = {item.family: round(item.share / total, 4) for item in answer.families}
    return {"shares": shares, "reason": answer.reason}
