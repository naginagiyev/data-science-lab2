import time
import numpy as np
import pandas as pd
from types import SimpleNamespace
from edatool.api import build_context
from edatool.agent.gate import make_guard
from edatool.core.enrichment import enrich
from edatool.agent.tools import CallCounter
from edatool.core.registry import run_detectors
from edatool.agent.session import accept_anonymized, build_sessions

# the context of a deterministic audit of a frame (detectors and links only)
def audited(frame, target, task, overrides=None):
    context = build_context(frame, target, task, overrides, None, {})
    run_detectors(context, "audit")
    enrich(context)
    return context

# a frame where every feature has one extreme isolated value (D02 review items)
def outlier_frame(names, rows=300, seed=0):
    rng = np.random.default_rng(seed)
    frame = {}
    for position, name in enumerate(names):
        values = rng.normal(50, 5, rows)
        values[7 * position + 3] = 5000
        frame[name] = [f"{value:.3f}" for value in values]
    frame["y"] = rng.choice(["a", "b"], rows)
    return pd.DataFrame(frame)

# review items on anonymized names (V1, V2, ...) are accepted without the agent
def test_anonymized_accepted():
    context = audited(outlier_frame(["V1", "V2", "price"]), "y", "binary-classification")
    accepted = accept_anonymized(context, context.settings)
    d02 = {issue.columns[0]: issue for issue in context.ledger.issue_list() if issue.code == "D02"}
    assert {"V1", "V2"} <= set(d02) and d02["V1"].status == "accepted" and d02["V1"].id in accepted
    assert d02["price"].status == "open"

# review items of the same code share sessions of at most batch_size items,...
# and sessions above max_sessions are not run (their items stay open)
def test_batches_and_cap():
    names = [f"amount_{letter}" for letter in "abcdefghijklmnopqrst"]
    context = audited(outlier_frame(names), "y", "binary-classification")
    settings = context.settings
    settings["agent"].update(batch_size=8, max_sessions=8)
    sessions = build_sessions(context, CallCounter(), settings)
    clusters = [state for state in sessions if state.kind == "cluster"]
    assert len(clusters) == 3 and max(len(state.review_ids) for state in clusters) <= 8
    assert sessions[-1].kind == "dataset"
    settings["agent"].update(max_sessions=2)
    capped = build_sessions(context, CallCounter(), settings)
    assert len([state for state in capped if state.kind == "cluster"]) == 1
    assert sum(issue.unresolved == "agent session cap reached" for issue in context.ledger.issue_list()) >= 8

# the guard stops a session at its wall-clock limit
def test_guard_time_limit():
    state = SimpleNamespace(steps=0, tokens={"input": 0, "output": 0}, failure=None, changes=0, key="cluster-1", open_items=lambda: [])
    settings = {"agent": {"guard_k": 6, "max_tokens_per_run": 10**9, "session_seconds": 0.01, "deadline": None}}
    guard = make_guard(state, settings)
    time.sleep(0.05)
    stopped = []
    guard(SimpleNamespace(model_output=None, token_usage=None), SimpleNamespace(interrupt=lambda: stopped.append(True)))
    assert stopped and state.failure == "session time limit reached"

# rare classes and target outliers are reported with no_action (the target...
# label space never changes) and need no review
def test_target_policy():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": [str(value) for value in rng.normal(size=300)], "y": ["a"] * 150 + ["b"] * 145 + ["c"] * 5})
    context = audited(frame, "y", "multi-class-classification")
    rare = [issue for issue in context.ledger.issue_list() if issue.code == "F05"]
    assert rare and rare[0].fix.action == "no_action" and not rare[0].needs_review
