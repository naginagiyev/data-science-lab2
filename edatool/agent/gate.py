import time
from edatool.core.progress import logger, short

# finds columns with conflicting fixes (drop_column with other fixes, or two...
# type conversions)
def fix_conflicts(state):
    fixes = {}
    for issue_id, issue in state.issues.items():
        decision = state.decisions.get(issue_id)
        if decision is not None and decision["status"] != "confirmed":
            continue
        fix = decision["fix"] if decision else (issue.fix.model_dump() if issue.fix else None)
        if fix is None or fix["action"] == "no_action":
            continue
        column = fix["params"].get("column") or (issue.columns[0] if len(issue.columns) == 1 else None)
        if column:
            fixes.setdefault(column, []).append((issue_id, fix))
    problems = []
    for column, items in fixes.items():
        actions = [fix["action"] for _, fix in items]
        if "drop_column" in actions and len(set(actions)) > 1:
            ids = [issue_id for issue_id, _ in items]
            problems.append(f"conflicting fixes on {column}: drop_column together with other fixes ({', '.join(ids)})")
        targets = {fix["params"].get("to") for _, fix in items if fix["action"] == "convert_type" and fix["params"].get("to")}
        if len(targets) > 1:
            problems.append(f"conflicting type conversions on {column}: {sorted(targets)}")
    return problems

# builds the done gate: when the agent calls final_answer, open items or...
# invalid decisions raise an error, so the agent must continue
def make_gate(state):
    def done_gate(final_answer, memory, agent=None):
        problems = [f"open item: {item}" for item in state.open_items()]
        for issue_id, decision in state.decisions.items():
            if decision["status"] == "confirmed" and not decision.get("diagnosis"):
                problems.append(f"{issue_id} is confirmed without a diagnosis")
            for call in decision.get("calls") or []:
                if call not in state.calls:
                    problems.append(f"{issue_id} cites unknown call {call}")
        problems += fix_conflicts(state)
        if problems:
            state.gate_failures += 1
            logger.debug(f"  [{state.key}] ! done gate failed, agent must continue: {short('; '.join(problems), 300)}")
            raise ValueError("Done gate failed. Fix these items, then call final_answer again: " + "; ".join(problems))
        logger.debug(f"  [{state.key}] done gate passed")
        state.gate_passed = True
        return True
    return done_gate

# builds the step callback: counts tokens, stops a session at its token cap,...
# its wall-clock limit or the audit deadline, reminds an idle agent after...
# guard_k steps and stops it after twice that many
def make_guard(state, settings):
    agent_settings = settings["agent"]
    patience = agent_settings["guard_k"]
    token_cap = agent_settings["max_tokens_per_run"]
    session_seconds = agent_settings.get("session_seconds")
    deadline = agent_settings.get("deadline")
    started = time.monotonic()
    progress = {"changes": 0, "idle": 0}
    def guard(step, agent=None):
        state.steps += 1
        thought = getattr(step, "model_output", None)
        if isinstance(thought, str) and thought.strip() and not thought.strip().startswith(("{", "[")):
            logger.debug(f"  [{state.key}] thinking: {short(thought.strip(), 300)}")
        usage = getattr(step, "token_usage", None)
        if usage is not None:
            state.tokens["input"] += usage.input_tokens
            state.tokens["output"] += usage.output_tokens
        if state.tokens["input"] + state.tokens["output"] > token_cap:
            state.failure = "token cap reached"
            logger.info(f"  [{state.key}] ! token cap reached, stopping the session")
            agent.interrupt()
            return
        if (session_seconds and time.monotonic() - started > session_seconds) or (deadline and time.time() > deadline):
            state.failure = "session time limit reached"
            logger.info(f"  [{state.key}] ! time limit reached, stopping the session")
            agent.interrupt()
            return
        if state.changes != progress["changes"]:
            progress["changes"] = state.changes
            progress["idle"] = 0
            return
        progress["idle"] += 1
        if progress["idle"] >= 2 * patience:
            state.failure = "agent stuck"
            logger.info(f"  [{state.key}] ! no progress for {progress['idle']} steps, stopping the session")
            agent.interrupt()
        elif progress["idle"] >= patience:
            items = state.open_items()
            logger.debug(f"  [{state.key}] ! no progress for {progress['idle']} steps, reminding the agent of the open items")
            state.reminder = f"No progress for {progress['idle']} steps. Open items: {'; '.join(items[:10]) or 'none, call final_answer'}."
    return guard
