import json
import time
import hashlib
import smolagents
from pathlib import Path
from edatool.schemas import Fix
from edatool.paths import prompts_dir
from smolagents.utils import AgentError
from edatool.core.enrichment import clusters
from edatool.agent.models import build_model
from concurrent.futures import ThreadPoolExecutor
from smolagents import LogLevel, ToolCallingAgent
from edatool.agent.gate import make_gate, make_guard
from edatool.agent.actions import cluster_actions, dataset_actions
from edatool.agent.tools import CallCounter, SessionState, read_tools
from edatool.core.progress import detail_line, logger, plural, stage_line

# issue codes that the dataset session reviews instead of a cluster session
dataset_codes = {"F02"}

# reads an agent prompt file
def prompt_text(name):
    return (prompts_dir / f"{name}.md").read_text(encoding="utf-8")

# hash of the prompts and tool descriptions; it is part of the cache key and...
# of the report meta
def prompt_hash():
    text = prompt_text("reviewer") + prompt_text("semantic")
    text += "".join(tool.description for tool in read_tools + cluster_actions + dataset_actions)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# one line per issue for the task text of the agent
def issue_line(issue):
    evidence = json.dumps({key: value for key, value in issue.evidence.items()}, default=str)[:400]
    rows = f" · {issue.rows.count} rows" if issue.rows else ""
    fix = f" · default fix {issue.fix.action} {json.dumps(issue.fix.params, default=str)}" if issue.fix else ""
    return f"- {issue.id} [{issue.priority}] {issue.code} {issue.title} · columns {issue.columns}{rows} · needs_review={issue.needs_review} · evidence {evidence}{fix}"

# one line per column for the task text of the agent
def schema_line(column):
    return f"- {column.name}: {column.type} (role {column.role}, confidence {column.confidence})"

# one line about the dataset for the task text of the agent
def dataset_header(context):
    name = context.raw.attrs.get("edatool", {}).get("source", "dataframe")
    return f"Dataset: {name}, {context.n_rows} rows, {context.raw.shape[1]} columns, target `{context.target}`, task {context.task}."

# builds the task text of a cluster session: dataset, schema, issues and...
# review items
def cluster_task(state):
    context = state.context
    columns = sorted({column for issue in state.issues.values() for column in issue.columns})
    schema = [schema_line(column) for column in context.schema if column.name in columns]
    lines = [
        f"Session: {state.key}",
        dataset_header(context),
        "",
        "Schema of the columns involved:",
        *schema,
        "",
        "Issues in this cluster:",
        *[issue_line(issue) for issue in state.issues.values()],
        "",
        f"Review items you must close: {', '.join(state.review_ids)}",
    ]
    return "\n".join(lines)

# builds the task text of the dataset session: every column with its type,...
# role and examples
def dataset_task(state):
    context = state.context
    hide = state.settings["agent"]["hide_examples"]
    lines = [f"Session: {state.key}", dataset_header(context), "", "Columns (name: type, role, confidence, 3 example values):"]
    for column in context.schema:
        examples = [] if hide else context.raw[column.name].astype(str).drop_duplicates().head(3).tolist()
        lines.append(f"- {column.name}: {column.type}, role {column.role}, confidence {column.confidence}, examples {examples}")
    if state.issues:
        lines += ["", "Review items:", *[issue_line(issue) for issue in state.issues.values()]]
    lines += ["", "Open items: " + "; ".join(state.open_items())]
    return "\n".join(lines)

# creates one session per cluster with review items, plus one dataset session
def build_sessions(context, counter, settings):
    issues = context.ledger.issue_list()
    by_id = {issue.id: issue for issue in issues}
    sessions = []
    for group in clusters(issues, context.ledger.edges):
        members = {issue_id: by_id[issue_id] for issue_id in group if by_id[issue_id].code not in dataset_codes}
        review = [issue_id for issue_id, issue in members.items() if issue.needs_review and issue.status == "open"]
        if review:
            key = f"cluster-{len(sessions) + 1}"
            sessions.append(SessionState(key, "cluster", context, members, review, counter, settings))
    dataset_issues = {issue.id: issue for issue in issues if issue.code in dataset_codes}
    review = [issue_id for issue_id, issue in dataset_issues.items() if issue.needs_review]
    sessions.append(SessionState("dataset", "dataset", context, dataset_issues, review, counter, settings))
    return sessions

# cache key of a session: dataset hash, issues, prompt hash, model and gate...
# flag
def session_cache_key(state, dataset_hash, model_id):
    fingerprints = sorted(issue.fingerprint for issue in state.issues.values())
    use_gate = state.settings["agent"].get("use_gate", True)
    text = json.dumps([dataset_hash, state.kind, fingerprints, prompt_hash(), model_id, use_gate])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

# reads the cached decisions of a session, if they exist
def load_cached(cache_dir, key):
    if cache_dir is None:
        return None
    path = Path(cache_dir) / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

# saves the decisions of a session that passed the done gate
def save_cached(cache_dir, key, state):
    if cache_dir is None or not state.gate_passed:
        return
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fingerprint_of = {issue_id: issue.fingerprint for issue_id, issue in state.issues.items()}
    content = {
        "decisions": {fingerprint_of[issue_id]: decision for issue_id, decision in state.decisions.items()},
        "links": [{**link, "child": fingerprint_of[link["child"]], "parent": fingerprint_of[link["parent"]]} for link in state.links],
        "added": state.added,
        "types": state.types,
        "roles": state.roles,
        "names": state.names,
        "tokens": state.tokens,
    }
    (Path(cache_dir) / f"{key}.json").write_text(json.dumps(content, default=str) + "\n", encoding="utf-8")

# applies cached decisions to a session without calling the model
def apply_cached(state, cached):
    by_fingerprint = {issue.fingerprint: issue_id for issue_id, issue in state.issues.items()}
    state.decisions = {by_fingerprint[key]: value for key, value in cached["decisions"].items() if key in by_fingerprint}
    state.links = [
        {**link, "child": by_fingerprint[link["child"]], "parent": by_fingerprint[link["parent"]]}
        for link in cached["links"]
        if link["child"] in by_fingerprint and link["parent"] in by_fingerprint
    ]
    state.added = cached["added"]
    state.types = cached["types"]
    state.roles = cached["roles"]
    state.names = cached["names"]
    state.tokens = cached.get("tokens", state.tokens)
    state.gate_passed = True
    state.cached = True

# runs one agent session with the tools, the done gate and the guard; uses...
# the cache when possible
def run_session(state, model, settings, dataset_hash):
    agent_settings = settings["agent"]
    cache_key = session_cache_key(state, dataset_hash, getattr(model, "model_id", "model"))
    cached = load_cached(agent_settings.get("cache_dir"), cache_key)
    if state.kind == "dataset":
        goal = f"review the meaning of {len(state.context.schema)} column names"
        if state.review_ids:
            goal += f" and {', '.join(state.review_ids)}"
    else:
        goal = "review " + ", ".join(f"{issue_id} ({state.issues[issue_id].title.lower()})" for issue_id in state.review_ids)
    logger.debug(f"  [{state.key}] start: {goal}")
    if cached is not None:
        apply_cached(state, cached)
        logger.debug(f"  [{state.key}] reused cached decisions from an earlier run (no API call)")
        state.cached = True
        log_session_end(state)
        return state
    is_dataset = state.kind == "dataset"
    actions = dataset_actions if is_dataset else cluster_actions
    tools = [tool(state) for tool in read_tools + actions]
    gate = make_gate(state)
    agent = ToolCallingAgent(
        tools=tools,
        model=model,
        instructions=prompt_text("semantic" if is_dataset else "reviewer"),
        final_answer_checks=[gate] if agent_settings.get("use_gate", True) else [],
        step_callbacks=[make_guard(state, settings)],
        max_steps=agent_settings["max_steps"],
        max_tool_threads=agent_settings["max_tool_threads"],
        verbosity_level=LogLevel.OFF,
    )
    try:
        agent.run(dataset_task(state) if is_dataset else cluster_task(state))
    except AgentError as error:
        state.failure = state.failure or f"agent error: {error}"
    except Exception as error:
        state.failure = f"{type(error).__name__}: {error}"
    if not agent_settings.get("use_gate", True) and state.failure is None:
        try:
            gate(None, None)
        except ValueError:
            pass
    if state.failure is None and agent_settings.get("use_gate", True) and not state.gate_passed:
        state.failure = "max steps reached without passing the gate"
    save_cached(agent_settings.get("cache_dir"), cache_key, state)
    log_session_end(state)
    return state

# logs the decisions of a session: one short line on screen, the details in...
# the log file
def log_session_end(state):
    for issue_id, decision in state.decisions.items():
        fix = decision.get("fix")
        fix_text = f" -> fix {fix['action']}" if fix else ""
        logger.debug(f"  [{state.key}] decision {issue_id}: {decision['status']}{fix_text} | {decision.get('diagnosis')}")
    if state.names:
        suspected = [name for name, verdict in state.names.items() if verdict["verdict"] == "suspected"]
        logger.debug(f"  [{state.key}] column names: {len(state.names) - len(suspected)} clean, {len(suspected)} suspected" + (f" ({', '.join(suspected)})" if suspected else ""))
    status = "gate passed" if state.gate_passed and not state.failure else f"FAILED ({state.failure})"
    logger.debug(f"  [{state.key}] end: {status} | {state.steps} steps | {state.tokens['input'] + state.tokens['output']:,} tokens")
    decisions = []
    for issue_id, decision in state.decisions.items():
        fix = decision.get("fix")
        arrow = f" → {fix['action']}" if fix and decision["status"] == "confirmed" else ""
        decisions.append(f"{issue_id:<11} {decision['status']}{arrow}")
    if state.names:
        suspected = [name for name, verdict in state.names.items() if verdict["verdict"] == "suspected"]
        decisions.append(f"{'names':<11} {len(state.names) - len(suspected)} clean" + (f", suspected: {', '.join(suspected)}" if suspected else ""))
    extra = "cached" if getattr(state, "cached", False) else f"{state.steps} steps"
    if state.gate_failures:
        extra += f", gate ✗{state.gate_failures}"
    lines = [detail_line(text, state.key if index == 0 else "", extra if index == 0 else "") for index, text in enumerate(decisions or ["no decisions"])]
    if state.failure:
        lines.append(detail_line(f"✗ stopped: {state.failure}"))
    logger.info("\n".join(lines))

# copies the decisions of a session into the main ledger and schema; open...
# items of a failed session are marked unresolved
def merge(context, state):
    ledger = context.ledger
    for issue_id, decision in state.decisions.items():
        issue = ledger.issues[issue_id]
        issue.status = decision["status"]
        issue.diagnosis = decision.get("diagnosis") or issue.diagnosis
        if decision.get("fix"):
            issue.fix = Fix(**decision["fix"])
        if decision.get("alternatives"):
            issue.alternatives = [Fix(**item) for item in decision["alternatives"]]
        issue.tool_calls = decision.get("calls") or []
    for link in state.links:
        ledger.link(link["child"], link["parent"], "caused_by", link["reason"])
        ledger.issues[link["child"]].tool_calls += [call for call in link["calls"] if call not in ledger.issues[link["child"]].tool_calls]
    ledger.detector = "agent review"
    for item in state.added:
        issue = ledger.add(
            item["code"],
            item["columns"],
            f"agent:{item['title']}",
            {"source": "agent review"},
            title=item["title"],
            priority=item["priority"],
            diagnosis=item["diagnosis"],
            fix_action=item["fix"]["action"],
            fix_params=item["fix"]["params"],
            needs_review=False,
        )
        issue.source = "agent"
        issue.tool_calls = item["calls"]
    for name, verdict in state.names.items():
        if verdict["verdict"] == "suspected":
            issue = ledger.add("G03", [name], "name_review", {"reason": verdict["reason"]}, diagnosis=verdict["reason"], needs_review=False)
            issue.source = "agent"
    schema = {column.name: column for column in context.schema}
    for name, semantic_type in state.types.items():
        schema[name].type = semantic_type
        schema[name].confidence = "high"
    for name, role in state.roles.items():
        schema[name].role = role
    if state.failure:
        for issue_id in state.review_ids:
            if issue_id not in state.decisions:
                ledger.issues[issue_id].unresolved = state.failure

# runs all agent sessions in parallel, merges them in a fixed order and...
# returns a summary and the trace
def review(context, settings, model=None):
    model = model or build_model(settings)
    dataset_hash = context.dataset_hash()
    counter = CallCounter()
    sessions = build_sessions(context, counter, settings)
    workers = max(1, settings["agent"]["max_parallel_sessions"])
    clusters_count = len(sessions) - 1
    logger.info("")
    logger.info(stage_line("Agent review", f"{getattr(model, 'model_id', type(model).__name__)} · {plural(clusters_count, 'issue cluster')} + column names"))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        finished = list(pool.map(lambda state: run_session(state, model, settings, dataset_hash), sessions))
    for state in finished:
        merge(context, state)
    total = sum(state.tokens["input"] + state.tokens["output"] for state in finished)
    added = [issue for issue in context.ledger.issue_list() if issue.source == "agent"]
    for issue in added:
        logger.info(detail_line(f"{issue.id:<11} {issue.title.lower()} ({', '.join(issue.columns)})", "added"))
    passed = sum(state.gate_passed and not state.failure for state in finished)
    mark = "✓" if passed == len(finished) else "✗"
    logger.info(detail_line(f"{mark} {passed}/{len(finished)} sessions passed the done gate · {total:,} tokens", "", f"{time.perf_counter() - started:6.1f} s"))
    summary = {
        "complete": all(state.gate_passed and state.failure is None for state in finished),
        "sessions": [
            {
                "key": state.key,
                "review_items": len(state.review_ids),
                "gate_passed": state.gate_passed,
                "failure": state.failure,
                "steps": state.steps,
                "tokens": state.tokens,
                "cached": getattr(state, "cached", False),
            }
            for state in finished
        ],
        "tokens": {
            "input": sum(state.tokens["input"] for state in finished),
            "output": sum(state.tokens["output"] for state in finished),
        },
        "model": getattr(model, "model_id", type(model).__name__),
        "smolagents_version": smolagents.__version__,
        "prompt_hash": prompt_hash(),
    }
    return summary, counter.trace
