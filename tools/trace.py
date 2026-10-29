import json
import time
import logging
from pathlib import Path
from tools.traceview import TraceView
from tools.tracehtml import write_html
from tools.runfolder import now_text, to_json_value

VIEWS = ["compact", "detailed", "full"]
STAGE_TEXT = {
    "read": "reads the file as a raw text view",
    "glance": "profiles every column: types, missing tokens, value patterns",
    "frame": "decides the target, the task, the metric and the id columns",
    "split": "locks the test rows and builds the cross-validation folds",
    "audit": "checks the training rows for data problems (code only)",
    "leakguard": "finds ids and columns known only after the event",
    "plan": "writes the cleaning plan from the catalog of tested operations",
    "compile": "tests every step on the folds and repairs failing steps",
    "gate": "keeps only the steps that do not hurt the cross-validated score",
    "model": "chooses model families and searches their settings",
    "test": "scores the model once on the locked test rows",
    "report": "writes scores.json and report.md",
}

# shortens a value for a one-line view (lists and dicts as compact JSON)
def short(value, limit=80):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=to_json_value)
    return text if len(text) <= limit else text[: limit - 3] + "..."

# the columns a step works on, as text
def step_columns(params):
    columns = params.get("columns") or ([params["column"]] if params.get("column") else [])
    return ", ".join(map(str, columns))

# the rows of a step table: group, operation, columns, other parameters,...
# reason and source
def step_rows(plan):
    rows = []
    for group in ["steps", "row_steps", "target_steps"]:
        for step in plan.get(group, []):
            params = {key: value for key, value in step.get("params", {}).items() if key not in ("column", "columns")}
            rows.append([group.replace("_steps", "").replace("steps", "column"), step.get("op", ""), step_columns(step.get("params", {})), short(params, 60) if params else "", step.get("reason", ""), step.get("source", "")])
    return rows

# forwards WARNING and ERROR records of the automl loggers to the trace (not...
# those of the LLM client: its calls already show their errors)
class TraceLogHandler(logging.Handler):

    # keeps the trace that receives the records
    def __init__(self, trace):
        super().__init__(level=logging.WARNING)
        self.trace = trace

    # one warning event per record
    def emit(self, record):
        if record.name.startswith("automl") and record.name != "automl.llm":
            self.trace.emit("warning", record.getMessage().splitlines()[0][:200], {"logger": record.name, "message": record.getMessage()})

# the event stream of one run: every event goes to trace.jsonl (full data) and...
# to the live view; decisions, LLM calls and errors arrive through the run...
# folder, the stage events and the stage details from the orchestrator
class Trace:

    # opens the trace of a run folder; view is None for no live output
    def __init__(self, folder, view=None, console=None):
        self.folder = folder
        self.started = time.monotonic()
        self.stage = None
        self.events = []
        self.usage = {"calls": 0, "tokens": 0, "cost_usd": 0.0}
        self.status = {"stage": "starting", "remaining": None}
        self.last_plan = None
        self.view = TraceView(view, console, live_status=self.status_text) if view is not None else None
        self.log_handler = TraceLogHandler(self)
        logging.getLogger("automl").addHandler(self.log_handler)

    # adds one event: stored in memory, appended to trace.jsonl and shown live
    def emit(self, kind, title, data=None, stage=None):
        event = {"time": now_text(), "elapsed": round(time.monotonic() - self.started, 2), "stage": stage or self.stage, "type": kind, "title": title, "data": data or {}}
        self.events.append(event)
        with self.folder.file("trace.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=to_json_value) + "\n")
        if self.view is not None:
            self.view.show(event)
        return event

    # the text of the live status line: stage, time and LLM usage so far
    def status_text(self):
        remaining = self.status["remaining"]
        left = f" | {remaining:.0f} s left" if remaining is not None else ""
        return f"{self.status['stage']} | {time.monotonic() - self.started:.0f} s{left} | LLM {self.usage['calls']} calls, ${self.usage['cost_usd']:.4f}"

    # receives the records the run folder writes and turns them into events
    def listen(self, name, record):
        if name == "llm.jsonl":
            self.usage["calls"] += 0 if record.get("cached") else 1
            self.usage["tokens"] += record.get("input_tokens", 0) + record.get("output_tokens", 0)
            self.usage["cost_usd"] += record.get("cost_usd", 0.0)
            prompt = record.get("prompt", "?")
            state = "cached" if record.get("cached") else ("ok" if record.get("ok") else "invalid" if record.get("answer") else "failed")
            self.emit("llm_call", f"{prompt} ({state})", record, stage=record.get("stage"))
        elif name == "decisions.jsonl":
            self.emit("decision", f"{record.get('kind')}: {record.get('subject')}", record, stage=record.get("stage"))
        elif name == "errors.jsonl":
            self.emit("error", record.get("error", "error"), record, stage=record.get("stage"))

    # the event at the start of a stage
    def stage_start(self, name, number, remaining):
        self.stage = name
        self.status.update(stage=f"{number:02d} {name}", remaining=remaining)
        self.emit("stage_start", name, {"number": number, "text": STAGE_TEXT.get(name, "")})

    # the event at the end of a stage, then its details
    def stage_end(self, name, status, note, state=None):
        self.emit("stage_end", note, {"ok": status.ok, "seconds": status.seconds, "error": status.error, "fallback": status.fallback, "note": note})
        if state is not None:
            try:
                for title, data in stage_details(name, state):
                    self.emit("detail", *self.plan_change(title, data) if data.get("plan") else (title, data))
            except Exception as error:
                self.emit("warning", f"stage details failed: {type(error).__name__}: {error}")

    # a plan detail after the first one shows only what changed: "unchanged",...
    # or the steps that were removed and added since the last plan shown
    def plan_change(self, title, data):
        rows, previous = data["table"]["rows"], self.last_plan
        self.last_plan = rows
        if previous is None:
            return title, data
        old, new = {tuple(row[:4]) for row in previous}, {tuple(row[:4]) for row in rows}
        if old == new:
            return f"{title.split(':')[0]}: unchanged ({len(rows)} steps)", {"lines": []}
        changes = [["removed"] + row[:5] for row in previous if tuple(row[:4]) not in new] + [["added"] + row[:5] for row in rows if tuple(row[:4]) not in old]
        table = {"columns": ["change", "group", "operation", "columns", "parameters", "reason"], "rows": changes}
        return f"{title.split(':')[0]}: {len(rows)} steps ({len(changes)} changed)", {"table": table, "full_plan": data["table"]}

    # the first event of a run
    def run_start(self, data):
        self.emit("run_start", data.get("run_id", "run"), data, stage="run")

    # the last event of a run; closes the live view and writes trace.txt and...
    # trace.html (only with a live view, so benchmark runs write trace.jsonl only)
    def close(self, data):
        self.stage = "run"
        self.emit("run_end", data.get("outcome", "done"), dict(data, usage=self.usage), stage="run")
        logging.getLogger("automl").removeHandler(self.log_handler)
        if self.view is not None:
            self.view.stop()
            self.folder.file("trace.txt").write_text(self.view.export_text(), encoding="utf-8", newline="\n")
            write_html(self.events, self.folder.file("trace.html"))

# the detail events of one finished stage, read from the run state: (title,...
# data) pairs with "lines" or a "table"
def stage_details(name, state):
    objects, context = state.objects, state.context
    if name == "read" and "read" in objects:
        report = objects["read"]
        lines = [f"{key}: {short(value)}" for key, value in report.items() if key in ("format", "encoding", "delimiter", "rows", "columns", "skipped_lines", "notes") and value not in (None, [], "")]
        return [("file", {"lines": lines})]
    if name == "glance" and objects.get("glance", {}).get("columns"):
        columns = objects["glance"]["columns"]
        types = {}
        for column in columns:
            types[column.get("type", "?")] = types.get(column.get("type", "?"), 0) + 1
        rows = [[column["name"], column.get("type", ""), f"{column.get('missing_share', 0):.0%}" if isinstance(column.get("missing_share"), (int, float)) else "", short(column.get("examples", column.get("top", "")), 50)] for column in columns[:40]]
        return [("column types", {"lines": [", ".join(f"{count} {kind}" for kind, count in sorted(types.items(), key=lambda item: -item[1]))]}),
                ("columns", {"table": {"columns": ["column", "type", "missing", "examples"], "rows": rows}})]
    if name == "audit" and objects.get("report") is not None:
        issues = objects["report"].issues
        rows = [[issue.id, issue.priority, issue.title, ", ".join(issue.columns[:4]), issue.diagnosis] for issue in issues]
        return [(f"{len(issues)} issues found", {"table": {"columns": ["id", "priority", "issue", "columns", "diagnosis"], "rows": rows}})] if issues else [("no issues found", {"lines": []})]
    if name in ("plan", "compile", "gate") and objects.get("plan") is not None:
        plan = (context.final_plan if name == "gate" and context.final_plan is not None else objects["plan"]).model_dump()
        title = {"plan": "cleaning plan", "compile": "plan after the data checks", "gate": "final plan (kept by the gate)"}[name]
        details = [(f"{title}: {len(step_rows(plan))} steps", {"plan": True, "table": {"columns": ["group", "operation", "columns", "parameters", "reason", "source"], "rows": step_rows(plan)}})]
        if name == "gate" and context.gate:
            gate = context.gate
            details.insert(0, ("scores", {"lines": [f"P0 (protected steps only) {gate['p0']['score']:.4f} -> final {gate['final_score']:.4f} ({context.task.metric}, {gate.get('rounds', 1)} round(s))"]}))
        return details
    if name == "model" and context.model:
        model = context.model
        rows = [[family, f"{score:.4f}" if isinstance(score, (int, float)) else str(score)] for family, score in model.get("cv", {}).items()]
        lines = [f"chosen: {' + '.join(model.get('chosen', []))}", f"CV score: {model['cv_score']:.4f}" if isinstance(model.get("cv_score"), (int, float)) else "CV score: n/a"]
        return [("model search", {"lines": lines}), ("cross-validated score per family", {"table": {"columns": ["family", f"CV {context.task.metric}"], "rows": rows}})]
    if name == "test" and context.scores.get("test"):
        test = context.scores["test"]
        lines = [f"test {test['metric']}: {test['test']:.4f} (dummy {test['dummy_test']:.4f})"]
        pairs = (test.get("importance") or {}).get("columns", [])
        top = sorted(((column, value) for column, value in pairs if isinstance(value, (int, float))), key=lambda item: -item[1])[:8]
        details = [("test score", {"lines": lines})]
        if top:
            details.append(("most important columns (permutation)", {"table": {"columns": ["column", "importance"], "rows": [[column, f"{value:.4f}"] for column, value in top]}}))
        return details
    if name == "report":
        return [("files", {"lines": [str(state.folder.file("report.md")), str(state.folder.file("trace.html"))]})]
    return []

# replays trace.jsonl of a run folder on screen (view None: no replay) and...
# rewrites trace.html; returns its path, or None without a trace
def replay(run_dir, view="detailed", console=None):
    path = Path(run_dir) / "trace.jsonl"
    if not path.exists():
        return None
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if view is not None:
        screen = TraceView(view, console)
        for event in events:
            screen.show(event)
    return write_html(events, Path(run_dir) / "trace.html")
