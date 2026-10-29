import io
import json
from rich import box
from rich.live import Live
from rich.rule import Rule
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.padding import Padding
from rich.spinner import Spinner
from rich.console import Console, Group

SOURCE_STYLE = {"llm": "magenta", "rules": "cyan", "checks": "yellow", "gate": "blue", "user": "green"}
REASON_KEYS = ("reason", "reasons", "rationale", "explanation", "notes", "summary", "why")
PROMPT_LINES = 8
TABLE_ROWS = 15

# the live status line: a spinner and a text that is computed at every refresh
class StatusLine:

    # keeps the function that gives the status text
    def __init__(self, text):
        self.text = text

    # the renderable of the current status
    def __rich__(self):
        return Columns([Spinner("dots", style="cyan"), Text(self.text(), style="dim")])

# indents a renderable under its event line
def indent(renderable, left=4):
    return Padding(renderable, (0, 0, 0, left))

# the prompt the model saw: the last user message (later messages are retries)
def prompt_text(record):
    messages = [message for message in record.get("messages") or [] if message.get("role") == "user"]
    return messages[-1]["content"] if messages else ""

# the answer of an LLM call as parsed JSON, or None when it is not JSON
def parsed_answer(record):
    try:
        return json.loads(record.get("answer") or "")
    except (ValueError, TypeError):
        return None

# a rich table from column names and rows; long tables are cut in the...
# detailed view
def make_table(columns, rows, limit=None):
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    for index, name in enumerate(columns):
        table.add_column(name, overflow="fold", max_width=48 if index == len(columns) - 1 or name in ("reason", "diagnosis") else 34)
    shown = rows if limit is None else rows[:limit]
    for row in shown:
        table.add_row(*[str(value) for value in row])
    if limit is not None and len(rows) > limit:
        table.add_row(*([f"... {len(rows) - limit} more"] + [""] * (len(columns) - 1)), style="dim")
    return table

# the parameters of a step in an LLM answer (a dict, or the JSON text of...
# params_json as the cleaner writes it)
def step_params(step):
    if not isinstance(step, dict):
        return {}
    if isinstance(step.get("params"), dict):
        return step["params"]
    try:
        params = json.loads(step.get("params_json") or "{}")
    except (ValueError, TypeError):
        return {}
    return params if isinstance(params, dict) else {}

# the step table of a plan answer (operation, columns, reason)
def answer_steps(steps):
    rows = []
    for step in steps:
        params = step_params(step)
        columns = params.get("columns") or ([params["column"]] if params.get("column") else [])
        rows.append([step.get("op", "?") if isinstance(step, dict) else str(step), ", ".join(map(str, columns)), step.get("reason", "") if isinstance(step, dict) else ""])
    return rows

# the live terminal view of a trace: one renderable per event, printed above...
# a live status line; a second console records the same output for trace.txt
class TraceView:

    # view is compact, detailed or full; live_status turns on the status line
    def __init__(self, view="detailed", console=None, live_status=None):
        self.view = view
        self.console = console or Console(highlight=False)
        self.recorder = Console(file=io.StringIO(), record=True, width=120, color_system=None, highlight=False)
        self.current = (0, "")
        self.live = None
        if live_status is not None and self.console.is_terminal:
            self.live = Live(StatusLine(live_status), console=self.console, refresh_per_second=4, transient=True)
            self.live.start()

    # prints a renderable on the screen and in the recorder
    def out(self, renderable):
        self.console.print(renderable)
        self.recorder.print(renderable)

    # shows one event (the compact view shows only the run and stage lines,...
    # warnings and errors)
    def show(self, event):
        kind = event["type"]
        if self.view == "compact" and kind not in ("run_start", "stage_start", "stage_end", "error", "run_end"):
            return
        renderer = getattr(self, f"show_{kind}", None)
        if renderer is not None:
            renderer(event)

    # the run header: data, variant, mode, budget and model
    def show_run_start(self, event):
        data = event["data"]
        lines = Text()
        for label, key in [("data", "data"), ("variant", "variant"), ("mode", "mode"), ("budget", "budget"), ("LLM", "model"), ("run folder", "folder")]:
            if data.get(key) not in (None, ""):
                lines.append(f"{label:<11}", style="dim").append(f"{data[key]}\n")
        self.out(Panel(lines, title="[bold]Agentic AutoML v2[/]", subtitle=data.get("run_id", ""), border_style="cyan", expand=False))

    # a rule with the stage number and name, and what the stage does
    def show_stage_start(self, event):
        number = event["data"].get("number", 0)
        self.current = (number, event["title"])
        if self.view == "compact":
            return
        self.out(Rule(Text(f" {number:02d}  {event['title']} ", style="bold cyan"), align="left", style="cyan"))
        if event["data"].get("text"):
            self.out(indent(Text(event["data"]["text"], style="dim italic"), 2))

    # the result line of a stage: note and seconds, or the error and fallback
    def show_stage_end(self, event):
        data = event["data"]
        number, name = self.current
        prefix = f"  {number:02d} {name:<10} " if self.view == "compact" else "  "
        if data.get("ok"):
            self.out(Text(prefix).append("✓ ", style="bold green").append(data.get("note") or "ok").append(f"   {data.get('seconds', 0):.1f} s", style="dim"))
        else:
            self.out(Text(prefix).append("✗ ", style="bold red").append(f"{data.get('error')}", style="red").append(f" -> {data.get('fallback') or 'no fallback'}").append(f"   {data.get('seconds', 0):.1f} s", style="dim"))

    # one decision: accepted or refused, kind, subject, reason and its source
    def show_decision(self, event):
        data = event["data"]
        icon, style = ("✓", "green") if data.get("accepted") else ("✗", "red")
        source = data.get("source", "")
        line = Text("  ").append(f"{icon} ", style=f"bold {style}").append(f"{data.get('kind')} ", style="dim").append(f"{data.get('subject')}", style="bold")
        line.append(f"  {data.get('reason', '')}").append(f"  [{source}]", style=SOURCE_STYLE.get(source, "white"))
        self.out(line)
        if self.view == "full":
            extra = {key: data[key] for key in ("evidence", "proposal") if data.get(key)}
            if extra:
                self.out(indent(Text(json.dumps(extra, ensure_ascii=False, default=str)[:2000], style="dim"), 6))

    # one LLM call: header with model, tokens, cost and time, the prompt (cut...
    # in the detailed view), the answer and, for an invalid answer, the errors
    def show_llm_call(self, event):
        record = event["data"]
        header = Text("  ").append("◆ LLM ", style="bold magenta").append(record.get("prompt", "?"), style="bold")
        details = f"  {record.get('model', '')} · {record.get('input_tokens', 0):,} in / {record.get('output_tokens', 0):,} out tokens · ${record.get('cost_usd', 0.0):.4f} · {record.get('seconds', 0.0):.1f} s"
        header.append(details, style="dim")
        if record.get("cached"):
            header.append("  cached", style="cyan")
        if record.get("attempt"):
            header.append(f"  retry {record['attempt']}", style="yellow")
        parts = [self.prompt_block(record)]
        answer = parsed_answer(record)
        if answer is not None:
            parts.append(self.answer_block(answer))
        elif record.get("answer"):
            parts.append(Text("answer  ", style="bold").append(record["answer"][:600]))
        if record.get("error"):
            parts.append(Text("✗ ", style="bold red").append(record["error"][:800], style="red"))
        self.out(header)
        self.out(indent(Group(*parts), 6))

    # the prompt text: the first lines in the detailed view, everything in full
    def prompt_block(self, record):
        lines = [line for line in prompt_text(record).splitlines() if line.strip()]
        shown = lines if self.view == "full" else lines[:PROMPT_LINES]
        text = Text("prompt  ", style="bold").append("\n        ".join(line[:160] for line in shown), style="dim")
        if len(shown) < len(lines):
            text.append(f"\n        ... {len(lines) - len(shown)} more lines (full prompt in trace.html)", style="dim italic")
        return text

    # the answer: a step table for plans, the reasons the model wrote, and the...
    # other short fields
    def answer_block(self, answer):
        parts = []
        if isinstance(answer, dict):
            for key, value in answer.items():
                if isinstance(value, list) and value and isinstance(value[0], dict) and "op" in value[0]:
                    parts.append(Text(f"answer  {key}: {len(value)} steps", style="bold"))
                    parts.append(make_table(["operation", "columns", "reason (written by the LLM)"], answer_steps(value), None if self.view == "full" else TABLE_ROWS))
                elif key.lower() in REASON_KEYS or key.lower().endswith("reason"):
                    parts.append(Text("reasoning  ", style="bold magenta").append(str(value)[:1200], style="italic"))
                else:
                    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
                    parts.append(Text(f"{key}  ", style="bold").append(text if self.view == "full" else text[:300]))
        else:
            parts.append(Text("answer  ", style="bold").append(json.dumps(answer, ensure_ascii=False)[:600]))
        return Group(*parts) if parts else Text("answer  (empty)", style="dim")

    # a stage detail: a title with lines or a table
    def show_detail(self, event):
        data = event["data"]
        self.out(Text("  ▸ ", style="cyan").append(event["title"], style="bold"))
        if data.get("table"):
            table = data["table"]
            self.out(indent(make_table(table["columns"], table["rows"], None if self.view == "full" else TABLE_ROWS), 4))
        for line in data.get("lines", []):
            self.out(indent(Text(str(line)), 4))

    # a warning from the pipeline's logger
    def show_warning(self, event):
        self.out(Text("  ⚑ ", style="bold yellow").append(event["title"], style="yellow"))

    # an error of a stage (the stage line shows the fallback)
    def show_error(self, event):
        data = event["data"]
        body = Text(data.get("error", event["title"]), style="red")
        if self.view == "full" and data.get("traceback"):
            body.append("\n" + data["traceback"][-1500:], style="dim")
        self.out(indent(Panel(body, title=f"error in {data.get('stage', event['stage'])}", border_style="red", expand=False), 2))

    # the summary at the end: outcome, time, scores, LLM usage and the files
    def show_run_end(self, event):
        data = event["data"]
        usage = data.get("usage", {})
        lines = Text()
        rows = [("outcome", data.get("outcome")), ("time", f"{data.get('seconds', 0):.1f} s"), ("CV score", data.get("cv")), ("test score", data.get("test")),
                ("LLM", f"{usage.get('calls', 0)} calls, {usage.get('tokens', 0):,} tokens, ${usage.get('cost_usd', 0.0):.4f}"), ("trace", data.get("trace"))]
        for label, value in rows:
            if value not in (None, ""):
                lines.append(f"{label:<11}", style="dim").append(f"{value}\n")
        style = "green" if data.get("outcome") == "done" else "red"
        self.out(Panel(lines, title="[bold]run finished[/]", border_style=style, expand=False))

    # stops the live status line
    def stop(self):
        if self.live is not None:
            self.live.stop()
            self.live = None

    # the plain text of everything shown (for trace.txt)
    def export_text(self):
        return self.recorder.export_text()
