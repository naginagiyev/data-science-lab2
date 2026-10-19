import re
import sys
import json
import logging
import secrets
import traceback
from pathlib import Path
from datetime import datetime, timezone

ASCII_SYMBOLS = {"✓": "ok", "✗": "x", "·": "-", "→": "->", "─": "-", "└": "`", "…": "..."}

# makes the console accept any text: UTF-8 with replacement, so a cp1252...
# console never raises UnicodeEncodeError
def prepare_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

# turns text into plain ASCII for the console: known symbols get a readable...
# replacement, everything else becomes "?"
def ascii_text(text):
    for symbol, replacement in ASCII_SYMBOLS.items():
        text = text.replace(symbol, replacement)
    return text.encode("ascii", "replace").decode("ascii")

# a log formatter that writes only ASCII, used for the console
class AsciiFormatter(logging.Formatter):
    def format(self, record):
        return ascii_text(super().format(record))

# returns the current UTC time as an ISO text
def now_text():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# returns a short, safe name for a dataset path (lowercase letters, digits, -)
def dataset_name(data_path):
    stem = Path(str(data_path)).stem.lower()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-")[:40] or "data"

# turns any object into something JSON can store (paths, sets, numpy values)
def to_json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)

# one run folder: creates it, writes JSON and JSONL files, saves the context...
# and records errors
class RunFolder:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, data_path, out_dir):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(Path(out_dir) / f"{stamp}-{dataset_name(data_path)}-{secrets.token_hex(3)}")

    def file(self, name):
        return self.path / name

    def write_json(self, name, data):
        text = json.dumps(data, indent=2, ensure_ascii=False, default=to_json_value)
        self.file(name).write_text(text + "\n", encoding="utf-8", newline="\n")

    def read_json(self, name):
        return json.loads(self.file(name).read_text(encoding="utf-8"))

    def append(self, name, record):
        line = json.dumps(record, ensure_ascii=False, default=to_json_value)
        with self.file(name).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    def read_lines(self, name):
        path = self.file(name)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def save_context(self, context):
        self.file("context.json").write_text(context.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")

    def log_error(self, stage, error):
        details = "".join(traceback.format_exception(error))
        self.append("errors.jsonl", {"time": now_text(), "stage": stage, "error": f"{type(error).__name__}: {error}", "traceback": details})
        logging.getLogger("automl").error(f"{stage}: {type(error).__name__}: {error}\n{details}")

# sends every log record to run.log (UTF-8, full details) and short ASCII lines...
# to the console; returns the handlers so the run can close them
def setup_logging(folder, verbose=False, console=True):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(folder.file("run.log"), encoding="utf-8", errors="replace")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    handlers = [file_handler]
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_handler.setFormatter(AsciiFormatter("%(message)s"))
        console_handler.addFilter(lambda record: record.name == "automl" or verbose)
        handlers.append(console_handler)
    for handler in handlers:
        root.addHandler(handler)
    for name in ["httpx", "openai", "urllib3", "LiteLLM"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    return handlers

# removes and closes the handlers of one run
def close_logging(handlers):
    root = logging.getLogger()
    for handler in handlers:
        root.removeHandler(handler)
        handler.close()
