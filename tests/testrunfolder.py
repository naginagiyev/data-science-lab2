import io
import logging
import numpy as np
from schemas import RunContext
from tools.runfolder import AsciiFormatter, RunFolder, ascii_text, close_logging, dataset_name, setup_logging

# the run folder name has the time, a safe dataset name and 6 hex digits
def test_create_folder(tmp_path):
    folder = RunFolder.create("data/My Data.csv", tmp_path)
    assert folder.path.exists()
    parts = folder.path.name.split("-")
    assert len(parts[0]) == 8 and len(parts[1]) == 6 and len(parts[-1]) == 6
    assert dataset_name("C:/x/Ibm Churn (v2).csv") == "ibm-churn-v2"

# JSON and JSONL files are UTF-8 and accept numpy values
def test_json_writers(tmp_path):
    folder = RunFolder(tmp_path / "run")
    folder.write_json("a.json", {"value": np.float64(0.5), "name": "Bakı €"})
    assert folder.read_json("a.json") == {"value": 0.5, "name": "Bakı €"}
    folder.append("log.jsonl", {"n": np.int64(1)})
    folder.append("log.jsonl", {"n": 2})
    assert folder.read_lines("log.jsonl") == [{"n": 1}, {"n": 2}]
    assert folder.read_lines("missing.jsonl") == []

# the context checkpoint can be read back
def test_save_context(tmp_path):
    folder = RunFolder(tmp_path / "run")
    context = RunContext(run_id="r1", mode="demo", variant="generic", data_path="x.csv", settings_hash="abc")
    folder.save_context(context)
    assert RunContext.model_validate_json(folder.file("context.json").read_text(encoding="utf-8")) == context

# errors go to errors.jsonl with the stage and the traceback
def test_log_error(tmp_path):
    folder = RunFolder(tmp_path / "run")
    try:
        raise ValueError("bad value")
    except ValueError as error:
        folder.log_error("glance", error)
    record = folder.read_lines("errors.jsonl")[0]
    assert record["stage"] == "glance"
    assert "ValueError: bad value" in record["error"]
    assert "Traceback" in record["traceback"]

# symbols become ASCII on the console, and a cp1252 stream never raises
def test_ascii_console(tmp_path):
    assert ascii_text("ok \u2713 \u2192 next \u00b7 \u2500\u2500 \u0259") == "ok ok -> next - -- ?"
    folder = RunFolder(tmp_path / "run")
    handlers = setup_logging(folder, console=False)
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    console = logging.StreamHandler(stream)
    console.setFormatter(AsciiFormatter("%(message)s"))
    logging.getLogger().addHandler(console)
    try:
        logging.getLogger("automl").info("stage \u2713 \u2192 Bak\u0131 \u20ac")
    finally:
        logging.getLogger().removeHandler(console)
        close_logging(handlers)
    stream.flush()
    assert b"stage ok -> Bak? ?" in stream.buffer.getvalue()
    assert "Bak\u0131 \u20ac" in folder.file("run.log").read_text(encoding="utf-8")
