import os
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from paths import progress_file, spend_file

CATEGORIES = ("dev", "experiments")

# holds a lock file next to the ledger, so two processes never write the...
# ledger at the same time; a lock older than 60 s is treated as stale
@contextmanager
def ledger_lock(path):
    lock = path.with_name(path.name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 30
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 60:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"The spend ledger lock {lock} is busy.")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)

# reads the ledger; a missing file is an empty ledger
def read_ledger(path=None):
    path = path or spend_file
    if not path.exists():
        return {"totals": {category: 0.0 for category in CATEGORIES}, "calls": []}
    return json.loads(path.read_text(encoding="utf-8"))

# writes the ledger through a temporary file, so a crash never leaves half a file
def write_ledger(ledger, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)

# returns the dollars spent per category and in total
def totals(path=None):
    ledger = read_ledger(path)
    spent = {category: float(ledger["totals"].get(category, 0.0)) for category in CATEGORIES}
    spent["total"] = sum(spent.values())
    return spent

# says if a call with this estimated cost may run: it must stay under the cap...
# of its category and under the total cap; returns (allowed, reason)
def check_allowed(estimate, spend_settings, path=None):
    spent = totals(path)
    category = spend_settings.category
    cap = spend_settings.dev_cap_usd if category == "dev" else spend_settings.experiments_cap_usd
    if spent[category] + estimate > cap:
        return False, f"the {category} cap of ${cap:.2f} would be crossed (spent ${spent[category]:.4f})"
    if spent["total"] + estimate > spend_settings.total_cap_usd:
        return False, f"the total cap of ${spend_settings.total_cap_usd:.2f} would be crossed (spent ${spent['total']:.4f})"
    return True, ""

# true when the total spend reached the level where optional LLM work stops
def optional_stopped(spend_settings, path=None):
    return totals(path)["total"] >= spend_settings.stop_optional_usd

# adds one paid call to the ledger and returns the new totals; writes a...
# warning into progress.md the first time the total passes the optional-stop level
def record(cost, category, model, input_tokens, output_tokens, source, spend_settings=None, path=None):
    path = path or spend_file
    with ledger_lock(path):
        ledger = read_ledger(path)
        ledger["totals"][category] = float(ledger["totals"].get(category, 0.0)) + cost
        ledger["calls"].append({
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "category": category,
            "source": source,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        })
        total = sum(float(value) for value in ledger["totals"].values())
        warn = spend_settings is not None and total >= spend_settings.stop_optional_usd and not ledger.get("optional_stop_warned")
        if warn:
            ledger["optional_stop_warned"] = True
        write_ledger(ledger, path)
    if warn and path == spend_file:
        note = f"\n- WARNING ({datetime.now().date()}): LLM spend reached ${total:.2f}; optional LLM work stops (buildplan §0.5).\n"
        with progress_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(note.lstrip("\n"))
    return totals(path)
