import json
import hashlib
from pathlib import Path
from edatool.paths import agent_file, catalog_file, thresholds_file, tokens_file

# reads a JSON file into a python object
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

# merges the overrides into the base settings, also inside nested...
# dictionaries
def deep_merge(base, overrides):
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

# loads the thresholds, tokens and agent settings and applies the caller's...
# overrides (a dict or a JSON path)
def load_settings(overrides=None):
    settings = {
        "thresholds": read_json(thresholds_file),
        "tokens": read_json(tokens_file),
        "agent": read_json(agent_file),
    }
    if isinstance(overrides, (str, Path)):
        overrides = read_json(overrides)
    return deep_merge(settings, overrides or {})

# returns a hash of the final settings, so a report shows which settings...
# produced it
def settings_hash(settings):
    canonical = json.dumps(settings, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

# loads the issue catalog: title, id prefix, priority, review flag and...
# default fix of each code
def load_catalog():
    return read_json(catalog_file)
