import pandas as pd
from pathlib import Path
from schemas import StopRun
from edatool import load_raw
from edatool.core.loader import typed_readers

SUPPORTED = {".csv", ".tsv"} | set(typed_readers)

# turns any DataFrame into the raw text view: every value a string, missing...
# values as ""
def to_raw_view(frame):
    frame = frame.copy()
    if all(pd.api.types.is_object_dtype(dtype) for dtype in frame.dtypes) and not frame.isna().any().any():
        return frame.astype(str)
    return frame.astype(str).where(frame.notna(), "")

# reads a file (or takes a DataFrame) as the raw text view and returns it with...
# a read report; unsupported, unreadable or empty files are hard stops
def read_table(source):
    if isinstance(source, pd.DataFrame):
        raw = to_raw_view(source)
        report = {"source": "dataframe", "format": "dataframe"}
    else:
        path = Path(source)
        if not path.exists():
            raise StopRun(f"The file {path} does not exist.")
        if path.suffix.lower() not in SUPPORTED:
            raise StopRun(f"Unsupported file type '{path.suffix}'. Use one of: {', '.join(sorted(SUPPORTED))}.")
        try:
            raw = load_raw(path)
        except Exception as error:
            raise StopRun(f"The file {path.name} could not be read: {type(error).__name__}: {error}") from error
        report = dict(raw.attrs.get("edatool", {}))
    if raw.shape[1] == 0:
        raise StopRun("The file has no columns.")
    if raw.shape[0] == 0:
        raise StopRun("The file has no data rows (only a header or nothing).")
    raw = raw.reset_index(drop=True)
    report.update(rows=int(raw.shape[0]), columns=int(raw.shape[1]))
    return raw, report

# removes rows whose target is empty and returns the frame and the count
def drop_missing_target(raw, target):
    keep = raw[target].astype(str).str.strip() != ""
    return raw[keep].reset_index(drop=True), int((~keep).sum())

# removes exact duplicate rows (demo mode only, before the split) and returns...
# the frame and the count
def drop_duplicate_rows(raw):
    duplicated = raw.duplicated()
    return raw[~duplicated].reset_index(drop=True), int(duplicated.sum())

# the hard stops about the target: it must exist, have enough labeled rows...
# (the message counts the removed duplicate rows) and, for classification, at...
# least two classes
def check_target(raw, target, task, min_rows, duplicates=0):
    if target not in raw.columns:
        raise StopRun(f"The target column '{target}' is not in the file. Columns: {', '.join(map(str, raw.columns[:20]))}.")
    if len(raw) < min_rows:
        removed = f" after removing {duplicates} exact duplicate rows (the features do not tell the rows apart)" if duplicates else ""
        raise StopRun(f"Only {len(raw)} labeled rows{removed}; at least {min_rows} are needed.")
    if task != "regression" and raw[target].astype(str).str.strip().nunique() < 2:
        raise StopRun(f"The target '{target}' has only one class; classification needs at least two.")
