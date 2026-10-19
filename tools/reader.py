import io
import re
import csv
import warnings
import pandas as pd
from pathlib import Path
from schemas import StopRun
from collections import Counter

# readers for formats that already have types; their values become text
TYPED_READERS = {
    ".json": pd.read_json, ".parquet": pd.read_parquet, ".pkl": pd.read_pickle, ".pickle": pd.read_pickle,
    ".feather": pd.read_feather, ".orc": pd.read_orc, ".h5": pd.read_hdf, ".hdf5": pd.read_hdf, ".xml": pd.read_xml,
}
EXCEL = {".xlsx", ".xls"}
SUPPORTED = {".csv", ".tsv"} | EXCEL | set(TYPED_READERS)
DELIMITERS = [",", ";", "\t", "|"]
NUMBER = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?%?$")
SCAN_LINES = 300

# decodes the bytes of a text file: a UTF-8 or UTF-16 BOM decides first, then...
# UTF-8, cp1252 and latin-1 are tried in this order
def decode(data):
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16"
    for encoding in ["utf-8", "cp1252"]:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"

# splits lines into fields with the csv module (quotes are respected)
def split_fields(lines, delimiter):
    try:
        return list(csv.reader(lines, delimiter=delimiter))
    except csv.Error:
        return [line.split(delimiter) for line in lines]

# the most common field count of the rows (counts above 1 first, so title and...
# note rows do not win a tie) and the share of rows that have it
def modal_count(rows):
    counts = Counter(len(row) for row in rows if any(cell.strip() for cell in row))
    if not counts:
        return 0, 0.0
    wide = {count: hits for count, hits in counts.items() if count > 1}
    count, hits = max((wide or counts).items(), key=lambda item: (item[1], item[0]))
    return count, hits / sum(counts.values())

# guesses the delimiter: the sniffer's guess and the usual delimiters are...
# scored by how consistent their field counts are
def choose_delimiter(lines, path):
    if path.suffix.lower() == ".tsv":
        return "\t"
    candidates = list(DELIMITERS)
    try:
        candidates.insert(0, csv.Sniffer().sniff("\n".join(lines[:50]), delimiters="".join(DELIMITERS)).delimiter)
    except csv.Error:
        pass
    scores = {}
    for delimiter in candidates:
        count, share = modal_count(split_fields(lines, delimiter))
        scores.setdefault(delimiter, (count > 1, share, count))
    return max(scores, key=lambda delimiter: scores[delimiter]) if scores else ","

# true when a row looks like a header: the full width and mostly text
def looks_like_header(row, width):
    cells = [cell.strip() for cell in row if cell.strip()]
    if len(row) != width or not cells:
        return False
    text_cells = [cell for cell in cells if not NUMBER.match(cell)]
    return len(text_cells) >= max(1, 0.5 * len(cells)) and len(cells) >= min(width, 2)

# the header row: the first full-width row of mostly text (rows above it are...
# title rows)
def find_header(rows, width):
    position = next((index for index, row in enumerate(rows) if looks_like_header(row, width)), None)
    return position if position is not None else next((index for index, row in enumerate(rows) if len(row) == width), 0)

# true when a row is a note: one text cell in the first column
def is_note(row, width):
    cells = [cell for cell in row if cell.strip()]
    first = row[0].strip() if row else ""
    return width > 1 and len(cells) == 1 and first != "" and not NUMBER.match(first)

# the note rows below the table: from the end, rows with another field count...
# or notes (blank lines are skipped)
def find_footer(rows, width):
    notes = []
    for row in reversed(rows):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != width or is_note(row, width):
            notes.append(row)
            continue
        break
    return list(reversed(notes))

# makes header names unique the way pandas does ("a", "a.1"); empty names...
# become "Unnamed: <position>"
def unique_names(names):
    seen = Counter()
    result = []
    for position, name in enumerate(names):
        name = str(name) if str(name).strip() else f"Unnamed: {position}"
        result.append(name if seen[name] == 0 else f"{name}.{seen[name]}")
        seen[name] += 1
    return result

# reads a CSV or TSV file with every value as text: encoding, delimiter, title...
# rows above the header and notes below the table are found first
def read_delimited(path):
    text, encoding = decode(path.read_bytes())
    if not text.strip():
        raise StopRun(f"The file {path.name} is empty.")
    lines = text.splitlines()
    delimiter = choose_delimiter(lines[:SCAN_LINES], path)
    rows = split_fields(lines[:SCAN_LINES], delimiter)
    width, _ = modal_count(rows)
    header = find_header(rows, width)
    data_lines = [line for line in lines[header + 1 :] if line.strip()]
    footer = find_footer(split_fields(data_lines[-50:], delimiter), width)
    footer = footer if len(footer) < len(data_lines) else []
    options = {"sep": delimiter, "dtype": str, "keep_default_na": False, "na_filter": False, "on_bad_lines": "warn", "skiprows": header}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.ParserWarning)
        frame = pd.read_csv(io.StringIO(text), **options)
    skipped = [int(line) for line in re.findall(r"Skipping line (\d+)", " ".join(str(item.message) for item in caught))]
    parsed_footer = [row for row in footer if len(row) <= width]
    if parsed_footer:
        frame = frame.iloc[: len(frame) - len(parsed_footer)]
    footer_start = len(lines) - len(footer)
    report = {
        "encoding": encoding, "delimiter": delimiter, "skipped_lines": [line for line in skipped if line <= footer_start],
        "title_rows": [delimiter.join(row) for row in rows[:header] if any(cell.strip() for cell in row)],
        "footer_rows": [delimiter.join(row) for row in footer],
    }
    return frame, report

# reads an Excel sheet without a header and finds the title rows, header and...
# notes like for text files
def read_excel(path):
    grid = pd.read_excel(path, header=None, dtype=object)
    rows = [["" if pd.isna(value) else str(value) for value in row] for row in grid.itertuples(index=False)]
    rows = [row[: max([index + 1 for index, cell in enumerate(row) if cell.strip()] or [0])] for row in rows]
    width, _ = modal_count(rows)
    header = find_header(rows, width)
    footer = find_footer(rows[header + 1 :], width)
    body = rows[header + 1 : len(rows) - len(footer)]
    names = unique_names(rows[header] + [""] * (width - len(rows[header])))[:width]
    data = [row[:width] + [""] * (width - len(row[:width])) for row in body if any(cell.strip() for cell in row)]
    titles = [" ".join(row) for row in rows[:header] if any(cell.strip() for cell in row)]
    return pd.DataFrame(data, columns=names, dtype=object), {"title_rows": titles, "footer_rows": [" ".join(row) for row in footer]}

# turns any DataFrame into the raw text view: every value a string, missing...
# values as ""
def to_raw_view(frame):
    return frame.astype(str).where(frame.notna(), "")

# reads a file as the raw text view and returns it with a read report
def load_file(path):
    extension = path.suffix.lower()
    if extension in {".csv", ".tsv"}:
        frame, report = read_delimited(path)
    elif extension in EXCEL:
        frame, report = read_excel(path)
    else:
        frame, report = TYPED_READERS[extension](path), {}
    short_rows = int(frame.isna().any(axis=1).sum())
    return to_raw_view(frame), {"source": path.stem, "format": extension, "short_rows": short_rows, **report}

# reads a file (or takes a DataFrame) as the raw text view and returns it with...
# a read report; unsupported, unreadable or empty files are hard stops
def read_table(source):
    if isinstance(source, pd.DataFrame):
        raw, report = to_raw_view(source), {"source": "dataframe", "format": "dataframe"}
    else:
        path = Path(source)
        if not path.exists():
            raise StopRun(f"The file {path} does not exist.")
        if path.suffix.lower() not in SUPPORTED:
            raise StopRun(f"Unsupported file type '{path.suffix}'. Use one of: {', '.join(sorted(SUPPORTED))}.")
        try:
            raw, report = load_file(path)
        except StopRun:
            raise
        except Exception as error:
            raise StopRun(f"The file {path.name} could not be read: {type(error).__name__}: {error}") from error
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
