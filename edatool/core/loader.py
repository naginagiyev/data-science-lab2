import io
import re
import csv
import warnings
import pandas as pd
from pathlib import Path
from collections import Counter

# readers for file formats that already have types; their values are turned...
# into text after loading
typed_readers = {
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".json": pd.read_json,
    ".parquet": pd.read_parquet,
    ".pkl": pd.read_pickle,
    ".pickle": pd.read_pickle,
    ".feather": pd.read_feather,
    ".orc": pd.read_orc,
    ".h5": pd.read_hdf,
    ".hdf5": pd.read_hdf,
    ".xml": pd.read_xml,
}
excel_extensions = {".xlsx", ".xls"}
candidate_delimiters = [",", ";", "\t", "|"]
number_pattern = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?%?$")
scan_lines = 300

# decodes the bytes of a text file: a UTF-8 or UTF-16 BOM decides first, then...
# UTF-8, cp1252 and latin-1 are tried in this order; returns text and encoding
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
# scored by how consistent their field counts are (count above 1 first)
def choose_delimiter(lines, path):
    if path.suffix.lower() == ".tsv":
        return "\t"
    candidates = list(candidate_delimiters)
    try:
        sniffed = csv.Sniffer().sniff("\n".join(lines[:50]), delimiters="".join(candidate_delimiters)).delimiter
        candidates.insert(0, sniffed)
    except csv.Error:
        pass
    best, best_score = ",", None
    for delimiter in candidates:
        count, share = modal_count(split_fields(lines, delimiter))
        score = (count > 1, share, count)
        if best_score is None or score > best_score:
            best, best_score = delimiter, score
    return best

# true when a row looks like a header: enough fields and mostly non-numeric text
def looks_like_header(row, width):
    cells = [cell.strip() for cell in row if cell.strip()]
    if len(row) != width or not cells:
        return False
    text_cells = [cell for cell in cells if not number_pattern.match(cell)]
    return len(text_cells) >= max(1, 0.5 * len(cells)) and len(cells) >= min(width, 2)

# finds the header row: the first row with the modal field count and mostly...
# non-numeric text (the rows above it are title rows)
def find_header(rows, width):
    for position, row in enumerate(rows):
        if looks_like_header(row, width):
            return position
    for position, row in enumerate(rows):
        if len(row) == width:
            return position
    return 0

# true when a row is a note: one non-empty cell, in the first column, that is...
# text and not a number
def is_note(row, width):
    cells = [cell for cell in row if cell.strip()]
    first = row[0].strip() if row else ""
    return width > 1 and len(cells) == 1 and first != "" and not number_pattern.match(first)

# finds the note rows below the table: from the end, rows whose field count...
# differs or that are notes (blank lines are skipped)
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

# reads a delimited text table with every value as text and records the line...
# numbers of broken lines
def read_text_table(text, delimiter, skip):
    options = {
        "sep": delimiter,
        "dtype": str,
        "keep_default_na": False,
        "na_filter": False,
        "on_bad_lines": "warn",
        "skiprows": skip,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.ParserWarning)
        df = pd.read_csv(io.StringIO(text), **options)
    messages = " ".join(str(warning.message) for warning in caught)
    skipped_lines = [int(line) for line in re.findall(r"Skipping line (\d+)", messages)]
    return df, skipped_lines

# reads a CSV or TSV file: encoding, delimiter, title rows above the header...
# and notes below the table are found first; returns the frame and diagnostics
def read_delimited(path):
    text, encoding = decode(Path(path).read_bytes())
    if not text.strip():
        raise ValueError(f"The file {Path(path).name} is empty.")
    lines = text.splitlines()
    head = lines[:scan_lines]
    delimiter = choose_delimiter(head, Path(path))
    rows = split_fields(head, delimiter)
    width, _ = modal_count(rows)
    header = find_header(rows, width)
    data_lines = [line for line in lines[header + 1 :] if line.strip()]
    footer = find_footer(split_fields(data_lines[-50:], delimiter), width)
    footer = footer if len(footer) < len(data_lines) else []
    df, skipped_lines = read_text_table(text, delimiter, header)
    parsed_footer = [row for row in footer if len(row) <= width]
    if parsed_footer:
        df = df.iloc[: len(df) - len(parsed_footer)]
    footer_start = len(lines) - len(footer)
    skipped_lines = [line for line in skipped_lines if line <= footer_start]
    diagnostics = {
        "encoding": encoding,
        "delimiter": delimiter,
        "engine": "c",
        "skipped_lines": skipped_lines,
        "title_rows": [delimiter.join(row) for row in rows[:header] if any(cell.strip() for cell in row)],
        "footer_rows": [delimiter.join(row) for row in footer],
    }
    return df, diagnostics

# reads an Excel sheet without a header, finds the title rows, header and...
# notes like for text files, and returns the table as text
def read_excel(path):
    grid = pd.read_excel(path, header=None, dtype=object)
    rows = [["" if pd.isna(value) else str(value) for value in row] for row in grid.itertuples(index=False)]
    rows = [row[: max([index + 1 for index, cell in enumerate(row) if cell.strip()] or [0])] for row in rows]
    width, _ = modal_count(rows)
    header = find_header(rows, width)
    footer = find_footer(rows[header + 1 :], width)
    body = rows[header + 1 : len(rows) - len(footer)]
    names = mangle_names(rows[header] + [""] * (width - len(rows[header])))[:width]
    data = [row[:width] + [""] * (width - len(row[:width])) for row in body if any(cell.strip() for cell in row)]
    df = pd.DataFrame(data, columns=names, dtype=object)
    titles = [" ".join(row) for row in rows[:header] if any(cell.strip() for cell in row)]
    diagnostics = {"title_rows": titles, "footer_rows": [" ".join(row) for row in footer]}
    return df, diagnostics

# makes header names unique the way pandas does ("a", "a.1"); empty names...
# become "Unnamed: <position>"
def mangle_names(names):
    seen = Counter()
    result = []
    for position, name in enumerate(names):
        name = str(name) if str(name).strip() else f"Unnamed: {position}"
        result.append(name if seen[name] == 0 else f"{name}.{seen[name]}")
        seen[name] += 1
    return result

# counts rows inside the data that repeat the header row
def count_repeated_headers(df):
    if df.empty:
        return 0
    header = [str(name) for name in df.columns]
    candidates = df[df.iloc[:, 0] == header[0]]
    return int((candidates == header).all(axis=1).sum())

# public entry point: loads a file as a raw text view and stores the parse...
# diagnostics in df.attrs; empty and header-only files raise a clear error
def load_raw(path):
    path = Path(path)
    extension = path.suffix.lower()
    if extension in {".csv", ".tsv"}:
        df, diagnostics = read_delimited(path)
        raw_text = True
    elif extension in excel_extensions:
        df, diagnostics = read_excel(path)
        df = df.astype(str).where(df.notna(), "")
        raw_text = False
    elif extension in typed_readers:
        typed = typed_readers[extension](path)
        df = typed.astype(str).where(typed.notna(), "")
        diagnostics = {}
        raw_text = False
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
    if df.shape[1] == 0:
        raise ValueError(f"The file {path.name} has no columns.")
    if df.shape[0] == 0:
        raise ValueError(f"The file {path.name} has a header but no data rows.")
    short_rows = int(df.isna().any(axis=1).sum())
    df = df.fillna("")
    df.attrs["edatool"] = {
        "source": path.stem,
        "format": extension,
        "raw_text": raw_text,
        "columns": df.shape[1],
        "short_rows": short_rows,
        "repeated_header_rows": count_repeated_headers(df),
        **diagnostics,
    }
    return df

# checks if a DataFrame is a raw text view (all text, no NaN), so hidden...
# tokens are still visible
def is_raw_view(df):
    if df.attrs.get("edatool", {}).get("raw_text") is False:
        return False
    all_text = all(pd.api.types.is_object_dtype(dtype) for dtype in df.dtypes)
    return all_text and not df.isna().any().any()
