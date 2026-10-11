import re
import csv
import warnings
import pandas as pd
from pathlib import Path

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

# guesses the delimiter from the first part of the file; TSV files always use...
# a tab
def sniff_delimiter(path, sample_size=4096):
    if path.suffix.lower() == ".tsv":
        return "\t"
    try:
        with open(path, "r", encoding="utf-8") as file:
            return csv.Sniffer().sniff(file.read(sample_size)).delimiter
    except Exception:
        return None

# reads a delimited file with every value as text and records the line...
# numbers of broken lines
def read_text_table(path, delimiter, encoding):
    options = {
        "dtype": str,
        "keep_default_na": False,
        "na_filter": False,
        "on_bad_lines": "warn",
        "encoding": encoding,
    }
    if delimiter is None:
        options.update({"sep": None, "engine": "python"})
    else:
        options["sep"] = delimiter
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.ParserWarning)
        df = pd.read_csv(path, **options)
    messages = " ".join(str(warning.message) for warning in caught)
    skipped_lines = [int(line) for line in re.findall(r"Skipping line (\d+)", messages)]
    return df, skipped_lines

# reads a CSV or TSV file, falls back to latin-1 when UTF-8 fails, and...
# returns parse diagnostics
def read_delimited(path):
    delimiter = sniff_delimiter(path)
    encoding = "utf-8"
    try:
        df, skipped_lines = read_text_table(path, delimiter, encoding)
    except UnicodeDecodeError:
        encoding = "latin-1"
        df, skipped_lines = read_text_table(path, delimiter, encoding)
    diagnostics = {
        "encoding": encoding,
        "delimiter": delimiter,
        "engine": "c" if delimiter is not None else "python",
        "skipped_lines": skipped_lines,
    }
    return df, diagnostics

# counts rows inside the data that repeat the header row
def count_repeated_headers(df):
    if df.empty:
        return 0
    header = [str(name) for name in df.columns]
    candidates = df[df.iloc[:, 0] == header[0]]
    return int((candidates == header).all(axis=1).sum())

# public entry point: loads a file as a raw text view and stores the parse...
# diagnostics in df.attrs
def load_raw(path):
    path = Path(path)
    extension = path.suffix.lower()
    if extension in {".csv", ".tsv"}:
        df, diagnostics = read_delimited(path)
        raw_text = True
    elif extension in typed_readers:
        typed = typed_readers[extension](path)
        df = typed.astype(str).where(typed.notna(), "")
        diagnostics = {}
        raw_text = False
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
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
