import csv
import pandas as pd
from pathlib import Path
from edabench.legacy.edaconfig import readers as external_readers

def read_delimited(input_path: str, sample_size: int = 4096) -> pd.DataFrame:
    delimiter = None
    ext = Path(input_path).suffix.lower()

    if ext == ".tsv":
        delimiter = "\t"
    else:
        try:
            with open(input_path, "r", encoding="utf-8") as file:
                sample = file.read(sample_size)
                delimiter = csv.Sniffer().sniff(sample).delimiter
        except Exception:
            delimiter = None

    read_kwargs = {"engine": "python", "sep": delimiter if delimiter is not None else None}
    try:
        return pd.read_csv(input_path, encoding="utf-8", **read_kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(input_path, encoding="latin-1", **read_kwargs)

def load_data(input_path: str, target_col: str) -> pd.DataFrame:
    ext = Path(input_path).suffix.lower()
    df: pd.DataFrame | None = None

    if ext in {".csv", ".tsv"}:
        df = read_delimited(input_path)
    elif ext in external_readers:
        df = external_readers[ext](input_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if df is None or df.empty:
        raise ValueError("Loaded dataframe is empty.")

    if df.shape[1] == 0:
        raise ValueError("Loaded dataframe has no columns.")

    if all(df[col].isna().all() for col in df.columns):
        raise ValueError("All columns are fully missing.")

    if target_col is not None and target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the dataset.")

    return df
