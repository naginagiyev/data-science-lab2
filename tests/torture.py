import json
import numpy as np
import pandas as pd

ROWS = 300

# a base table: two numbers, a category and a binary target
def base_table(rows=ROWS, seed=0):
    rng = np.random.default_rng(seed)
    amount = rng.normal(50, 10, rows)
    group = rng.choice(["north", "south", "east"], rows)
    label = np.where(amount + (group == "north") * 8 + rng.normal(0, 5, rows) > 52, "yes", "no")
    return pd.DataFrame({"amount": amount.round(2).astype(str), "count": rng.integers(0, 9, rows).astype(str), "region": group, "y": label})

# writes text with the given encoding
def write_text(path, text, encoding="utf-8"):
    path.write_bytes(text.encode(encoding))
    return path

# writes a frame as CSV text
def write_csv(frame, path, **options):
    frame.to_csv(path, index=False, lineterminator="\n", **options)
    return path

# writes the torture file: one column
def one_column(folder):
    return write_csv(base_table()[["y"]], folder / "onecolumn.csv")

# writes the torture file: one row
def one_row(folder):
    return write_csv(base_table().head(1), folder / "onerow.csv")

# writes the torture file: header only
def header_only(folder):
    return write_text(folder / "headeronly.csv", "amount,count,region,y\n")

# writes the torture file: all missing
def all_missing(folder):
    frame = base_table()
    frame[["amount", "count", "region"]] = ""
    return write_csv(frame, folder / "allmissing.csv")

# writes the torture file: one class
def one_class(folder):
    return write_csv(base_table().assign(y="yes"), folder / "oneclass.csv")

# writes the torture file: class with one row
def class_with_one_row(folder):
    frame = base_table()
    frame.loc[0, "y"] = "maybe"
    return write_csv(frame, folder / "classonerow.csv")

# writes the torture file: unicode headers
def unicode_headers(folder):
    frame = base_table().rename(columns={"amount": "Şəhər məbləği", "count": "Возраст", "region": "💰 region"})
    return write_csv(frame, folder / "unicode.csv")

# writes the torture file: duplicate names
def duplicate_names(folder):
    frame = base_table()
    text = frame.to_csv(index=False, lineterminator="\n").replace("amount,count,", "amount,amount,", 1)
    return write_text(folder / "duplicates.csv", text)

# writes the torture file: semicolon decimal comma
def semicolon_decimal_comma(folder):
    frame = base_table()
    frame["amount"] = frame["amount"].str.replace(".", ",", regex=False)
    return write_csv(frame, folder / "semicolon.csv", sep=";")

# writes the torture file: tab delimiter
def tab_delimiter(folder):
    return write_csv(base_table(), folder / "tab.tsv", sep="\t")

# writes the torture file: pipe delimiter
def pipe_delimiter(folder):
    return write_csv(base_table(), folder / "pipe.csv", sep="|")

# writes the torture file: excel titles
def excel_titles(folder):
    frame = base_table()
    path = folder / "titles.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([["Sales report"], ["Region: all"], ["Generated 2026-01-01"]]).to_excel(writer, index=False, header=False)
        frame.to_excel(writer, index=False, startrow=3)
        pd.DataFrame([["Note: amounts in USD"]]).to_excel(writer, index=False, header=False, startrow=len(frame) + 5)
    return path

# writes the torture file: many columns
def many_columns(folder):
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(rng.normal(size=(ROWS, 500)).round(3), columns=[f"f{index}" for index in range(500)]).astype(str)
    frame["y"] = np.where(rng.normal(size=ROWS) + frame["f0"].astype(float) > 0, "yes", "no")
    return write_csv(frame, folder / "wide.csv")

# writes the torture file: million rows
def million_rows(folder):
    rng = np.random.default_rng(3)
    rows = 1_000_000
    amount = rng.normal(50, 10, rows)
    frame = pd.DataFrame({"amount": amount.round(2), "count": rng.integers(0, 9, rows), "region": rng.choice(["north", "south", "east"], rows)})
    frame["y"] = np.where(amount + rng.normal(0, 5, rows) > 52, "yes", "no")
    return write_csv(frame, folder / "million.csv")

# writes the torture file: utf8 bom
def utf8_bom(folder):
    return write_text(folder / "bom.csv", base_table().to_csv(index=False, lineterminator="\n"), "utf-8-sig")

# writes the torture file: cp1252 euro
def cp1252_euro(folder):
    frame = base_table()
    frame["amount"] = frame["amount"] + " €"
    return write_text(folder / "cp1252.csv", frame.to_csv(index=False, lineterminator="\n"), "cp1252")

# writes the torture file: quoted newlines
def quoted_newlines(folder):
    frame = base_table()
    frame["note"] = [f"line one\nline two {index}" for index in range(ROWS)]
    return write_csv(frame, folder / "newlines.csv")

# writes the torture file: numbers and words
def numbers_and_words(folder):
    frame = base_table()
    frame.loc[frame.index % 7 == 0, "amount"] = "unknown"
    frame.loc[frame.index % 11 == 0, "amount"] = "about fifty"
    return write_csv(frame, folder / "mixed.csv")

# writes the torture file: constant features
def constant_features(folder):
    frame = base_table()
    return write_csv(pd.DataFrame({"a": "1", "b": "same", "y": frame["y"]}), folder / "constant.csv")

# writes the torture file: date features
def date_features(folder):
    frame = base_table()
    dates = pd.date_range("2020-01-01", periods=ROWS, freq="D")
    return write_csv(pd.DataFrame({"signup": dates.strftime("%Y-%m-%d"), "last": (dates + pd.Timedelta(days=30)).strftime("%d/%m/%Y"), "y": frame["y"]}), folder / "dates.csv")

# writes the torture file: text features
def text_features(folder):
    frame = base_table()
    words = ["good service and fast delivery", "bad quality and slow answer", "average product nothing special", "excellent price would buy again"]
    rng = np.random.default_rng(4)
    text = [" ".join(rng.choice(words, 3)) for _ in range(ROWS)]
    return write_csv(pd.DataFrame({"review": text, "comment": [item[::-1] for item in text], "y": frame["y"]}), folder / "text.csv")

# writes the torture file: high cardinality id
def high_cardinality_id(folder):
    frame = base_table()
    frame.insert(0, "customer", [f"CUST-{index:06d}" for index in range(ROWS)])
    return write_csv(frame, folder / "idcolumn.csv")

# writes the torture file: units and packed
def units_and_packed(folder):
    frame = base_table()
    rng = np.random.default_rng(5)
    weight = rng.uniform(1, 3, ROWS)
    frame["weight"] = [f"{value:.2f} kg" if index % 2 else f"{value * 1000:.0f} g" for index, value in enumerate(weight)]
    frame["size"] = [f"{width:.1f}x{height:.1f}" for width, height in zip(rng.uniform(1, 9, ROWS), rng.uniform(1, 9, ROWS))]
    return write_csv(frame, folder / "units.csv")

# writes the torture file: inf values
def inf_values(folder):
    frame = base_table()
    frame.loc[frame.index % 13 == 0, "amount"] = "inf"
    frame.loc[frame.index % 17 == 0, "amount"] = "-inf"
    return write_csv(frame, folder / "inf.csv")

# writes the torture file: odd names
def odd_names(folder):
    frame = base_table().rename(columns={"amount": "amount [usd]", "count": "count {n}", "region": 'region "a": b'})
    return write_csv(frame, folder / "oddnames.csv")

# writes the torture file: json input
def json_input(folder):
    path = folder / "table.json"
    path.write_text(json.dumps(base_table().to_dict(orient="records")), encoding="utf-8")
    return path

# writes the torture file: parquet input
def parquet_input(folder):
    path = folder / "table.parquet"
    base_table().to_parquet(path)
    return path

# one case per torture file: (name, writer, target, task, expected); expected...
# is "model" or a part of the stop message of the reader
BINARY = "binary-classification"
CASES = [
    ("one column", one_column, "y", BINARY, "exact duplicate rows"),
    ("one data row", one_row, "y", BINARY, "labeled rows"),
    ("header only", header_only, "y", BINARY, "no data rows"),
    ("all values missing", all_missing, "y", BINARY, "exact duplicate rows"),
    ("target with one class", one_class, "y", BINARY, "only one class"),
    ("a class with one row", class_with_one_row, "y", "multi-class-classification", "model"),
    ("unicode headers", unicode_headers, "y", BINARY, "model"),
    ("duplicate column names", duplicate_names, "y", BINARY, "model"),
    ("semicolon and decimal commas", semicolon_decimal_comma, "y", BINARY, "model"),
    ("tab delimiter", tab_delimiter, "y", BINARY, "model"),
    ("pipe delimiter", pipe_delimiter, "y", BINARY, "model"),
    ("excel with title rows and a footer", excel_titles, "y", BINARY, "model"),
    ("500 columns", many_columns, "y", BINARY, "model"),
    ("utf-8 bom", utf8_bom, "y", BINARY, "model"),
    ("cp1252 with euro", cp1252_euro, "y", BINARY, "model"),
    ("quoted fields with newlines", quoted_newlines, "y", BINARY, "model"),
    ("numbers and words in one column", numbers_and_words, "y", BINARY, "model"),
    ("only constant features", constant_features, "y", BINARY, "exact duplicate rows"),
    ("only date features", date_features, "y", BINARY, "model"),
    ("only long-text features", text_features, "y", BINARY, "model"),
    ("high-cardinality id", high_cardinality_id, "y", BINARY, "model"),
    ("units and packed values", units_and_packed, "y", BINARY, "model"),
    ("inf values", inf_values, "y", BINARY, "model"),
    ("column names with brackets and quotes", odd_names, "y", BINARY, "model"),
    ("json input", json_input, "y", BINARY, "model"),
    ("parquet input", parquet_input, "y", BINARY, "model"),
]
MILLION = ("1,000,000 rows", million_rows, "y", BINARY, "model")
