import pytest
import numpy as np
import pandas as pd
from schemas import StopRun
from tools.reader import check_target, drop_duplicate_rows, drop_missing_target, read_table, to_raw_view

# a CSV file is read as text: blanks, "?" and leading zeros stay as written
def test_read_csv_keeps_text(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("id,charge,flag\n007,1.5,yes\n008, ,?\n", encoding="utf-8")
    raw, report = read_table(path)
    assert raw["id"].tolist() == ["007", "008"]
    assert raw["charge"].tolist() == ["1.5", " "]
    assert raw["flag"].tolist() == ["yes", "?"]
    assert report["rows"] == 2 and report["columns"] == 3 and report["delimiter"] == ","

# a DataFrame becomes the same raw view: numbers as text, missing as ""
def test_dataframe_source():
    raw, report = read_table(pd.DataFrame({"a": [1.5, np.nan], "b": ["x", None]}))
    assert raw["a"].tolist() == ["1.5", ""] and raw["b"].tolist() == ["x", ""]
    assert report["format"] == "dataframe"
    assert to_raw_view(pd.DataFrame({"c": [1, 2]}))["c"].tolist() == ["1", "2"]

# missing, unsupported, empty and header-only files are hard stops
def test_hard_stops(tmp_path):
    with pytest.raises(StopRun, match="does not exist"):
        read_table(tmp_path / "missing.csv")
    (tmp_path / "data.xyz").write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(StopRun, match="Unsupported"):
        read_table(tmp_path / "data.xyz")
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    with pytest.raises(StopRun):
        read_table(tmp_path / "empty.csv")
    (tmp_path / "header.csv").write_text("a,b\n", encoding="utf-8")
    with pytest.raises(StopRun, match="no data rows"):
        read_table(tmp_path / "header.csv")

# rows without a target and exact duplicate rows are removed and counted
def test_row_cleaning():
    raw = pd.DataFrame({"x": ["1", "1", "2", "3"], "y": ["a", "a", " ", "b"]})
    kept, dropped = drop_missing_target(raw, "y")
    assert dropped == 1 and kept["x"].tolist() == ["1", "1", "3"]
    unique, duplicates = drop_duplicate_rows(kept)
    assert duplicates == 1 and len(unique) == 2

# the target must exist, have enough rows and two classes for classification
def test_check_target():
    raw = pd.DataFrame({"x": [str(i) for i in range(30)], "y": ["a"] * 30})
    with pytest.raises(StopRun, match="not in the file"):
        check_target(raw, "z", "binary-classification", 20)
    with pytest.raises(StopRun, match="only one class"):
        check_target(raw, "y", "binary-classification", 20)
    with pytest.raises(StopRun, match="labeled rows"):
        check_target(raw.head(5), "x", "regression", 20)
    check_target(raw, "x", "regression", 20)

TABLE = "name,price,city\nAnna,1.5,Bakı\nBen,2.5,Москва\nCem,3.5,Paris\n"

# writes bytes to a file and reads it as the raw view with its read report
def load_bytes(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return read_table(path)

# UTF-8 with a BOM, UTF-16 and cp1252 files load with the right text
def test_encodings(tmp_path):
    bom, report = load_bytes(tmp_path, "bom.csv", b"\xef\xbb\xbf" + TABLE.encode("utf-8"))
    assert list(bom.columns) == ["name", "price", "city"] and report["encoding"] == "utf-8-sig"
    utf16, _ = load_bytes(tmp_path, "u16.csv", TABLE.encode("utf-16"))
    assert utf16["city"].tolist() == ["Bakı", "Москва", "Paris"]
    euro, report = load_bytes(tmp_path, "euro.csv", "item;price\nTea;3 €\nCake;4,50 €\n".encode("cp1252"))
    assert euro["price"].tolist() == ["3 €", "4,50 €"] and report["encoding"] == "cp1252"

# semicolon files with decimal commas, tab and pipe files get their delimiter
def test_delimiters(tmp_path):
    semicolon, _ = load_bytes(tmp_path, "semi.csv", b"a;b;c\n1,5;2,25;x\n3,0;4,75;y\n5,5;6,0;z\n")
    assert list(semicolon.columns) == ["a", "b", "c"] and semicolon["a"].tolist() == ["1,5", "3,0", "5,5"]
    tab, report = load_bytes(tmp_path, "tab.csv", b"a\tb\n1\t2\n3\t4\n")
    assert report["delimiter"] == "\t" and tab.shape == (2, 2)
    pipe, report = load_bytes(tmp_path, "pipe.csv", b"a|b|c\n1|2|3\n4|5|6\n")
    assert report["delimiter"] == "|" and pipe.shape == (2, 3)

# title rows above the header and notes below the table are removed and reported
def test_title_and_footer_csv(tmp_path):
    text = "Sales report 2024\nExported by CRM\n\n" + TABLE + "\nTotal rows: 3\nSource: internal\n"
    raw, report = load_bytes(tmp_path, "report.csv", text.encode("utf-8"))
    assert list(raw.columns) == ["name", "price", "city"] and len(raw) == 3
    assert len(report["title_rows"]) == 2 and len(report["footer_rows"]) == 2

# an Excel file with title rows and a footer note loads the same way
def test_excel_layout(tmp_path):
    path = tmp_path / "book.xlsx"
    rows = [["Quarterly numbers", None, None], ["Company X", None, None], [None, None, None], ["name", "price", "city"]]
    rows += [["Anna", 1.5, "Bakı"], ["Ben", 2, "Москва"], ["Cem", 3.5, "Paris"], [None, None, None], ["Note: prices in USD", None, None]]
    pd.DataFrame(rows).to_excel(path, header=False, index=False)
    raw, report = read_table(path)
    assert list(raw.columns) == ["name", "price", "city"] and raw["price"].tolist() == ["1.5", "2", "3.5"]
    assert len(report["title_rows"]) == 2 and len(report["footer_rows"]) == 1

# quoted fields with newlines stay one value; a last data row with one value...
# and empty cells is kept
def test_quotes_and_sparse_last_row(tmp_path):
    raw, _ = load_bytes(tmp_path, "quotes.csv", b'id,text,n\n1,"line one\nline two",5\n2,plain,6\n3,,\n')
    assert raw["text"].tolist()[0] == "line one\nline two" and len(raw) == 3
    sparse, _ = load_bytes(tmp_path, "sparse.csv", b"a,b,c\n1,2,3\n4,,\n")
    assert len(sparse) == 2
