import pytest
import pandas as pd
from edatool import audit, load_raw

TABLE = "name,price,city\nAnna,1.5,Bakı\nBen,2.5,Москва\nCem,3.5,Paris\n"

# writes bytes to a file and loads it as the raw view
def load_bytes(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return load_raw(path)

# UTF-8 with a BOM, UTF-16 and cp1252 files load with the right text
def test_encodings(tmp_path):
    bom = load_bytes(tmp_path, "bom.csv", b"\xef\xbb\xbf" + TABLE.encode("utf-8"))
    assert list(bom.columns) == ["name", "price", "city"] and bom.attrs["edatool"]["encoding"] == "utf-8-sig"
    utf16 = load_bytes(tmp_path, "u16.csv", TABLE.encode("utf-16"))
    assert utf16["city"].tolist() == ["Bakı", "Москва", "Paris"]
    euro = load_bytes(tmp_path, "euro.csv", "item;price\nTea;3 €\nCake;4,50 €\n".encode("cp1252"))
    assert euro["price"].tolist() == ["3 €", "4,50 €"] and euro.attrs["edatool"]["encoding"] == "cp1252"

# semicolon files with decimal commas, tab and pipe files get their delimiter
def test_delimiters(tmp_path):
    semicolon = load_bytes(tmp_path, "semi.csv", b"a;b;c\n1,5;2,25;x\n3,0;4,75;y\n5,5;6,0;z\n")
    assert list(semicolon.columns) == ["a", "b", "c"] and semicolon["a"].tolist() == ["1,5", "3,0", "5,5"]
    tab = load_bytes(tmp_path, "tab.csv", b"a\tb\n1\t2\n3\t4\n")
    assert tab.attrs["edatool"]["delimiter"] == "\t" and tab.shape == (2, 2)
    pipe = load_bytes(tmp_path, "pipe.csv", b"a|b|c\n1|2|3\n4|5|6\n")
    assert pipe.attrs["edatool"]["delimiter"] == "|" and pipe.shape == (2, 3)

# title rows above the header and notes below the table are removed and reported
def test_title_and_footer_csv(tmp_path):
    text = "Sales report 2024\nExported by CRM\n\n" + TABLE + "\nTotal rows: 3\nSource: internal\n"
    raw = load_bytes(tmp_path, "report.csv", text.encode("utf-8"))
    assert list(raw.columns) == ["name", "price", "city"] and len(raw) == 3
    info = raw.attrs["edatool"]
    assert len(info["title_rows"]) == 2 and len(info["footer_rows"]) == 2
    report = audit(raw, "price", "regression", use_agent=False)
    layout = [issue for issue in report.issues if issue.code == "A01"]
    assert layout and layout[0].evidence["title_rows"]

# an Excel file with 3 title rows and a footer note loads the same way
def test_excel_layout(tmp_path):
    path = tmp_path / "book.xlsx"
    rows = [["Quarterly numbers", None, None], ["Company X", None, None], [None, None, None], ["name", "price", "city"]]
    rows += [["Anna", 1.5, "Bakı"], ["Ben", 2, "Москва"], ["Cem", 3.5, "Paris"], [None, None, None], ["Note: prices in USD", None, None]]
    pd.DataFrame(rows).to_excel(path, header=False, index=False)
    raw = load_raw(path)
    assert list(raw.columns) == ["name", "price", "city"] and raw["price"].tolist() == ["1.5", "2", "3.5"]
    assert len(raw.attrs["edatool"]["title_rows"]) == 2 and len(raw.attrs["edatool"]["footer_rows"]) == 1

# quoted fields with newlines stay one value; a last data row with one value...
# and empty cells is kept
def test_quotes_and_sparse_last_row(tmp_path):
    raw = load_bytes(tmp_path, "quotes.csv", b'id,text,n\n1,"line one\nline two",5\n2,plain,6\n3,,\n')
    assert raw["text"].tolist()[0] == "line one\nline two" and len(raw) == 3
    sparse = load_bytes(tmp_path, "sparse.csv", b"a,b,c\n1,2,3\n4,,\n")
    assert len(sparse) == 2

# empty and header-only files raise a clear error
def test_empty_files(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        load_bytes(tmp_path, "empty.csv", b"")
    with pytest.raises(ValueError, match="no data rows"):
        load_bytes(tmp_path, "header.csv", b"a,b,c\n")
