import io
import gzip
import zipfile
from benchmark import natural

# a zip file with one member, as bytes
def zipped(name, data):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, data)
    return buffer.getvalue()

# unpack walks gzip, then nested zip members, and returns the inner bytes
def test_unpack_nested():
    inner = zipped("folder/table.csv", b"a,b\n1,2\n")
    outer = zipped("inner.zip", inner)
    assert natural.unpack(outer, ["inner.zip", "folder/table.csv"]) == b"a,b\n1,2\n"
    assert natural.unpack(gzip.compress(b"x\n"), compressed=True) == b"x\n"
    assert natural.unpack(b"plain") == b"plain"

# every dataset has a free licence, a source, a target, and the kinds are...
# balanced: as many mess-only as leak-only as both
def test_specs_complete_and_balanced():
    kinds = [spec["kind"] for spec in natural.NATURAL_DATASETS.values()]
    assert set(kinds) == {"mess", "leak", "both"}
    assert kinds.count("mess") == kinds.count("leak") == kinds.count("both")
    for spec in natural.NATURAL_DATASETS.values():
        assert spec["licence"] and spec["licence_url"] and spec["source"] and spec["citation"] and spec["target"]
        for file_spec in spec["files"].values():
            assert "url" in file_spec or "openml" in file_spec
