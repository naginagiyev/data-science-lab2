import io
import logging
import pandas as pd
from edatool import audit
from edatool.core.progress import safe_text

# a console handler that remembers logging errors instead of printing them
class RecordingHandler(logging.StreamHandler):
    def __init__(self, stream):
        super().__init__(stream)
        self.errors = 0

    def handleError(self, record):
        self.errors += 1

# edatool used as a library with a cp1252 console never raises...
# UnicodeEncodeError: symbols become ASCII and other characters "?"
def test_cp1252_console():
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    handler = RecordingHandler(stream)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        frame = pd.DataFrame({"Bakı şəhəri": [str(i % 7) for i in range(60)], "Город": ["a", "b", "c"] * 20, "y": ["x", "z"] * 30})
        audit(frame, "y", "binary-classification", use_agent=False)
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    stream.flush()
    text = stream.buffer.getvalue().decode("cp1252")
    assert handler.errors == 0
    assert "edatool audit - DataFrame" in text and "─" not in text

# symbols get readable ASCII replacements
def test_safe_text():
    assert safe_text("✓ done · 2 → 3 └─ x", "cp1252") == "ok done - 2 -> 3 `- x"
    assert safe_text("Bakı", "ascii") == "Bak?"
