import time
import pickle
import pytest
import textwrap
import pandas as pd
from schemas import Step
from operations.registry import build
from tools.sandbox import SandboxError, check_code, run_code

GOOD = textwrap.dedent("""    import re
    import pandas as pd

    class CleaningStep:
        def fit(self, frame):
            return self

        def transform(self, frame):
            def size(text, unit):
                match = re.search(r"(\\d+)\\s*(GB|TB)\\s*" + unit, text)
                if not match:
                    return 0.0
                return float(match.group(1)) * (1000 if match.group(2) == "TB" else 1)
            memory = frame["Memory"]
            return pd.DataFrame({"ssd_gb": memory.map(lambda text: size(text, "SSD")), "hdd_gb": memory.map(lambda text: size(text, "HDD"))}, index=frame.index)
""")

# a frame of laptop memory texts
def memory_frame(rows=60):
    values = ["128GB SSD + 1TB HDD", "256GB SSD", "1TB HDD", "512GB SSD"]
    return pd.DataFrame({"Memory": [values[index % 4] for index in range(rows)], "Price": [str(index) for index in range(rows)]})

# the AST check accepts allowed code and refuses forbidden imports, names,...
# dunder attributes and a wrong class shape
def test_ast_check():
    assert check_code(GOOD, "CleaningStep") == []
    cases = {
        "import os\nclass CleaningStep:\n    def fit(self, frame): pass\n    def transform(self, frame): return frame": "'os' is not allowed",
        "class CleaningStep:\n    def fit(self, frame): open('x')\n    def transform(self, frame): return frame": "'open' is not allowed",
        "class CleaningStep:\n    def fit(self, frame): return frame.__class__\n    def transform(self, frame): return frame": "'__class__' is not allowed",
        "class CleaningStep:\n    def fit(self, frame): pass": "transform",
        "class A:\n    pass\nclass B:\n    pass": "exactly one class",
        "def broken(:": "syntax error",
        "import requests\nclass CleaningStep:\n    def fit(self, frame): pass\n    def transform(self, frame): return frame": "'requests' is not allowed",
    }
    for code, expected in cases.items():
        problems = check_code(code, "CleaningStep")
        assert any(expected in problem for problem in problems), (expected, problems)

# generated code runs in a separate process on parquet data
def test_run_code():
    frame = memory_frame()
    output = run_code(GOOD, "CleaningStep", frame, [frame.head(4)])[0]
    assert output["ssd_gb"].tolist() == [128.0, 256.0, 0.0, 512.0] and output["hdd_gb"].tolist() == [1000.0, 0.0, 1000.0, 0.0]

# a time-out kills the process tree
def test_timeout():
    code = "class CleaningStep:\n    def fit(self, frame):\n        while True:\n            pass\n    def transform(self, frame):\n        return frame\n"
    start = time.monotonic()
    with pytest.raises(SandboxError, match="longer than"):
        run_code(code, "CleaningStep", memory_frame(), [memory_frame()], timeout=3)
    assert time.monotonic() - start < 30

# no API key reaches the sandbox process (the check is switched off here only...
# to let the code read the environment)
def test_no_api_key_in_subprocess(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-test")
    code = "import os\nimport pandas as pd\nclass CleaningStep:\n    def fit(self, frame):\n        pass\n    def transform(self, frame):\n        keys = ','.join(sorted(os.environ))\n        return pd.DataFrame({'keys': [keys] * len(frame)}, index=frame.index)\n"
    output = run_code(code, "CleaningStep", memory_frame(), [memory_frame()], check=False)[0]
    assert "OPENAI_API_KEY" not in output["keys"].iloc[0] and "PYTHONHASHSEED" in output["keys"].iloc[0]

# the freecode operation fits and transforms through the sandbox and pickles
def test_freecode_operation():
    operation = build(Step(op="freecode", params={"code": GOOD}))
    frame = memory_frame()
    output = operation.fit(frame).transform(frame.head(3))
    assert list(output.columns) == ["ssd_gb", "hdd_gb"] and output["hdd_gb"].tolist() == [1000.0, 0.0, 1000.0]
    restored = pickle.loads(pickle.dumps(operation))
    assert restored.transform(frame.head(2))["ssd_gb"].tolist() == [128.0, 256.0]
