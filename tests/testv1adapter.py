import os
import json
import pytest
import subprocess
import numpy as np
import pandas as pd
from paths import root_dir, v1_dir

V1_PYTHON = v1_dir / ".venv" / "Scripts" / "python.exe"

# the v1 adapter must fail cleanly when the LLM cannot be reached: a fake...
# OPENAI_BASE_URL refuses every connection, so no real call is possible
@pytest.mark.slow
@pytest.mark.skipif(not V1_PYTHON.exists(), reason="the v1 environment is not installed")
def test_v1_adapter_fails_cleanly(tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"age": rng.integers(18, 80, 200), "plan": rng.choice(["a", "b"], 200), "Churn": rng.choice(["Yes", "No"], 200)})
    frame.iloc[:160].to_csv(tmp_path / "train.csv", index=False)
    frame.iloc[160:].drop(columns="Churn").to_csv(tmp_path / "test.csv", index=False)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("OPENAI")}
    environment.update(OPENAI_API_KEY="sk-fake-test-key", OPENAI_BASE_URL="http://127.0.0.1:9/v1", PYTHONPATH=str(root_dir), PYTHONIOENCODING="utf-8")
    command = [
        str(V1_PYTHON), str(v1_dir / "headless.py"), "--train", str(tmp_path / "train.csv"), "--test", str(tmp_path / "test.csv"),
        "--target", "Churn", "--task", "binary-classification", "--metric", "roc_auc", "--budget", "60", "--seed", "0",
        "--workdir", str(tmp_path / "w"), "--out", str(tmp_path / "out"),
    ]
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert meta["ok"] is False
    assert "Connection" in meta["error"] or "connect" in meta["error"].lower()
    assert not (tmp_path / "out" / "predictions0.csv").exists()
