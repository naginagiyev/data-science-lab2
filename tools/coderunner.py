import os
import sys
import subprocess
from paths import sandbox_dir, PROJECT_ROOT

class CodeRunner:
    def __init__(self, file_name: str, timeout: int = 1800):
        self.file_path = sandbox_dir / file_name
        self.timeout = timeout

    def add(self, script):
        header = "".join([
            "# initialize paths\n",
            "from paths import data_dir\n",
            "from paths import sandbox_dir\n",
            "from paths import models_dir\n\n",
            "# Generated code starts here\n"
            ])

        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(header + script)

    def get_output(self):
        # run with the same interpreter as the pipeline and make the project root importable,
        # otherwise the generated script can't import `paths` and fails before it even starts
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PROJECT_ROOT), env.get("PYTHONPATH")]))
        env["MPLBACKEND"] = "Agg"
        try:
            result = subprocess.run(
                [sys.executable, str(self.file_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                cwd=PROJECT_ROOT,
                env=env,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return {"operation": "error", "output": f"Script timed out after {self.timeout} seconds. Make it faster or remove blocking calls."}
        if result.returncode != 0:
            return {"operation": "error", "output": result.stderr}
        return {"operation": "success", "output": result.stdout.strip() if result.stdout else "No output"}
