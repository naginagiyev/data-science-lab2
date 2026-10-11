import os
import ast
import sys
import json
import shutil
import tempfile
import subprocess
import pandas as pd
from pathlib import Path

ALLOWED_IMPORTS = {"pandas", "numpy", "re", "math", "datetime", "string", "collections", "itertools", "functools", "statistics", "unicodedata"}
FORBIDDEN_MODULES = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib", "ctypes", "pickle", "marshal", "requests", "urllib", "http"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "breakpoint", "getattr", "setattr", "delattr", "vars"}
RUNNER = """import sys
import json
import importlib.util
import pandas as pd
spec = importlib.util.spec_from_file_location("generated", "generated.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
step = getattr(module, sys.argv[1])()
step.fit(pd.read_parquet("fit.parquet"))
for index in range(int(sys.argv[2])):
    frame = pd.read_parquet(f"in{index}.parquet")
    output = step.transform(frame.copy())
    if not isinstance(output, pd.DataFrame):
        raise TypeError("transform must return a DataFrame")
    if len(output) != len(frame) or not output.index.equals(frame.index):
        raise ValueError("transform changed the number of rows or the index")
    output.columns = [str(column) for column in output.columns]
    output.to_parquet(f"out{index}.parquet")
print(json.dumps({"ok": True}))
"""

# raised when generated code fails the checks or the run
class SandboxError(RuntimeError):
    pass

# the AST check of generated code: allowed imports only, no forbidden modules...
# or names, no dunder attributes, and one class with fit and transform; returns...
# a list of problems
def check_code(code, class_name):
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return [f"syntax error: {error.msg} (line {error.lineno})"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [(node.module or "").split(".")[0]]
        else:
            modules = []
        for module in modules:
            if module in FORBIDDEN_MODULES or module not in ALLOWED_IMPORTS:
                problems.append(f"import of '{module}' is not allowed; allowed: {', '.join(sorted(ALLOWED_IMPORTS))}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"the name '{node.id}' is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            problems.append(f"the attribute '{node.attr}' is not allowed")
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1 or classes[0].name != class_name:
        problems.append(f"the code must define exactly one class named {class_name}")
    else:
        methods = {node.name for node in classes[0].body if isinstance(node, ast.FunctionDef)}
        problems += [f"the class needs a method {name}(self, frame)" for name in ["fit", "transform"] if name not in methods]
    return list(dict.fromkeys(problems))

# the minimal environment of the sandbox process: no API keys, UTF-8 output,...
# fixed hash seed
def minimal_environment(folder):
    environment = {name: os.environ[name] for name in ["SYSTEMROOT", "PATH"] if name in os.environ}
    environment.update(TEMP=str(folder), TMP=str(folder), PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0")
    return environment

# kills a process and its children (taskkill /T on Windows)
def kill_tree(process):
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
    else:
        process.kill()

# runs generated code in a separate python process in a new temporary folder:...
# the class is fitted on fit_frame and transforms every frame of frames; the...
# outputs come back as DataFrames; a time-out kills the process tree
def run_code(code, class_name, fit_frame, frames, timeout=60, check=True):
    if check:
        problems = check_code(code, class_name)
        if problems:
            raise SandboxError("; ".join(problems))
    folder = Path(tempfile.mkdtemp(prefix="sandbox"))
    try:
        (folder / "generated.py").write_text(code, encoding="utf-8")
        (folder / "runner.py").write_text(RUNNER, encoding="utf-8")
        fit_frame.to_parquet(folder / "fit.parquet")
        for index, frame in enumerate(frames):
            frame.to_parquet(folder / f"in{index}.parquet")
        process = subprocess.Popen([sys.executable, "runner.py", class_name, str(len(frames))], cwd=folder, env=minimal_environment(folder), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, error = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(process)
            process.communicate()
            raise SandboxError(f"the generated code took longer than {timeout} s")
        if process.returncode != 0:
            message = error.decode("utf-8", errors="replace").strip().splitlines()
            raise SandboxError(f"the generated code failed: {message[-1] if message else 'no message'}")
        return [pd.read_parquet(folder / f"out{index}.parquet") for index in range(len(frames))]
    finally:
        shutil.rmtree(folder, ignore_errors=True)

# a JSON text of the allowed imports for prompts
def allowed_imports_text():
    return json.dumps(sorted(ALLOWED_IMPORTS))
