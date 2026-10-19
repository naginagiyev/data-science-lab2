import io
import re
import tokenize
import subprocess
from paths import root_dir

FROZEN_PREFIXES = ("docs/background/",)
SKIPPED_FILES = {"docs/buildplan.md", "uv.lock"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".txt"}
PYTHON_NAME = re.compile(r"^[a-z0-9]+\.py$")
FOLDER_NAME = re.compile(r"^[a-z0-9]+$")
SPECIAL_NAMES = {"__init__.py", "__main__.py"}
DEFINITION = re.compile(r"^(async def |def |class )")

# lists the project files that git tracks or would track, without ignored...
# files and the frozen legacy copies
def project_files():
    command = ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    output = subprocess.run(command, cwd=root_dir, capture_output=True, check=True).stdout
    names = [name for name in output.decode("utf-8").split("\0") if name]
    names = [name for name in names if not name.startswith(FROZEN_PREFIXES) and name not in SKIPPED_FILES]
    return [root_dir / name for name in names if (root_dir / name).is_file()]

# returns the python files of the project
def python_files():
    return [path for path in project_files() if path.suffix == ".py"]

# returns the problems of one python file: double empty lines, missing empty...
# line or comment before a top-level definition, and unsorted imports
def layout_problems(path):
    lines = path.read_text(encoding="utf-8").split("\n")
    problems = []
    for index, line in enumerate(lines):
        if index > 0 and line.strip() == "" and lines[index - 1].strip() == "" and index < len(lines) - 1:
            problems.append(f"{index + 1}: two empty lines")
        if not DEFINITION.match(line) and not line.startswith("@"):
            continue
        before = index - 1
        if before >= 0 and lines[before].startswith("@"):
            continue
        if before < 0 or not lines[before].startswith("#"):
            problems.append(f"{index + 1}: no comment above the definition")
        while before >= 0 and lines[before].startswith("#"):
            before -= 1
        if before >= 0 and lines[before].strip() != "":
            problems.append(f"{index + 1}: no empty line before the definition")
    imports = []
    for index, line in enumerate(lines):
        if line.startswith(("import ", "from ")):
            imports.append(line)
            continue
        if imports and line.strip() != "":
            problems.append(f"{index + 1}: no empty line after the imports")
        if imports or (line.strip() and not line.startswith("#")):
            break
    lengths = [len(line) for line in imports]
    if lengths != sorted(lengths):
        problems.append("imports are not sorted by line length")
    return problems

# returns the string literals of a python file that use single quotes although...
# they contain no double quote; strings inside f-string fields (like...
# f"{row['name']}") are allowed
def quote_problems(path):
    problems = []
    tokens = tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline)
    starts = []
    texts = []
    for token in tokens:
        if token.type == tokenize.STRING and not starts:
            body = token.string.lstrip("rbuRBUfF")
            if body.startswith("'") and '"' not in body:
                problems.append(f"{token.start[0]}: single quotes")
        elif token.type == tokenize.FSTRING_START:
            starts.append(token)
            texts.append("")
        elif token.type == tokenize.FSTRING_MIDDLE and texts:
            texts[-1] += token.string
        elif token.type == tokenize.FSTRING_END and starts:
            start = starts.pop()
            text = texts.pop()
            if not starts and start.string.endswith("'") and '"' not in text:
                problems.append(f"{start.start[0]}: single quotes")
    return problems

# python files and their folders use lowercase letters and digits only
def test_file_names():
    wrong = []
    for path in python_files():
        relative = path.relative_to(root_dir)
        if path.name not in SPECIAL_NAMES and not PYTHON_NAME.match(path.name):
            wrong.append(str(relative))
        wrong += [f"{relative} (folder {part})" for part in relative.parts[:-1] if not FOLDER_NAME.match(part)]
    assert wrong == []

# text files end with exactly one newline (no empty line at the end)
def test_end_of_files():
    wrong = []
    for path in project_files():
        if path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        if text and (not text.endswith("\n") or text.endswith("\n\n")):
            wrong.append(str(path.relative_to(root_dir)))
    assert wrong == []

# one empty line between definitions, a comment above each top-level...
# definition, and imports sorted by line length
def test_python_layout():
    wrong = {}
    for path in python_files():
        problems = layout_problems(path)
        if problems:
            wrong[str(path.relative_to(root_dir))] = problems
    assert wrong == {}

# strings use double quotes
def test_double_quotes():
    wrong = {}
    for path in python_files():
        problems = quote_problems(path)
        if problems:
            wrong[str(path.relative_to(root_dir))] = problems
    assert wrong == {}
