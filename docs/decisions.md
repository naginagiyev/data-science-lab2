# Decisions

Every judgment call made while carrying out `buildplan.md`: date, decision, reason, alternatives.

## 2026-10-11 · P0.2 · Git line endings

- Decision: add `.gitattributes` with `* text=auto eol=lf` (binary for png, pdf, xlsx, gz). Branch
  name `main`.
- Reason: Git for Windows has `core.autocrlf=true` at system level, so a clone would get CRLF files.
  LF everywhere keeps file bytes and hashes the same in every clone (benchmark manifests hash files).
- Alternatives: keep the system default (CRLF in working copies); set `core.autocrlf=false` locally
  (does not reach a fresh clone).

## 2026-10-11 · P0.3 · Commit hashes in progress.md

- Decision: a commit cannot contain its own hash, so each task line in `progress.md` gets its hash in
  the next commit (backfill).
- Reason: §0.3 asks for the hash after every task, and history must not be rewritten (no amend).
- Alternatives: amend each commit (rewrites history); a second "progress" commit per task (doubles
  the number of commits).

## 2026-10-11 · P0.3 · Test file names

- Decision: pytest collects `python_files = ["test*.py"]`; test files are named like
  `tests/testsettings.py`. `tests/torture.py` writes the torture files and `tests/testtorture.py`
  runs them.
- Reason: the user's rule allows no underscores in new Python file names, but pytest's default is
  `test_*.py`.
- Alternatives: `test_*.py` names (break the rule).

## 2026-10-11 · P0.3 · Style test from the start

- Decision: add `tests/teststyle.py` in P1.3. It checks the user's style rules (one empty line between
  definitions, imports sorted by length, file names, double quotes, no empty line at the end of a
  file) on project code, but not on the frozen copies `benchmark/legacy/v1/` and `edabench/legacy/`.
  P12.1 reuses it.
- Reason: the rules then hold on every commit instead of being fixed at the end.
- Alternatives: one style script at P12.1 only.

## 2026-10-11 · P0.3 · Folder names

- Decision: no code or fixture folder is called `data`, `models`, `sandbox`, `env`, `venv`, `build`
  or `dist`; test fixtures live in `tests/fixtures/`.
- Reason: `.gitignore` ignores these names at every depth.
- Alternatives: anchor the ignore patterns to the root (changes the frozen v1 ignore rules).

## 2026-10-11 · P0.5 · LLM check before the new environment

- Decision: the throwaway LLM check runs from the scratchpad with the old v1 `.venv` Python (it has
  `openai` and `python-dotenv`) and reads the key from `.env`. The spend ledger
  `benchmark/results/spend.json` starts by hand in the format that `benchmark/spend.py` uses later.
- Reason: P0.5 comes before P1.1, which builds the new environment.
- Alternatives: move the check after P1.1 (breaks the task order).
