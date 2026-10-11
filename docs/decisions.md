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

## 2026-10-11 · P0.5 · LLM availability, parameters and prices

- Checked: `gpt-4.1-nano` (served as `gpt-4.1-nano-2025-04-14`) and `gpt-4.1-mini` (served as
  `gpt-4.1-mini-2025-04-14`) both answer a strict structured-output call (`chat.completions.parse`
  with a Pydantic model) with `temperature=0`, `seed=0` and `max_tokens`. No parameter was rejected.
- Prices (OpenAI pricing page, standard tier, per million tokens, checked 2026-10-11):
  `gpt-4.1-nano` input 0.10, cached input 0.025, output 0.40; `gpt-4.1-mini` input 0.40, cached
  input 0.10, output 1.60. They equal the defaults of `buildplan.md` §2.5, so
  `prices_per_million_tokens` stays as planned. Costs are computed without the cached-input
  discount (an upper bound).
- Cost of the check: 2 calls, 65 input and 9 output tokens each, $0.0000505 in total, written to
  `benchmark/results/spend.json` (category `dev`).

## 2026-10-11 · P1.1 · Environment and the architecture check

- Decision: the main environment is CPython 3.12.14 x86_64 (`cpython-3.12-windows-x86_64-none` in
  `.python-version`; uv accepted the key). The architecture check is
  `sysconfig.get_platform() == "win-amd64"` instead of `platform.machine() == "AMD64"`.
- Reason: on this Windows-on-ARM machine Python 3.12 asks Windows for the native CPU, so
  `platform.machine()` prints `ARM64` even in the emulated x86_64 interpreter. `sys.version` says
  `MSC v.1944 64 bit (AMD64)`, `sysconfig.get_platform()` says `win-amd64`, the process sees
  `PROCESSOR_ARCHITECTURE=AMD64`, and the x86_64-only wheels (LightGBM, CatBoost) load and train.
- pytest: `addopts = -m "not slow and not llm"`, so `uv run pytest -q` is the fast suite; run the
  others with `-m slow` or `-m llm`. Ruff skips the frozen `benchmark/legacy` and `edabench/legacy`.
- Versions (from `uv.lock`, 2026-10-11): pandas 2.3.3, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1,
  lightgbm 4.7.0, catboost 1.2.10, optuna 5.0.0, pydantic 2.14.0, openai 3.28.0, python-dotenv 1.2.4,
  tenacity 9.2.1, joblib 1.6.0, pyarrow 26.0.0, openpyxl 3.1.5, rich 15.0.0, questionary 2.1.1,
  tabulate 0.10.0, smolagents 1.26.0, skrub 0.11.0, matplotlib 3.11.2, autorank 1.3.0, openml 0.15.1;
  dev: pytest 9.1.1, ruff 0.17.0, psutil 7.2.2. autorank installed without conflicts.
- Alternatives: none needed; the `uv venv` fallback of the plan was not required.

## 2026-10-11 · P1.3 · Core files

- Decision: `schemas.py` defines the `Task` literal itself until edatool is vendored (P3.1), then it
  imports it from `edatool.schemas`. Settings sections forbid unknown keys, and `--set` only changes
  keys that exist, so a typo fails at once. The settings hash is the first 16 hex digits of a SHA-256
  of the canonical JSON. `metrics.json` has the primary and reported metrics per task and the list of
  "lower is better" metrics.
- Decision: the style test requires a `#` comment above every top-level function and class, not
  above methods or nested functions. This is exactly what the vendored edatool code does (§0.6 says
  to copy its style).
- Decision: `main.py` is a placeholder until P1.6 rewrites it, so the style test passes now.

## 2026-10-11 · P1.5 · LLM layer

- Decision: the OpenAI provider sends strict structured outputs through `chat.completions.create`
  with `response_format` built by the SDK helper `type_to_response_format_param` (a private module of
  the pinned openai 3.28), and returns the raw JSON text. The client validates the text itself with
  Pydantic and with an optional `check` callback of the caller (for example the cleaner's plan
  checks); both kinds of error are sent back to the LLM (up to `schema_retries`).
- Reason: one validation path for every provider (the fake provider and, in semester 2, providers
  without strict schemas), and the raw answer is always logged.
- Decision: the API key is read from the environment or with `dotenv_values(.env)`; it is never put
  into `os.environ`, so subprocesses never inherit it.
- Decision: the ledger `benchmark/results/spend.json` is shared by processes through a lock file
  (`spend.json.lock`, stale after 60 s) and written atomically. A call is refused when its estimate
  (prompt characters / 3 input tokens + 1,500 output tokens) would cross the category cap or the
  total cap, or when the run's call or cost budget is used up. Cached answers cost nothing and are not
  counted against `max_calls_per_run`.
- Decision: each line of `llm.jsonl` also stores the rendered messages and the raw answer (research
  data for decision-quality analysis and debugging).
- The `llm`-marked real test (`tests/testllmreal.py`) cost $0.0000107 with `gpt-4.1-nano`.

## 2026-10-11 · P1.6 · Orchestrator shape

- Decision: every stage is a function `stage(state)` that receives one `RunState` (settings, run
  folder, `RunContext`, `Budget`, LLM client and a dictionary of objects passed between stages, such
  as frames and reports). `run_stage(state, name, function, fallback)` wraps it as in §3.16; a
  fallback returns a short label that is stored in `StageStatus.fallback`. `StopRun` is the only
  exception that ends a run (the hard stops of §3.1).
- Reason: §3.16 says `stage(ctx, data, settings)`; one state object keeps the signatures short and
  lets later stages read what earlier stages produced without global state.
- Decision: files written by code use LF line endings (`newline="\n"`), so run folders and ledgers
  are byte-identical on every machine.
- Alternatives: separate arguments per stage (long signatures that change with every new stage).
