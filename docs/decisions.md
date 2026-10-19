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

## 2026-10-11 · P2.1 · Harness details

- Decision: systems run as modules (`python -m benchmark.systems.<name>`) with the project root on
  `PYTHONPATH`, 4 threads (`OMP_NUM_THREADS` and friends), `PYTHONIOENCODING=utf-8` and **no**
  `OPENAI_API_KEY` in the environment (an LLM adapter reads it from `.env` itself, inside its own
  client). `benchmark/systems/common.py` uses only pandas and numpy, so the separate baseline
  environments can import it too.
- Decision: an adapter writes `predictions<i>.csv` for the i-th test file (`predictions0.csv` is the
  deployment file; Group D adds `predictions1.csv` for the naive file) and `meta.json`.
- Decision: the scorer uses the sorted label keys of the training and test labels as classes (a class
  that appears only in the test rows still gets a column); missing probability columns are zeros and
  rows are renormalized; a file with only a `label` column is scored as hard probabilities (v1's real
  behaviour, B4).
- Decision: outer splits are saved per dataset and seed in `cache/benchmark/splits/` with a hash of the
  clean table, so every level of a dataset uses the same rows. The fixed creditcardfraud sample
  (50,000 rows, stratified, seed 0) is the same for every seed.
- Decision: the fast tests never read `data/` (it is git-ignored, so a fresh clone has no data);
  they build small synthetic tables. Tests that need the real dev files are marked `slow`.

## 2026-10-11 · P2.2 · Main-environment baselines

- Decision: the `lightgbm` baseline replaces the JSON special characters `[ ] { } " : ,` in column
  names with `_` (LightGBM refuses them); everything else is the plan's "pd.read_csv defaults, object
  columns as category, default parameters". The `skrub` baseline is
  `skrub.tabular_pipeline(HistGradientBoosting*(random_state=seed), n_jobs=4)` (skrub 0.11).
- Reason: a user would rename such columns at the first error; failing on a name is not what the
  benchmark measures.

## 2026-10-11 · P2.3 · Separate baseline environments

- Installed with `uv run python -m benchmark.envs.install <name>` into `benchmark/envs/<name>/.venv`
  (CPython 3.12.14 x86_64, `win-amd64`). FLAML: `flaml[automl]==2.7.0` (xgboost 2.1.4); AutoGluon:
  `autogluon.tabular[lightgbm,catboost,xgboost,fastai]==1.6.3` (torch 2.13.0, catboost 1.2.10,
  lightgbm 4.7.0, xgboost-cpu 3.3.0, pandas 2.3.3; 1.5 GB). Both installed on Windows without WSL.
- Decision: AutoGluon runs with `presets="medium_quality"`, `num_cpus=4`, `num_gpus=0` and
  `ag_args_ensemble={"fold_fitting_strategy": "sequential_local"}` (no Ray), its model folder in the
  short workdir `cache/w/<8 hex>/ag`. The `[all]` extra (Ray, vision and text models) is not installed.
- Decision: every adapter gives its system `budget × 0.9 − time already used` seconds to fit
  (`common.fit_seconds`), keeping 10% for predicting and writing. The runner measures the wall clock
  of the whole process; FLAML overran a 60 s budget by up to 17 s (it retrains the best model at the
  end), which stays inside the hard time-out (budget × 1.25 + 60 s). The measured seconds are in the
  results, so the overrun is visible.
- 60 s check on the 4 local datasets (seed 0): all cells `ok` (`benchmark/results/p2-flaml.csv`,
  `p2-autogluon.csv`).

## 2026-10-11 · P2.4 · v1 adapter (built, never run with the real LLM)

- `benchmark/legacy/v1/.venv` was created from the frozen `uv.lock` with
  `uv sync --project benchmark/legacy/v1 --locked --no-install-project` (Python 3.10.18 x86_64), so
  nothing is written into the frozen folder except `.venv`.
- `benchmark/legacy/v1/headless.py` copies the frozen code to `<workdir>/v1` (without `.venv`), creates
  `data/`, `models/` and `sandbox/`, makes Python import the v1 modules from the copy, copies the train
  file to `data/dataset.csv`, and runs EDA, ConfigFiller, the processing plan and code, the v1 split,
  the training plan and code, and the tuning loop, with the given target and task. It then loads
  `models/preprocessor.pkl` (cloudpickle) and `models/model.onnx`, transforms the test features (with a
  placeholder target, because v1 keeps the target in its processed data) and uses the ONNX probability
  output when there is one, otherwise the labels as hard probabilities (v1's behaviour, B4).
- Registry: `timeout` 1,200 s (20 minutes) and `needs_key`: the runner passes `OPENAI_API_KEY` in the
  environment to this system only (v1 reads it from the environment; its copy has no `.env`).
- Known limitation, kept on purpose: v1 maps text targets to integers outside its pipeline and does
  not save the mapping (C5), so its class columns may not match the true labels; the scorer then
  fails the cell and gives it the dummy score. This is v1's real behaviour.
- Test (`tests/testv1adapter.py`, marked `slow`): with `OPENAI_BASE_URL=http://127.0.0.1:9/v1`, v1's EDA
  runs and the first LLM call (ConfigFiller) raises `APIConnectionError`; the adapter exits with code
  1 and writes `meta.json` with `ok: false` and the error. No real LLM call was made.

## 2026-10-11 · P2.5 · TabPFN skipped

- Decision: the optional `tabpfn` baseline is skipped (`[-]`), as the plan says for gated weights.
- Reason: `tabpfn==9.1.0` installs on Windows x86_64, but the first `fit` calls
  `ensure_license_accepted` for its Hugging Face weights and starts a browser login (it may have
  opened a browser tab); without an accepted license and a token, no weights download. The
  environment (about 1 GB) was deleted. `benchmark/envs/tabpfn.txt` stays, so the user can install it
  later with `uv run python -m benchmark.envs.install tabpfn` after accepting the license.
- AIDE is not installed in this plan (§2.5).

## 2026-10-11 · P3.2 · Reader and glance

- Decision: the pattern statistics live in `edatool/core/patterns.py` from the start (the plan allows
  them in `tools/glance.py` first); edatool does not use them yet, so its behaviour is unchanged until
  the P4.4 detectors.
- Pattern shares (on at most 2,000 non-empty values): plain number, number with unit or suffix (top 5
  units), currency, percent, date-like (top 3 of 7 date patterns), boolean-like, packed (two or more
  numbers, not a date, no currency) with the top 3 digit templates, and long text (5 words or more).
- Glance uses edatool's `infer_schema` on at most 50,000 rows (target unknown at that point). Missing
  share = blank cells and edatool's hidden-missing tokens. The prompt line per column has the type,
  missing share and tokens, uniques, up to 4 examples, the range and the pattern shares above 5%.

## 2026-10-11 · P3.3 · Splitter

- Decision: `SplitInfo.test_rows` are positions in the read table (after rows with an empty target and,
  in demo mode, exact duplicates are removed); `SplitInfo.folds` are the validation positions of each
  fold in the training frame (the table without the test rows, re-indexed from 0).
- Decision: time folds are contiguous blocks of the time-sorted training rows (`KFold` without shuffle
  on the sorted order); the time key is parsed as numbers when at least 90% parse, otherwise as dates
  (`format="mixed"`); unparseable values sort first. Group folds use `GroupKFold(shuffle=True)`
  (scikit-learn 1.9). If edatool's `presplit` fails, classification is stratified and regression random.

## 2026-10-11 · P3.4 · Typing step and generic stage

- Decision: `TypingStep` works only on object (text) columns; operations that produce numbers, dates
  or pandas categories are left alone (so `as_category` must output the `category` dtype). It uses
  pandas' own NA list (`pandas._libs.parsers.STR_NA_VALUES`, exact match, so `"?"` and `" "` stay),
  `pd.to_numeric` for numbers (it strips spaces and reads `inf`, like `read_csv`), and turns columns of
  only `True`/`False` spellings without missing cells into 0/1 (pandas makes them `bool`). A parity
  test compares it with `pd.read_csv` defaults.
- Decision: `GenericStage` is skrub's `TableVectorizer` (skrub 0.11 works with scikit-learn 1.9, so no
  `ColumnTransformer` fallback was needed): `cardinality_threshold=31` (at most 30 levels is low),
  `StringEncoder(n_components=30)` for high cardinality, `DatetimeEncoder(resolution="day",
  add_weekday=True, add_total_seconds=True)`; low cardinality is an `OrdinalEncoder` (unknown -1,
  missing NaN) for tree models and skrub's default one-hot for linear models, which also get a median
  imputer and a standard scaler. Infinite numbers become NaN before vectorizing. skrub's own null
  strings (for example `"?"`) become missing inside the vectorizer, exactly as in the `skrub` baseline.
- Decision: output names are sanitized (`[ ] { } " : ,` → `_`, duplicates get `_2`, `_3`), the map to
  the original names is kept in `name_map_`; a table with no usable column gets one constant
  `no_features` column, so every model can still fit (it then predicts the prior or mean).

## 2026-10-11 · P3.5 · Model engine

- Decision: the preparation (agent steps + typing + generic stage) is fitted once per fold and per
  preparation kind (`tree` or `linear`) and cached; Optuna trials then refit only the model. The final
  refit fits the preparation and the model again on all training rows (row filters applied).
- Decision: time: 10% of the stage budget is kept for the ensemble and the refit; the rest is shared in
  candidate order (dummy, linear, LightGBM, CatBoost), each family getting its plan share (5%, 50%, 35%)
  of what is left, so unused time passes on. The default parameters are always the first trial and may
  use the whole stage deadline; later trials stop at the family deadline. Each fold of a trial gets an
  equal share of the time left (LightGBM callback raises `EarlyStopException`, CatBoost
  `after_iteration` callback returns False), so a cut treats all folds alike.
- Decision: boosted models use early stopping (50 rounds) on a 10% inner split of the fold's training
  part (stratified when possible); the refit uses the mean best number of trees over the folds.
  Optuna catches `ValueError`, `RuntimeError`, `LightGBMError` and `CatBoostError` per trial; if Optuna
  itself fails, the family is evaluated once with default parameters.
- Check on kinghousedata (17,298 training rows, 5 group folds, 60 s): 58.7 s used; CV RMSE dummy 374k,
  linear 205k, LightGBM 121k, CatBoost 118k; the CatBoost + LightGBM average (117k) was chosen.

## 2026-10-11 · P3.6 · Evaluator, FittedModel, report, predict

- Decision: `FittedModel` keeps the `TrainedModel` (preparation pipelines + models), the label encoder,
  the raw input columns, task, metric, plan, an optional target transform, versions and the settings
  hash. Input (a path or a DataFrame) is read as the raw view; missing input columns become empty and
  extra columns are ignored, both with a warning. `predict_frame` gives one probability column per
  class plus `label`, or `prediction`.
- Decision: the evaluator scores the locked test rows once, after the refit; a test class the model
  never saw gets a zero probability column (renormalized), so log-loss stays defined. The dummy test
  score comes from the training labels.
- Decision: the framer's rule part (`agents/framer.py`: candidate ranking with name hints from
  `settings.framer.name_hints`, last-column bonus, never an id/constant/empty/text column; rule task)
  is built now because the frame stage needs it without flags; the LLM part comes in P7.1.
- Decision: new setting `engine.max_search_rows` (50,000): on larger tables the Optuna search runs on a
  fixed stratified sample (folds mapped to it) and the final refit uses all rows. Reason: on the full
  creditcardfraud table (226,980 training rows) the 5-fold preparation alone used the whole 60 s
  budget and only the dummy model was trained. With the sample: test ROC-AUC 0.983, but the model
  stage took 79 s instead of about 60 s (the refit on all rows is slow); P8.3 measures this at 300 s.
- `generic` on the 4 local datasets (60 s budget, demo mode): ibmchurn test ROC-AUC 0.8527 (dummy
  0.5000), kinghousedata test RMSE 105,029 (dummy 336,712; group split on `id`), winequality test
  log-loss 0.8753 (dummy 1.2021), creditcardfraud (full table) test ROC-AUC 0.9833 (dummy 0.5000).

## 2026-10-11 · P4.1 · Auditor baseline

- Decision: type accuracy is reported on the 4 draft label files (`--drafts`), clearly marked as
  unreviewed; without `--drafts` the command uses no file.
- Decision: ydata-profiling and cleanlab live in `benchmark/envs/edabench` (pandas<3, numpy, scipy,
  scikit-learn, joblib, pydantic, python-dotenv, tabulate, matplotlib, openml, cleanlab[datalab]>=2.9,
  ydata-profiling>=4.18.4, setuptools<81); the `baselines` experiment runs with that Python and the
  project root on `PYTHONPATH`.

## 2026-10-11 · P4.2 · Windows-safe edatool output

- Decision: a `logging.Filter` on the `edatool` logger (`SafeConsoleText` in `edatool/core/progress.py`)
  looks at the console handlers (not file handlers) of the logger and its parents; when one of them
  cannot write unicode (for example cp1252), the message is made safe for that encoding: the symbols
  `✓ ✗ · → ─ └─ × … ≥ ≤` get ASCII replacements and any other character becomes `?`. The CLI also
  reconfigures stderr to UTF-8 with `errors="replace"` (stdout already was).
- Reason: in library mode the host's handlers decide the encoding; the filter keeps the host's
  logging free of `UnicodeEncodeError` (the `../II` crash). With a cp1252 console the same audit gave
  2 logging errors without the filter and 0 with it (test `tests/testedatooloutput.py`).
- Alternative: an ASCII-only output everywhere (loses the symbols on UTF-8 consoles).

## 2026-10-11 · P4.3 · Loader robustness

- Decision: text files are decoded by BOM first (UTF-8 BOM → `utf-8-sig`, UTF-16 BOM → `utf-16`),
  then strictly as UTF-8, cp1252 and finally latin-1. The delimiter is chosen among the sniffer's
  guess and `, ; \t |` by the consistency of the field counts on the first 300 lines (a field count
  above 1 first, then the share of lines with the modal count). The modal width prefers counts above 1,
  so title and note rows never decide it.
- Decision: the header is the first row with the modal field count and mostly non-numeric text; the
  non-empty rows above it are title rows. Below the table, rows (from the end) whose field count
  differs or that are notes are footer rows. A note is a row whose only non-empty cell is non-numeric
  text in the first column; so a sparse last data row such as `4,,` is kept. The table is still parsed
  by pandas (`skiprows` = header line); footer rows are removed afterwards. Excel files use the same
  logic on the cell grid.
- Decision: empty files and header-only files raise `ValueError` with a clear message (the pipeline's
  reader turns it into a hard stop). Title rows, footer rows and a non-UTF-8 encoding are reported in
  `attrs["edatool"]` and as an A01 `file_layout` issue.
- Check: the 4 local datasets load with the same shapes as before (creditcardfraud 5.6 s).

## 2026-10-11 · P4.4 · Semantic-mess detectors

- B07 numbers with units (categorical, text and id columns): at least 90% of at least 20 present
  values match number + unit (`edatool/core/patterns.unit_of`; percent and currency are B01 cases).
  Evidence: unit counts, share, examples; fix `convert_type {"to": "numeric", "units": [...]}`;
  needs review when there are 2 or more units (the factors must come from the agent or the LLM).
- B08 packed values: the top 3 digit templates (digits → `#`) cover at least 80% of the values and each
  has at least 2 numeric slots; columns that are mostly dates are excluded. Fix `extract_pattern` with
  a ready regex built from the top template (spaces become `\s*`) and names `<column>_1, _2, ...`.
  `extract_pattern` was added to `FixAction`. Priority should_fix, no review.
- B09 mixed date formats: at least 90% of the values match one of 7 date patterns and at least 2
  formats have a share of at least 1%. For `dd/mm/yyyy` vs `mm/dd/yyyy` the values decide the order
  when a part is above 12; otherwise the order is ambiguous and needs review. Fix
  `convert_type {"to": "datetime", "formats": [strptime formats], "dayfirst": ...}`.
- New hidden-missing tokens: `n.a.`, `not available`, `--`, `#N/A`, `nil`. The two agent prompts now
  contain `{fix_actions}`, filled from `FixAction` by `session.prompt_text` (C4).
- The 4 local datasets give no B07, B08 or B09 issue (no new false alarms).

## 2026-10-11 · P4.6 · Auditor re-benchmark

- Decision: the four new injected kinds run after the old ones with `default_rng(seed + 1000)` on unused
  positive numeric columns, so the old kinds' data and random choices are unchanged; old-kind precision
  is computed without issues on the new columns (scratch re-scoring, numbers in
  `docs/edatool-baseline.md`).
- Decision: B07's number-with-unit pattern accepts an exponent (`1.2e+06kg`); found because the first
  re-run missed 2 of 3 mixed-unit columns.

## 2026-10-11 · P5.1 + P5.2 · Operations (one commit)

- Decision: P5.1 (column operations) and P5.2 (row and target operations) are committed together,
  because the registry and the shared operation tests cover both kinds.
- Operations are scikit-learn transformers with a Pydantic parameter model (`Params`, unknown
  parameters are errors). `transform` keeps the index and row count (checked in the base class) and
  records `stats_` (present values, parse failures with up to 10 examples, changed values, seconds).
  Row operations only return a keep-mask for training rows (`RowFilter`); calling `transform` on them
  raises. `map_target_labels` refuses mappings between labels whose spelling keys differ (lowercase, no
  punctuation, no repeated spaces), so classes are never merged; `log_target` is undone exactly.
- Choices left open by Appendix B: `parse_number` with `percent=true` divides values that end in `%` by
  100; `parse_units` reads values without a unit with `default_factor` (else they fail);
  `date_difference` treats 4-digit numbers as years and other values as dates (fractional years);
  `as_category` outputs the pandas `category` dtype (the typing step leaves it alone).

## 2026-10-11 · P5.3 · Translator

- Decision: the protected codes (A02 saved index, C02 hidden missing tokens, A04 exact duplicates, F03
  target spellings) stay protected even when their issue is still an open review item (no agent in the
  `rules` variant): these steps are safe by construction. Other open review items become gated steps.
- Decision: A05 (conflicting duplicates) names the target but is translated as a gated, training-only
  row step (`drop_conflicting_rows`); every other issue on the target is skipped (rule 12).
- Decision: `consider` issues are not translated; D02 limits come from the flagged values in the
  evidence (`above` just under the smallest flagged high value, `below` just over the largest flagged
  low value); B07 with one unit gets factor 1; B07 with two or more units and no factors is skipped and
  left to the cleaner; steps are ordered replace/missing first, then parsing, then categories,
  indicators, grouping, text features and drops.
- Check on deterministic audits: ibmchurn gives replace_values `" "` → missing (protected),
  parse_number TotalCharges (gated), drop_duplicate_rows (protected), drop_conflicting_rows (gated);
  customerID is dropped by the leak guard (protected). kinghousedata gives parse_datetime `date`,
  as_category `zipcode`, set_missing_where for `yr_renovated` 0 + indicator, D02 limits, drop
  `waterfront` (gated); `id` is dropped by the leak guard and used as the group key. winequality gives
  no target step (F05 is no_action).

## 2026-10-11 · P5.4 · Compiler, data checks, verify

- `validate_plan` walks the steps in order and tracks the columns that exist (inputs must exist,
  outputs are added, removed columns disappear); new output names may not overwrite existing columns.
- Data checks run on the gate sample (at most `gate.max_rows` training rows) and the gate folds: each
  step is fitted on the training part and applied to the validation part; a failing step is reported
  once and left out for the later steps, so all failures of a plan come out in one pass. The target
  check compares every output column with the raw target values (and the target name).
- Verify runs on all training rows after fitting the agent steps; snapshots carry the raw target; P01
  and P02 are ignored; it returns still-open and new must-fix issues and the resolved count.

## 2026-10-11 · P5.5 · Leak guard

- Evidence: structural = A02 and E03 (not dismissed) and the framer's id columns (plus an id-like group
  key of the split); statistical = G01, G02, G04 in any status; semantic = G03 (agent name review) or
  an agent confirmation of G01/G02/G04 (a review item with status `confirmed` and a `drop_column` fix).
  Decisions follow the §3.6 table; the target is never dropped; all drops form one protected
  `drop_columns` step with source `leakguard`.
- Decision: with the guard off (`full-noleakguard`) only the structural drops (ids, saved index)
  remain, because ids are not leaks; the ablation then measures leaks only.

## 2026-10-11 · P5.6 · Validation gate

- The quick model is LightGBM (300 trees, learning rate 0.05, 31 leaves, `gate.n_jobs` = 4, a new
  setting) behind the typing and generic stages (tree kind), on a stratified sample of at most
  `gate.max_rows` training rows with `gate.folds` folds of the run's split strategy. The plan score is
  the mean of the fold scores; row filters touch fold-training parts only.
- Decision: a gated step whose removal makes the plan invalid (a later step needs its output) is not
  tested and stays; when the stage deadline passes, the remaining steps are not tested and stay. Harmful
  steps (effect below minus the tolerance) are removed together, the reduced plan is scored once more,
  and it is kept when it is not worse than P0 within the tolerance; otherwise P0 is the final plan.
- Tolerances as in the plan: ROC-AUC 0.002 absolute; log-loss and RMSE 0.5% relative to the P0 score.

## 2026-10-11 · P5.7 · Rules end to end; gate speed

- Decision: the default `gate.max_rows` is 10,000 instead of 20,000 (plan §2.5). Reason: the gate gets
  about 45 s inside a 120 s run (stage limit = time left minus `min_model_seconds`), and one plan score
  on kinghousedata took 4.1 s at 17,290 rows (about 11 s with a second run in parallel) and 2.9 s at
  10,000 rows with the same score; with 9 gated steps the per-step tests did not finish.
- Decision: per-step tests run riskiest first (drops, missing-value rules, row filters, grouping,
  clipping, replacements, log transforms, then parsing). When single-step tests would not fit in the
  time left (estimated from the P0 and P1 timings), steps are tested in groups of the same operation (one
  test per operation type); every step records how it was tested. When the reduced plan still loses to
  P0 and some steps were untested, P0 plus the tested steps that did not hurt is scored once more.
- Result (120 s, benchmark mode, seed 0): `rules` vs `generic`: ibmchurn ROC-AUC 0.8528 vs 0.8517;
  kinghousedata RMSE 100,064 vs 102,687 (first run before the gate change: 102,791 with all gated steps
  rejected); winequality log-loss 0.8245 vs 0.8076 (protected training-only duplicate drop of 240 rows);
  creditcardfraud ROC-AUC 0.99997 vs 0.99996.
- The run folder now also holds `leaks.json`, `plans.json` (every version), `checks.json` (data checks
  and verify), `gate.json`, `verify/` and, in benchmark mode, `prepared.parquet`.

## 2026-10-11 · P6.1 · Dataset registry

- Checked against OpenML on 2026-10-11: all ids of Appendix D exist and their names, default targets
  and tasks match (binary datasets have 2 classes, multi-class 3). Versions, licences, sizes,
  readable-name shares and table hashes are in `benchmark/results/datasets.json`; data cards in
  `docs/datasets.md` are generated from it by `benchmark/datacards.py`. No spare was needed.
- Licences: all allow research use; miami_housing is CC BY-NC-SA (non-commercial; kept, because the
  thesis is non-commercial research). "Public" (OpenML's label) is taken as usable.
- Readable names: SDSS17 (54%: `u g r i z` photometric bands) and diamonds (67%: `x y z` dimensions) are
  below an 80% word-like share, but these are standard names with a known meaning, so both stay in
  Group A. credit-approval (spare C) is fully anonymized (`A1`...`A15`) and is not a usable replacement.
- Fixed samples: SDSS17 and diamonds 30,000 rows (as planned); also Food_Delivery_Time and adult 30,000
  rows (large natural-mess tables; same rule).
- The TabArena bank-marketing (did 46910) has no `duration` column, so its known leak is not there.

## 2026-10-11 · P6.2 · Corruptor

- Numbers are rewritten with `Decimal` from their original text (currency with thousand separators,
  percent, `k`/`M` suffixes, unit conversions), so every format corruption is undone exactly
  (`restore`); types that destroy values (fake missing, impossible values, plain missing) are restored
  from the stored original cells (`cells.parquet`: row, column, original, corrupted, type).
- Allocation: each type wants 25/50/75% of its eligible columns (at least one); columns are handed
  out one per type and round in a seeded order, so all types get columns; one type per column; at
  most 80% of the features; never the target. Levels are nested: a level-independent generator fixes
  the column order of each type and a row order per column, and level k takes the first columns and
  rows, so a higher level contains the lower ones (cells grow strictly with the level in the tests).
- Eligibility choices: units and mixed units need a non-negative numeric column with at least 20
  values; impossible values need a non-negative numeric column with at least 5 values; packed pairs
  need two numeric columns without empty cells; dates need a text column that is at least 90% dates;
  spellings need 3 to 50 text levels; booleans need exactly 2 levels.
- Unit families from the column name: months/days, age years/months, weight kg/g, length m/cm, time
  h/min, volume l/ml; otherwise `k` and `M` thousands/millions. Type 1 styles: currency (money names or
  values below 1,000; minority "1,250 USD"), percent (rate names, values up to 100), thousands `k`.
- Type 9 adds `customer_id` (level 1) and also `Unnamed: 0` (level 2 and 3). Type 10 (layout) is only
  applied when files are written: BOM (level 1), title rows and a footer note (level 2), `;` with
  decimal commas and cp1252 (level 3); the robust loader reads all three back (test).
- Files of a single type live in `level<k>-<type>/`; `truth.json` has the items, the train and test row
  positions and the file hashes.

## 2026-10-11 · P6.3 · Leak generator

- Decision: "the leak agrees with the positive class in 98% (strong) or 80% (moderate) of rows" is
  read as: the flag or date exists for 98% / 80% of the positive rows and never for negative rows
  (it is written after the event, like a refund after a confirmed fraud). With 2% of all rows flipped
  instead, the one-feature AUC of a strong flag is about 0.98, exactly G01's threshold, and detection
  would be chance. The multi-class grade style keeps 98% / 80% correct grades. Regression: target x
  U(0.97, 1.03) (strong) or target plus normal noise with sd = 0.75 x sd(target) (correlation about 0.8).
- Strengths: strong for credit-g, Fitness_Club, Bank_Customer_Churn, maternal_health_risk,
  Another-Dataset-on-used-Fiat-500; moderate for Marketing_Campaign, E-CommereShippingData,
  concrete_compressive_strength, healthcare_insurance_expenses, diamonds (the pilot strengths of
  Appendix D are kept).
- Check (deterministic audit of each Group D table with its leak): G01 finds all 5 strong leaks (G02
  also for two flags); of the moderate ones only the date leak is found (G04); the other four are not
  found by statistics alone, so they need the agent's name review (G03), which is what E3 measures.
- Speed: the pipeline tests now use a 12 s budget and the engine deadline test 12 s (fast suite under
  the 3-minute limit).

## 2026-10-11 · P6.5 · Group C (natural mess)

- Checked on the downloaded versions (`benchmark/results/natural-mess.md`, generated by
  `python -m benchmark.datacards --natural`): HR_Analytics (`>20`, `never`, `>4`, size ranges such as
  `50-99` and `10/49`, `"nan"` as text), in_vehicle_coupon_recommendation (income ranges as text,
  `50plus`, `1d`/`2h`; `time` is already numeric in this version), heloc (special values -7, -8, -9 in 23
  columns), bank-marketing (`unknown` tokens, `pdays = -1`; no `duration` column) are messy.
- Decision: Food_Delivery_Time is replaced by the spare employee_salaries (did 42125; dates as
  `mm/dd/yyyy` text, a mostly empty job-title column), because the TabArena version of Food_Delivery_Time
  is cleaned (only trailing spaces in two categories). Group C is now HR_Analytics,
  in_vehicle_coupon_recommendation, heloc, bank-marketing and employee_salaries; the pilot natural
  datasets stay HR_Analytics, in_vehicle_coupon_recommendation and heloc.

## 2026-10-11 · P7.1 · Framer

- The LLM framer runs only in the LLM variants and only when a flag is missing or the rule task is
  ambiguous (integers with 3 to 20 levels); the benchmark always passes both flags, so it makes no
  call there. Its answer is checked by code (target in the candidate list, existing column names, a
  two-valued target is binary, id columns must be id-like) and errors go back to the LLM (2 retries);
  then the rules win. `--ask` asks the user with questionary when the confidence is below 0.6.
- Decision: rule candidates weigh earlier name hints more (`target`, `label`, `class`, ... before
  `grade` or `rating`), and the few-levels bonus is small (0.2). Reason: the first real check picked
  `grade` instead of `price` on kinghousedata, because both words were hints of equal weight.
- Decision: an LLM id column is accepted only when the glance types it as id, or when its name contains
  an id word (id, key, code, number, no, uuid, guid, index, nr) and at least 90% of its values are
  unique. Reason: nano listed all wine features as ids, and `Time` for creditcardfraud.
- Real check (gpt-4.1-nano, 5,000-row samples, no flags): ibmchurn → Churn (binary, id customerID),
  kinghousedata → price (regression, id id), winequality → quality (multi-class), creditcardfraud →
  Class (binary); 4 of 4 right; two invalid id lists were corrected by the retry.

## 2026-10-11 · P7.2 · edatool agent integration

- The audit stage of the LLM variants passes `llms/agentmodel.AgentModel` to `edatool.audit`: every
  agent call is checked (run budget left, global caps, the audit stage deadline), logged to `llm.jsonl`
  (stage `audit`, prompt `edatool-agent`) and written to the ledger; the agent's usage is added to the
  run's LLM usage. edatool also gets `cache_dir = cache/edatool`, `session_seconds`, `max_sessions`,
  the audit `deadline` and G01 `n_jobs` from the settings. Without an API key the model still builds
  (placeholder key); its calls then fail and the deterministic audit result is kept.
- Real check (`full`, gpt-4.1-nano, ibmchurn level 3 seed 0 train file, demo mode, 180 s): 14 agent
  calls, 81,185 tokens, $0.0085; the run finished in 173 s with test ROC-AUC 0.8114 (dummy 0.5).
- Open problem found (for P7.9): nano's name review marked `tenure`, `TotalCharges` and the packed
  `SeniorCitizen_MonthlyCharges` as known only after the event; with `leakage.drop_semantic_only=true`
  the leak guard dropped them. The semantic prompt, the semantic-only policy and the pilot model
  (gpt-4.1-mini) are to be checked in the dev iteration.

## 2026-10-11 · P7.3 · Cleaner

- The cleaner answers with one list of operations (`op`, `params_json`, `issues`, `reason`); code
  parses `params_json`, sorts each operation into column, row or target steps by its catalog kind,
  validates parameters and columns in order after the protected steps, and refuses target mappings that
  merge classes; all errors go back to the LLM (2 retries), then the translator plan is used. Code puts
  the protected steps (leak guard drops and protected translator steps) first and drops LLM repeats of
  them; every LLM step is gated.
- Prompt inputs: the task, the protected steps as JSON lines, `issues.md` of the audit (at most 9,000
  characters), a columns view from the training rows (type, uniques, empty share, value counts of up
  to 15 values, pattern stats; names, types and counts only in `full-nosamples`), the gated default
  plan and the catalog text from `operations/registry.py`.
- Fix: the spend ledger creates its folder before taking the lock.
- Real check (gpt-4.1-nano, 5,000 rows of the ibmchurn level-3 train file, demo mode, 180 s): one
  cleaner call ($0.0011) gave a valid plan on the first try with parse_boolean for 3 yes/no columns and
  map_categories for spelling variants; test ROC-AUC 0.8255. The leak guard again dropped `TotalCharges`
  by nano's name review (open problem for P7.9).

## 2026-10-11 · P7.4 · Repair loop

- The compile stage checks the plan, sends each failing step to the LLM (`prompts/repair.md`: the step,
  the structured error, the failed examples, the value counts of its column and the catalog entries of
  the step and of related operations), puts the corrected or replacing step in place (source `repair`)
  and checks the plan again; a step gets at most `cleaner.max_repairs` (3) attempts, then it is
  dropped. Protected steps are never sent to repair (they are dropped with a decision if they fail).
  The `rules` variant and `full-norepair` drop failing steps at once. Every attempt is a decision.
- An identical repair prompt (same step, same error) comes from the LLM cache, so a repeated attempt
  costs nothing.

## 2026-10-11 · P7.5 · Refine loop

- After the gate, the LLM variants may revise the plan when a step was removed or the plan lost to P0
  (`prompts/refine.md`: direction of the metric, the P0 score, the full plan's fold scores, the effect
  of every step, the rejected steps, the current gated steps and the catalog). The answer is a full
  plan in the cleaner format with the same code checks; revised steps get source `refine`; protected
  steps are added by code. Each revision is checked (failing steps repaired or dropped) and gated; at
  most `cleaner.max_refines` (2) rounds, and the gate stage deadline also stops the loop. The plan with
  the best gated score of all rounds is the final plan; `gate.json` keeps every round.

## 2026-10-11 · P7.6 · Sandbox and custom operation

- `tools/sandbox.py`: the AST check allows only the listed imports (pandas, numpy, re, math, datetime,
  string, collections, itertools, functools, statistics, unicodedata), refuses the listed modules and
  names (plus `getattr`, `setattr`, `delattr`, `vars`) and every dunder attribute (`__class__`, ...),
  and needs exactly one class with `fit` and `transform`. The code runs with the main Python in a new
  temporary folder with only SYSTEMROOT, PATH, TEMP, TMP, PYTHONIOENCODING and PYTHONHASHSEED (no API
  key); data goes in and out as parquet; rows and index are checked; a time-out (60 s) kills the
  process tree with `taskkill /T /F`. This is a safety net, not a security boundary (README, P11.2).
- The `custom` operation keeps its training values and refits the generated code on them in the
  sandbox at every transform, so prediction gives the same result without keeping a live process. Its
  code is generated in the compile stage (`prompts/customop.md`), checked by the AST check and a trial
  run (fit on 200 training rows, transform 100 other rows) whose errors go back to the LLM; a custom
  step without valid code is dropped with a decision. It is registered in the catalog but not in the
  17 column operations of the contract tests (it needs a sandbox per transform).

## 2026-10-11 · P7.7 · Free-code variant

- `freecode` keeps the audit, review, leak guard and gate of `full`, but the LLM writes one class
  `CleaningStep` (`prompts/freecode.md`) that cleans the whole feature table instead of choosing
  catalog operations. The prompt gets the same task, audit issues and column view (with or without
  values) as the cleaner. The code is checked by the AST check and a trial run in the sandbox (fit on
  300 training rows, transform 100 others); errors go back to the LLM (the client's schema retries).
  This trial loop is the variant's repair: the class is one step, so the catalog repair and refine
  prompts do not apply (`cleaner` is off in this variant; a step that fails the data checks is
  dropped, and the gate then compares it with the protected-only plan). Without valid code the plan is
  the protected steps only, with a decision.
- The plan is the protected steps, then one gated `freecode` step. The `freecode` operation is in the
  registry (so plans, the gate and `FittedModel` work) but hidden from the catalog text, and the
  cleaner refuses it.
- Sandbox results are cached in the process by (code, training rows, input rows): the CV folds and
  the engine candidates repeat the same transforms. On the 400-row test run this cut sandbox calls
  from 53 to 18 (about 0.8 s each under emulation). The same cache serves `custom`.
- Nano check (5,000-row samples, budget 150 s): churn level 3: valid code on the first try (numbers,
  yes/no, packed column), the gate removed it (effect -0.004 roc_auc), test roc_auc 0.825.
  kinghousedata: three answers failed the trial run ("transform changed the number of rows or the
  index"; the code returned the 300 fit rows), so the run used the protected steps only and logged
  the failure. The sandbox error now names the expected and returned rows and index; on the re-run nano
  fixed its code on the second answer, and the gate removed the step (no gain on the clean file).
  Both runs finished with a model and a report; cost about $0.01 for the three runs.

## 2026-10-11 · P7.8 · Summary paragraph

- The facts of `prompts/summary.md` are the template report itself (cut at 6,000 characters): it
  already holds only checked facts (task, split, audit, leaks, plan, gate, scores, cost). The report is
  written first, then the summary is asked, then the report is written again with a `## Summary`
  section, so the summary call is in the cost table.
- Code checks the answer: at most 120 words, and every number must appear in the facts (thousands
  commas ignored; counts up to 10 are allowed). A wrong answer goes back to the LLM (schema retries);
  without a valid answer, or when the provider fails, the template report stays. Only the LLM variants
  ask for a summary, and only when `report.llm_summary` is true.
- Real nano check on the churn freecode report: the first answer failed the number check, the second
  was valid (5 sentences); cost $0.0004.

## 2026-10-12 · P7.9 · Dev iteration (part 1): dev samples, semantic review, backends

- **Dev samples.** The dev cells (P6.4) are full size, and the level-3 files have layout mess (title
  rows, `;`, cp1252), so they cannot be sampled after writing. `benchmark.materialize --sample-rows N
  --group dev5k` takes a fixed sample of the clean table first (6,250 rows, so about 5,000 training
  rows) and then corrupts and splits it as usual; the sampled table gets its own saved split
  (`<name>-n6250`). Grid: `benchmark/grids/p7-dev.json` (6 dev datasets x levels 0 and 3 x seed 0,
  120 s).
- **Semantic name review (open problem from P7.2).** Nano marked normal features (TotalCharges, a
  packed SeniorCitizen_MonthlyCharges) as post-event leaks, and the semantic-only policy dropped them.
  The edatool prompt `agent/prompts/semantic.md` now says that 'suspected' needs a clear
  outcome-or-later meaning, lists value kinds that exist before the prediction (totals so far,
  tenure, counts, balances, prices, packed or renamed columns) as 'clean', and says "when unsure,
  choose 'clean'". The policy (`drop_semantic_only=true`) stays. Re-run on the level-3 churn sample
  with nano: only the three id columns were dropped (before: five).
- **Backends (`modelling/backends.py`, `modelling/backendworker.py`).** No task of §6 names them, but
  `full+ag` and `rules+ag` are in `pilot-main`, so they are built here. The preparation (agent steps
  + typing step, no generic stage) runs in the main process; the prepared rows go as parquet to a
  worker in the AutoGluon or FLAML environment, which fits with the remaining model seconds minus
  15 s (AutoGluon: medium_quality, sequential folds, 4 CPUs, as the baseline adapter) and later
  predicts. The model folder is short (`cache/b/<token>`) because AutoGluon paths inside the long
  run folders could pass the Windows path limit; `model.joblib` keeps the folder path. The "CV" score
  of a backend run is the backend's own validation score. Any backend error is logged as a decision
  and the own engine trains instead. Checked: slow tests (real FLAML and AutoGluon fit and predict)
  and a `rules+ag` run on the level-3 churn sample (test roc_auc 0.8115).

## 2026-10-12 · P7.9 · Dev iteration (part 2): findings of the free variants

Iteration 1 (`generic` and `rules` on the 12 dev5k cells, 120 s): all 24 runs ok. `rules` was worse
than `generic` on houses l3 (rmse 0.412 vs 0.294), kinghousedata l3 and Is-this-a-good-customer l3;
both were at the dummy level on creditcardfraud l3. Causes and fixes:

- **Packed decimals taken for ids.** edatool typed unique packed values (`L:34.03 W:7.03`,
  `0.65/0.23`) as `id`, so E03 and the framer marked them, and the leak guard dropped them (houses
  lost MedianIncome). `edatool/core/typeinference.py`: a unique text column where at least half of
  the values contain a decimal number is `categorical_nominal` (low confidence), not `id`.
- **Mixed currency formats.** `$52` and `39 USD` in one column did not parse, so edatool saw label
  variants. `USD`, `EUR` and `GBP` are now numeric cleaning tokens (`edatool/config/tokens.json`);
  the column becomes B01 and the translator's `parse_number` removes them.
- **Magnitude suffixes.** B07 with the units k and M was skipped ("two or more units"). The
  translator now gives k = 1e3, M = 1e6, B = 1e9 when the units are only magnitude suffixes and k is
  one of them (then M means million, not metres).
- **group_rare on numbers written as text.** D05/E04 on a column that another step parses to
  numbers is left out (with a skipped-issue reason).
- **Engine time on wide text.** With packed columns, skrub's StringEncoder made one fold
  preparation take about 10 s, so the 5-fold preparations of `linear` and the tree kind used the
  whole 120 s and only the dummy was trained. The candidates are now dummy, lightgbm, linear,
  catboost (trees first), and a family whose preparation kind is new is skipped when the time left
  is less than 1.2 x the last preparation time + 2 s. creditcardfraud l3: LightGBM CV roc_auc 0.960
  instead of the dummy.
- The edabench deterministic results are unchanged after these edatool changes (recall 0.987,
  precision 0.855, the post-P4 numbers).
- **Verify time.** On creditcardfraud l3 the translator plan had 34 steps; edatool verify built one
  snapshot per step (about 4 s each), so the compile stage took 121 s (limit 45 s) and the cell timed
  out. Verify now checks at most `verify.max_snapshots` (6) evenly spaced snapshots (always the last
  one); the issues of a skipped step count on the next checked snapshot. It is skipped when the
  compile stage time is already used up. Re-run of the cell: 117 s, ok.

Iteration 2 (after the fixes above), dev5k, 120 s, seed 0 (`rules` vs `generic`; lower is better
for rmse and log_loss):

| dataset | level | metric | generic | rules |
|---|---|---|---|---|
| Is-this-a-good-customer | 0 | roc_auc | 0.6703 | 0.6706 |
| Is-this-a-good-customer | 3 | roc_auc | 0.6599 | 0.6518 |
| creditcardfraud | 0 | roc_auc | 0.9167 | 0.9227 |
| creditcardfraud | 3 | roc_auc | 0.7492 | 0.6979 |
| houses | 0 | rmse | 0.2318 | 0.2300 |
| houses | 3 | rmse | 0.2926 | 0.2374 |
| ibmchurn | 0 | roc_auc | 0.8391 | 0.8405 |
| ibmchurn | 3 | roc_auc | 0.8212 | 0.8350 |
| kinghousedata | 0 | rmse | 114,912 | 109,776 |
| kinghousedata | 3 | rmse | 152,848 | 114,343 |
| winequality | 0 | log_loss | 0.8277 | 0.8283 |
| winequality | 3 | log_loss | 0.9924 | 0.8983 |

`rules` >= `generic` on 9 of 12 cells and clearly better on the messy level-3 files of houses,
kinghousedata, ibmchurn and winequality. The dev5k test set of creditcardfraud has only 2 fraud
rows, so its roc_auc is noise (it moved by 0.25 between identical generic runs of iteration 1 and 2
after the engine change); it is not used for conclusions. Is-this-a-good-customer l3 (-0.008) and
winequality l0 (+0.0006 log_loss) are within the run-to-run noise of this small sample.

## 2026-10-12 · P7.9 · Dev iteration (part 3): first `full` runs with nano

`full` on the 12 dev5k cells (nano, 120 s): 12/12 ok, mean cost $0.0092 per run (total $0.11),
7 to 26 LLM calls per run (most of them edatool agent calls).

| dataset | level | metric | generic | rules | full |
|---|---|---|---|---|---|
| Is-this-a-good-customer | 0 | roc_auc | 0.6703 | 0.6706 | 0.6736 |
| Is-this-a-good-customer | 3 | roc_auc | 0.6599 | 0.6518 | 0.6518 |
| creditcardfraud | 0 | roc_auc | 0.9167 | 0.9227 | 0.9455 |
| creditcardfraud | 3 | roc_auc | 0.7492 | 0.6979 | 0.6883 |
| houses | 0 | rmse | 0.2318 | 0.2300 | 0.2285 |
| houses | 3 | rmse | 0.2926 | 0.2374 | 0.2332 |
| ibmchurn | 0 | roc_auc | 0.8391 | 0.8405 | 0.8384 |
| ibmchurn | 3 | roc_auc | 0.8212 | 0.8350 | 0.8350 |
| kinghousedata | 0 | rmse | 114,912 | 109,776 | 114,453 |
| kinghousedata | 3 | rmse | 152,848 | 114,343 | 128,840 |
| winequality | 0 | log_loss | 0.8277 | 0.8283 | 0.8478 |
| winequality | 3 | log_loss | 0.9924 | 0.8983 | 0.9001 |

Findings from the decision logs of the worst cells:

- **kinghousedata l3.** The LLM plan was close to the translator plan but added `drop_columns
  waterfront` ("almost all rows have the same value"). In `rules` the gate tested the same drop and
  removed it (CV effect -3,856). In `full` the audit agent and the cleaner used about 20 s of the 120 s,
  so the gate had time only for P0 and P1, and every step stayed untested. Fix (`checks/gate.py`): a
  gated `drop_columns` that could not be tested is left out (dropping a feature is the riskiest
  change; the model can ignore a useless column). After the deadline the reduced plan is not
  re-scored (`reduced_not_scored` in gate.json). Also, gate decisions are now logged per step (a
  `removed` flag on each effect) instead of per operation name, so two steps with the same operation
  no longer share one verdict.
- **winequality l0.** All three cleaner answers failed the checks (nano proposed
  `map_target_labels` from "3.0" to "3", values that do not occur), so the translator plan was used;
  the plan was the same as in `rules`, and the score gap comes from the model time (67 s vs 86 s).
  The check now says "'3.0' does not occur in the target; map only values that occur (or leave out
  map_target_labels)" instead of "classes are never merged".
- **Model time.** At 120 s the LLM stages take 10 to 25 s, which the model stage of `full` does not
  get. This is part of the comparison (same wall-clock budget for every system) and is reported, not
  corrected.

## 2026-10-12 · P7.9 · Dev iteration (part 4): second nano round, mini check, stop

- `full` iteration 2 (nano, after the part-3 fixes; most answers from the cache, $0.007 for 12 runs):
  12/12 ok. kinghousedata l0 109,733 (was 114,453) and l3 114,523 (was 128,840); the other cells
  within noise of iteration 1. Without creditcardfraud (2 test positives), `full` >= `rules` on all
  cells but kinghousedata l3, where they are equal within 0.2%.
- Is-this-a-good-customer l3 (`generic` 0.660 vs `rules`/`full` 0.652): understood as noise. The
  table has 1,378 training rows and about 39 test positives; the final plan of `rules` and `full` is P0
  (only the two id columns dropped), and the dropped saved index has roc_auc 0.515 with the target,
  so `generic` does not gain from a leak. On this small table the gate kept P0 even though every
  single step had a positive leave-one-out effect (folds of about 460 rows make P1 vs P0 noisy).
- **Mini check** (`gpt-4.1-mini`, grid `p7-mini.json`, results `benchmark/results/p7-mini.csv`):

| dataset | level | full (mini) | full (nano) | rules | generic | LLM calls | cost (USD) |
|---|---|---|---|---|---|---|---|
| ibmchurn (roc_auc) | 0 | 0.8420 | 0.8422 | 0.8405 | 0.8391 | 8 | 0.0149 |
| ibmchurn (roc_auc) | 3 | 0.8376 | 0.8375 | 0.8350 | 0.8212 | 12 | 0.0288 |
| kinghousedata (rmse) | 0 | 109,776 | 109,733 | 109,776 | 114,912 | 13 | 0.0246 |
| kinghousedata (rmse) | 3 | 114,749 | 114,523 | 114,343 | 152,848 | 21 | 0.0454 |
| houses (rmse) | 0 | 0.2285 | 0.2290 | 0.2300 | 0.2318 | 11 | 0.0251 |
| houses (rmse) | 3 | 0.2344 | 0.2343 | 0.2374 | 0.2926 | 18 | 0.0374 |

- **Measured cost per `full` run:** nano $0.0092 (iteration 1, no cache); mini $0.029 mean, $0.045
  max (target <= $0.05). Most tokens are edatool agent calls; level-3 files cost more (more issues to
  review).
- **Stop.** Success 100% (36 `full` runs, 48 free runs, all ok after the verify fix); mini cost per
  run <= $0.05; on the messy level-3 files `full` >= `rules` >= `generic` (kinghousedata l3: `full` and
  `rules` equal within 0.4%), and the exceptions are written down above. Paid `full` runs: 31 of 40
  (1 semantic check, 12 + 12 nano, 6 mini). Development spend: $0.36 of $1.50.

## 2026-10-12 · P8.1 and P8.2 · Torture suite and fault injection

- `tests/torture.py` writes the 27 torture files of §5 at test time; `tests/testtorture.py` runs the
  `rules` variant on each (20 s budget). The hard stops are fast tests; the files that should give a
  model are `slow` (they run the whole pipeline, about 5 minutes together). The 1,000,000-row file is a
  separate slow test and is measured in P8.3.
- In demo mode exact duplicate rows are removed before the split (§3.1). A file with one column (only
  the target), with all feature values missing, or with only constant features therefore keeps 2
  rows, one per class, and stops. This is the correct outcome, but the message now says why: "Only 2
  labeled rows after removing 298 exact duplicate rows (the features do not tell the rows apart); at
  least 20 are needed." In benchmark mode no rows are removed, so the same files train a model.
- Fault injection (`tests/testfaults.py`): one `rules` run where an operation, Optuna, the CatBoost
  candidate and edatool verify all raise (fast test: the step is dropped, LightGBM uses its default
  parameters, CatBoost is skipped, the verify error is in checks.json, the model predicts), and one
  `full` run where every LLM call raises (slow test, 19 s: translator plan, template report). The
  provider-failure paths are also covered by fast unit tests (cleaner, summary, agent model).

## 2026-10-12 · P8.3 and P8.4 · Performance and reproducibility

- `benchmark/performance.py` runs a file through the CLI in a subprocess and samples the resident
  memory of the whole process tree with psutil every 0.5 s; results in
  `benchmark/results/performance.json`.
- First measurement (`rules`, 300 s): creditcardfraud took 375 s. Causes and fixes:
  - edatool verify ran on all 227k training rows (99 s for a 45 s stage). Verify now uses at most
    `verify.max_rows` (20,000) training rows.
  - The engine searched on a 50,000-row sample but refitted on all rows without reserving the time
    (71 s). `refit_plan` now estimates the refit from the fold preparation and the best trial, scaled
    to all rows; when it does not fit in the time left, only the best family is refitted, and if
    that still does not fit, it is refitted on the search sample (both logged).
  - The model stage took all remaining time, so the test stage ran past the budget.
    `Budget.model_seconds` keeps 4% of the total budget for the test and report stages.
- Final measurement (300 s, wall time including interpreter start): creditcardfraud (284,807 rows)
  286 s, peak 2.14 GB, test roc_auc 0.971; the 1,000,000-row torture file (141,957 training rows
  after the demo-mode duplicate removal) 267 s, peak 1.35 GB. `full` was not measured: there are no
  cached LLM answers for these files and the plan allows it only from the cache.
- Reproducibility (`tests/testreproducible.py`): two `full` runs with the same seed and a warm LLM
  cache give identical plans, CV scores and test scores; the second run asks the provider only for the
  summary (its facts include the run id and stage times). The test turns Optuna off, because a
  time-bounded search runs a different number of trials under a different machine load; with Optuna on,
  results are reproducible only up to that.
- Fresh clone (P8.5): `git clone` into a temporary folder, `uv sync` (exit 0), `uv run pytest -q`:
  182 passed. The first run took 192 s (every module compiled and imported cold in the new folder);
  the second run took 140 s, inside the 3-minute target.

## 2026-10-12 · P9.1 · Materialize

- Written (all in `cache/benchmark/files/`, about 475 MB, not in git): Group A at levels 0 to 3 and
  seeds 0 to 2 (180 cells); the single-type files for `pilot-types` and the full `types` grid: the 10
  mixed mess types and the layout type at level 3, seed 0, for all 15 Group A datasets (165 cells);
  Group C at seeds 0 to 2 (15 cells); Group D at seeds 0 to 2 (30 cells). The leak type exists only
  in Group D. `benchmark/results/files.json` lists the 390 cells with rows, mess and sha256 hashes
  (`python -m benchmark.materialize --manifest`).
- Bug fixed: `corrupt_packed` inserted the packed column at the old position of the first source
  column after both were dropped, which is out of range when the second column comes first (SDSS17).
  It now uses the smaller of the two positions. Every file written before had the second column after
  the first (otherwise it would have failed), so no existing file changes.

## 2026-10-12 · P9.3 · Smoke grid and freeze

- Smoke grid (2 dev5k datasets x levels 0, 3 x seed 0 x 10 systems, 60 s, `full` with nano): first
  38/40 ok. FLAML failed on both level-3 files because LightGBM inside FLAML refuses JSON characters in
  column names (`Unnamed: 0` from the saved-index mess type). The FLAML adapter now replaces
  `[ ] { } " , :` in column names, the same rule as the LightGBM baseline (a user of FLAML would do
  this first; without it FLAML would fail on every messy file for a naming detail). After the fix:
  40/40 ok. `full` cost $0.003 for 4 runs (nano).
- FLAML on houses l3 scored rmse 1.249 (dummy 0.565): its predictions are almost constant and 1.1
  below the true mean. Checked: the training file reads correctly and the saved index does not predict
  the target (Spearman -0.014). The near-unique text columns (customer_id, the packed column) become
  categories that never occur in the test file. This is FLAML's own behaviour on messy text columns,
  not an adapter bug, so it stays.
- `python -m benchmark.runner --grid <name> --check`: the 6 full grids (`main`, `leaks`,
  `ablations`, `types`, `subsets`, `budgets`) load, every cell has its files, and the first cell
  starts (`v1` in `subsets` is not started: it calls an LLM outside the v2 spend control). Every pilot
  grid cell has its files.
- Tag `v2-freeze` after this commit. From now on, method changes are not allowed; bug fixes go to
  `docs/deviations.md` with a re-run of the affected cells.

## 2026-10-12 · P9.4 · E7 framer accuracy

- `benchmark/framereval.py`: 40 OpenML datasets, the 31 of the benchmark pools plus 9 OpenML-CC18
  classification datasets (diabetes, vehicle, ilpd, qsar-biodeg, wdbc, balance-scale, breast-w,
  dresses-sales, kc1); at most 100,000 rows. Truth: the OpenML default target; the task from the pool
  spec (CC18: binary or multi-class by the number of classes). Each table is framed without flags,
  by rules only and by rules + `gpt-4.1-mini` (the framer's LLM step chooses among the rules' top 5
  candidates). The pool lists 31 OpenML datasets, not 32, so a ninth CC18 dataset (kc1) was added to
  reach 40.
- Result (`benchmark/results/framer.csv`): rules 32/40 targets right (31/40 target and task); rules
  + LLM 36/40 (35/40). LLM misses: E-CommereShippingData (chose Customer_rating), miami_housing
  (structure_quality), bank-marketing (default), credit-approval (A1, anonymous names); in these the
  true target was not among the 5 rule candidates, so the LLM could not choose it. wine_quality: target
  right, but the integer quality score was framed as multi-class while the pool treats it as
  regression. Cost $0.025.

## 2026-10-12 · P10.1 (written during the pilot) · Analysis, figures, projection, repair checks

- `benchmark/analysis.py` follows §4.7: a failed run gets the dummy score of its cell (not removed);
  normalized score per cell, mean over seeds then datasets; average ranks; Friedman + Nemenyi
  critical difference (Demšar's q values at alpha 0.05); Wilcoxon with Holm over the baselines,
  win/tie/loss counts; bootstrap intervals over datasets; reliability and cost; leak detection from
  the run folders (`leaks.json`; structural id drops are not counted as leak drops).
- `benchmark/figures.py` regenerates every figure (PDF and PNG) and table (CSV and Markdown) with
  one command. Colours: the 8 categorical slots of the dataviz skill's validated reference palette in
  a fixed system order, plus a distinct marker per system (secondary encoding); the palette validator
  needs Node, which is not installed, so the pre-validated palette is used unchanged. Dummy is the 0
  line of the normalized score and is not drawn.
- `benchmark/projection.py` follows §4.10. Seed noise is measured on oriented scores scaled by the
  seed-0 cell's (best - dummy), because the seed-1 cells contain only the free systems and would have
  another best. `--predict` writes the held-out predictions before the held-out cells run.
- `benchmark/repairs.py` (for P9.8): the cleaning-only check runs the `rules` variant up to the gate
  (new run option `stop_after`; evaluation tooling, not a method change), fits the final plan's steps
  on the training rows and scores the corrupted cells against `cells.parquet` (numbers within 1e-6
  relative, labels equal, dates by their parts, fake-missing and impossible values repaired when they
  become missing; a corrupted column is matched to the best prepared column, so renamed and split
  columns count).

## 2026-10-12 · P9.5 · pilot-main

- Run in four chunks of at most 2 hours (resumable runner): (1) the first 100 cells (dummy,
  lightgbm, skrub, flaml, part of autogluon); (2) autogluon, generic, rules; (3) the first 10 `full`
  cells; (4) the other `full` and the `full+ag` cells. The 4 held-out cells are excluded by the grid's
  `holdout` list and run last (P9.13). The previous agent session ended during chunk 2; the runner
  process kept running and finished normally (no cell lost).
- Spend check after the first 10 `full` runs (§4.6): mean $0.023 per run (max $0.038, 4 to 21 LLM
  calls), below $0.06, so `pilot-ablations` and `pilot-types` keep their planned size. Experiments
  spend at that point: $0.26 of $2.75.
- Result: 212 cells (the 4 held-out cells excluded), 211 ok. `full` mean cost $0.026 per run;
  `full+ag` $0.006 (its LLM answers mostly come from the cache of `full`). Experiments spend after
  `pilot-main`: $0.83.
- One failure: `full` on students_dropout_and_academic_success level 3 was killed at the 210 s
  cell time-out (budget 120 s). Cause: the audit stage took 57 s (72 issues to review) and used up
  the time before the model stage; the plan stage then still made its LLM call (55 s, 34 steps) and
  the gate still ran (46 s) although their stage limits were already 0. This is a reliability bug of
  the frozen system (stages do not skip their work when their limit is 0), not a harness bug, so by
  the pre-registration it stays a failure with the dummy score; it is reported under RQ4. Recommended
  fix before a full benchmark (not applied, it would change the frozen method): when a stage's
  limit is 0, the LLM stages use their fallback and the gate keeps P0 without testing.

## 2026-10-12 · P9.6 · pilot-leaks

- 25/25 ok, $0.13. Leak detection on the 5 Group D datasets: `full` dropped all 5 leak columns with no
  other non-id column dropped (recall 1.0, precision 1.0); `rules` (no semantic review) and
  `full-noleakguard` dropped none. Naive vs deployment: AutoGluon, LightGBM, `rules` and
  `full-noleakguard` score almost perfectly when the leak values are present in the test file and fall
  sharply when they are masked; `full` has the same score in both (it did not use the leak).

## 2026-10-12 · P9.7 · pilot-ablations

- 20/20 ok, $0.14. Mean normalized score at level 3 on the 4 ablation datasets (with the
  `pilot-main` systems in the same cells): `rules` 0.995, `full-noreview` 0.991, `full-nosamples`
  0.988, `full+ag` 0.988, `full-nogate` 0.987, `full-norepair` 0.987, `freecode` 0.986, `generic`
  0.894, skrub 0.799, autogluon 0.784, `full` 0.746 (its time-out on students_dropout counts as the
  dummy; on the other 3 datasets `full` is 0.991 to 1.000), lightgbm 0.673, flaml 0.631. On this
  small sample the ablated parts change little; every v2 variant with a cleaning stage is far above
  the systems without one.

## 2026-10-14 · P9.8 · pilot-types, repair accuracy, E6b reading

- `pilot-types` (4 datasets x 4 key types x full/generic/autogluon, level 3): 47/48 ok, $0.34. The
  failure is again `full` on students_dropout (spellings), the same stage-limit time-out as in
  `pilot-main`. Mean normalized score per type (full / autogluon / generic): packed 1.000 / 0.868 /
  0.952; units 1.000 / 0.910 / 0.934; mixed units 0.971 / 0.976 / 0.957; spellings 0.747 (time-out
  counted as dummy) / 0.995 / 0.984.
- Repair accuracy of the cleaning stage (`benchmark/results/repairs.csv`; share of corrupted training
  cells restored): cleaning-only `rules` on all 15 Group A datasets (95 + 15 files): booleans 0.988,
  fake missing 0.938, packed 0.889, dates 0.825, mixed units 0.788, spellings 0.711, units 0.539,
  impossible values 0.004 (the translator has no rule that finds sign flips or x100 scale errors);
  `full` (from the prepared tables of `pilot-types`, 4 datasets): packed 0.996, mixed units 0.826,
  spellings 0.818, units 0.690. Plain missing values and ids are not repair types (nothing to restore).
- E6b reading robustness (`reading` grid: 5 free systems x the 15 layout files, level 3, 120 s;
  `benchmark/results/reading.csv`): `generic` and `rules` read 12/15 files and train; LightGBM,
  skrub and AutoGluon 0/15 (their CSV readers fail on the title rows, `;` and cp1252, or take the
  title line as the header). The 3 failures of `generic` and `rules` are the regression files: the
  target is written with decimal commas ("39,6") and is not converted, so every target value looks
  non-numeric and the run stops ("Only 0 labeled rows"). A real weakness of the frozen system
  (recommended fix before a full benchmark: convert decimal commas in a regression target).
- Harness and evaluation fixes during this task are listed in `docs/deviations.md`.

## 2026-10-14 · P9.9 to P9.12 · pilot-seed, pilot-natural, pilot-budget, E8

- Paused on 2026-10-14 at the user's request during `pilot-seed` (84/96 done, all processes ended);
  resumed later with the runner, no cell lost.
- `pilot-seed` (seed 1, free systems, levels 0 and 3): 96/96 ok, $0. Used by the projection for the
  seed noise.
- `pilot-natural` (Group C: HR_Analytics, in_vehicle_coupon, heloc; seed 0): 12/12 ok, $0.12. `full`,
  `generic` and AutoGluon are within 0.006 roc_auc of each other on all three files (LightGBM a little
  lower): no gain from the cleaning stage on these naturally messy files.
- `pilot-budget` (Bank_Customer_Churn and diamonds, level 3, 300 s): 4/4 ok. The order of `full` and
  AutoGluon is the same at 120 s and 300 s (`full` ahead on Bank_Customer_Churn, AutoGluon ahead on
  diamonds).
- E8 (`python -m edabench.evaluate benchmark`, deterministic): recall 0.987 (strict 0.978), precision
  0.855 on 372 injected issues, the same as after P4. Copied to `benchmark/results/edabench-final.*`;
  the baseline comparison (ydata-profiling, cleanlab, old eda.py) of P4.1 is in
  `benchmark/results/edabench-baselines.*`. The P4.1 files in `edabench/results/` stay unchanged.
- Experiments spend after P9.12: $1.61 of $2.75.

## 2026-10-14 · P9.13 · Held-out cells

- The 4 held-out cells ran last (`--holdout`), after the predictions were committed (7230685): 4/4
  ok. 3 of 4 actual scores fell inside their predicted 95% intervals (E-CommereShippingData full
  0.7359 and autogluon 0.7164; concrete full rmse 3.714 inside; concrete autogluon rmse 11.979 outside
  [9.175, 10.272], worse than predicted). By the pre-registered rule (at least 3 of 4) the projection
  is not called unreliable.

## 2026-10-14 · P11.1 · Code quality pass

- Searched every project module for top-level functions and classes that nothing references. The
  corruptor (dispatch by `globals()`) and the edatool detectors (registered by name) are used
  dynamically. Three were really unused: `modelling/cv.py: cross_val_oof` (superseded by the engine's
  own fold loop) is removed; `benchmark/leaks.py: write_leak_table` gets a command line
  (`python -m benchmark.leaks` regenerates `benchmark/results/leaks.md`, byte-identical);
  `benchmark/spend.py: optional_stopped` (the "$4: stop optional LLM work" rule of §1.4) is now
  checked by the runner, which skips LLM cells once the total spend reaches it (harness only; the
  total is about $2, so no past result is affected).
- `ruff check --select F .` is clean; the style test passes.

## 2026-10-16 · P13 · Workspace cleanup

- Asked by the user: remove what is not needed, organize the folders, and join results that were
  spread over many files. The user checked the removal list first.
- Removed: the frozen v1 copy `benchmark/legacy/v1/` and its environment (never run; restorable from
  the tag `v1-dslab1` and the git history), with the `subsets` grid, the `v1` registry entry, the v1
  adapter test and the v1-only code in the runner; the dev grids `p2-*`, `p3-generic`, `p5-rules`,
  `p7-*` (their result rows are kept); `benchmark/envs/tabpfn.txt` (TabPFN was skipped);
  `edabench/runlegacy.py` (unused); duplicate copies of the audit results.
- Joined: the 17 per-grid result files into `benchmark/results/runs.csv` (rows unchanged; the column
  `grid` names the experiment; the runner resumes per grid from it); the 11 tables into
  `benchmark/results/tables.md` (same Markdown); the leak table, corruptor examples and natural-mess
  check are written straight into `docs/datasets.md` (byte-identical). `docs/results.md` and
  `docs/projection.md` regenerate with the same numbers.
- The audit results live only in `edabench/results/`; the final P9.12 run is `benchmark-final.*`.
  `docs/results.md` (E8) now shows it (recall 0.987, precision 0.855) instead of the P4.1 numbers it
  showed by mistake (0.986, 0.896).
- Moved to `docs/`: `buildplan.md`, `progress.md`, `study.html`, and the original design inputs
  (`architecture/`, before git-ignored) as `docs/background/` (now in git).
- Git-ignored data deleted: AutoGluon backend models (`cache/b`, 4.9 GB), the saved `model.joblib`
  of every benchmark run folder (554 MB; reports, decisions, LLM logs and predictions are kept),
  work folders, dev run folders, the generated 1M-row file, the edabench baseline cache and Python
  caches. Kept: the LLM caches, OpenML downloads, benchmark files and run logs.

## 2026-10-19 · P13.3 · GitHub and clean commit history

- Asked by the user: push the project to `https://github.com/naginagiyev/data-science-lab2.git` on a
  branch `V2` (the default branch for now), push every later commit, and show only the user as
  author. The 87 commits so far ended with a Claude co-author line, so the commit messages were
  rewritten once (`git filter-branch --msg-filter`) to drop that line. Authors, dates and file
  contents of every commit are unchanged; only the commit ids changed.
- Every old commit id in `docs/progress.md`, `docs/decisions.md`, `docs/edatool-baseline.md` and the
  `git_hash` column of `benchmark/results/runs.csv` was replaced by its new id (711 places), so the
  references still point to the right commits. The pre-registration order is kept: the held-out
  predictions commit still comes before the held-out runs, with its original date.
- The tags `prereg`, `v1-dslab1` and `v2-freeze` now point to the rewritten commits; they stay local
  (not pushed, by the user's choice). A bundle of the old history is kept outside the project
  (`../test-history-backup-2026-10-19.bundle`).
- The local branch `main` is renamed to `V2`. `CLAUDE.md` now says to push after every commit.

## 2026-10-19 · P14.0 · Restructuring: baseline

- The user asked to make every part of the pipeline the same depth, to merge the vendored edatool
  into the main code, and to scope the project as an MSc thesis. Plan: phases P14.0 to P14.10
  (`docs/buildplan.md` §6). Local git only (no push), no co-author lines.
- Python lines before P14: edatool 4,582 (plus 1,236 lines of JSON config), edabench 2,331,
  benchmark 2,872, operations 1,491, tools 746, modelling 637, orchestrator 605, agents 567,
  checks 346, llms 332, main/settings/schemas/paths 425, tests 2,958. Fast suite: 191 passed.
- Found while making the baseline: `tests/testagentmodel.py::test_provider_failure_falls_back` breaks
  only the smolagents model; the summary of its `full` run used the real OpenAI provider and the real
  ledger (2 nano calls, $0.00035, logged in `spend.json`). The file goes in P14.3; until then it is
  left out of the test runs.

## 2026-10-19 · P14.1 · Shared foundations

- `tools/columntypes.py` holds the type inference and the value patterns (from edatool
  `typeinference.py` and `patterns.py`). The thresholds that only the type inference uses are module
  constants; the tokens are in `configuration/tokens.json`. One list of date patterns now carries
  its strptime format (before: four separate lists). `name_words` of the framer is the same function
  as `name_tokens` and was removed.
- Parity check on the 4 local datasets (20,000 rows each): the same types, confidences and typed
  values as edatool's `infer_schema`.
- `tools/reader.py` is the only reader (from edatool `loader.py`). The read report is returned
  directly instead of being stored in `DataFrame.attrs["edatool"]`; an empty file now stops with
  "The file ... is empty." instead of a wrapped `ValueError`.
- `tools/splitter.py` makes its own split advice: H01 (repeated ids, group split) and H02 (target
  trend over a date, time split). H03 to H05 never changed the advice and were dropped. As before,
  the last matching column wins.
- `tests/testloader.py` is merged into `tests/testreader.py`; `tests/testsplitter.py` tests the advice.

## 2026-10-19 · P14.2 · The audit package

- `audit/` replaces edatool's detectors: `context.py` (what every check gets), `structure.py`,
  `formats.py`, `values.py`, `target.py` and `core.py` (`run_audit`, `issues_text`). Catalog and
  thresholds are one file, `configuration/audit.json`. The audit is code only; it runs once on the
  training rows (types once, then the checks in order; a failing check is recorded in `failed`).
- Kept are the 29 codes that the translator or the leak guard act on. Dropped, because no code
  used them: A01, A03, A06, A07, B03, C01, D04, E07, E09, F01, F02, F04 to F07, H03 to H05, P01 to
  P05, the future-date rule of D03, the issue links (enrichment), fingerprints, row sets, coverage
  and the console printer. G02 (pure categories) is now G01 with rule `purity`; G04 (missingness
  predicts the target) is covered by the G01 tree, which uses missing values as a signal
  (scikit-learn 1.9). E07 (exact linear relation) is left to E08 (rank correlation).
- The issue contract is flat (`schemas.Issue`: code, columns, row count, evidence, priority, rule,
  action, params, diagnosis) and has no status: without the agent nothing confirms or dismisses an
  issue. Ids are `<code>-<n>` (for example `C02-1`). Two behaviour changes follow: a low-confidence
  id column is always dropped (before, only the agent could keep it), and B05 similarity pairs and
  B07 with two or more units go to the LLM cleaner (before, also to the agent).
- The cleaner prompt gets `issues_text(report)` (evidence and default fix per issue) instead of the
  `issues.md` file. The audit report is saved as `audit.json` in the run folder.
- Parity: on the 4 local datasets and on 9 benchmark files (7 level-3 files, 2 Group D leak files),
  the new audit finds the same (code, columns) pairs as edatool's deterministic audit.
- edatool `verify` was removed here and not in P14.4: it needs the old report type, and its result
  (`checks.json`, `verify`) never changed the plan, the gate or the model.
- The condition language of `drop_rows_where` moved to `operations/rules.py` (unchanged). The
  catalog text in prompts says "audit fixes" instead of "edatool fixes".
- `tests/testdetectors.py` became `tests/testaudit.py`; `testleakage`, `testtranslator`,
  `testcompiler`, `testleaks`, `testcorruptor`, `testfaults` follow the new contract.
- Check: fast suite 188 passed; `main.py run data/ibmchurn.csv --variant rules --budget 120` ends with
  all 12 stages ok (audit 3.7 s, 5 issues; test roc_auc 0.8532).

## 2026-10-19 · P14.3 · Leak review as a normal agent

- edatool's review agent (smolagents `ToolCallingAgent`, up to 8 parallel sessions with 10 read
  tools, action tools and a done gate) is replaced by `agents/leakreview.py`: one `LlmClient.ask`
  call with a pydantic answer, like the framer and the summary. It sees every column with its type
  and three example values, plus the G01 evidence of the audit, and returns only the columns that
  are known after the target event, each with a reason. Code checks the answer (known columns, never
  the target, a reason for each) and sends problems back; a failed call gives no semantic evidence.
- Why: the pipeline used only one output of the agent, the semantic leak judgement (G03 from the
  name review, or a confirmed G01). Its other work (confirming, dismissing and linking issues,
  setting types and roles) either duplicated the LLM cleaner, which reads the same issues, or was
  never read. The leak review keeps the part that found 5 of 5 leaks in the pilot; the prompt keeps
  the wording of edatool's name review (`semantic.md`).
- The variant option `review` is now `leakreview`. Its decisions are logged as stage `leakguard`,
  kind `review`, source `llm`. The decision table of the leak guard is unchanged.
- Deleted: `llms/agentmodel.py` (the smolagents bridge, which had its own copy of the budget
  checks), `tests/testagentmodel.py` (the test that made real calls) and `tests/testauditpolicy.py`
  (edatool agent internals). `smolagents` (and the packages only it needed) left `pyproject.toml`.
- New tests in `tests/testleakreview.py` (fake provider only): the planted after-event column is
  returned, wrong answers are sent back, a failure gives nothing, and in a `full` run the review turns
  a G01 warning into a drop.
- Check: a `full` run on ibmchurn with a fake provider (every call except the review fails) ends with
  all 12 stages ok; the cleaner falls back to the translator plan and the report has no summary.

## 2026-10-19 · P14.4 · Slim the cleaning checks

- Removed the `custom` operation: the cleaner could ask for an operation whose code an LLM then wrote
  (`prompts/customop.md`, `prepare_custom_steps`, `CustomOperation`). It was a second code-writing
  path next to the `freecode` variant, which stays as the code-writing comparison. The `freecode`
  operation moved to `operations/freecode.py`; the sandbox stays for it.
- Kept: the LLM cleaner, repair (at most 3 per step), refine (at most 2 rounds), the data checks and
  the gate. These are the plan, check and feedback loop the thesis studies.
- `tests/testsandbox.py`: the custom-step tests became one test of the `freecode` operation.

## 2026-10-19 · P14.5 · edatool and edabench deleted

- `edatool/` (the vendored auditor) and `edabench/` (its own benchmark: injector, labels, baselines
  ydata-profiling and cleanlab, the 1,091-line legacy `eda.py`) are deleted, with the edabench uv
  environment (`benchmark/envs/edabench`, 686 MB), `cache/edatool` and `docs/edatool-baseline.md`.
  Experiment E8 (audit quality) goes with them; its last numbers (recall 0.987, precision 0.855)
  stay in the git history and in the P9.12 line of `docs/progress.md`.
- `column_as_text` (OpenML column to CSV text) moved into `benchmark/datasets.py`;
  `benchmark/datacards.py` uses `run_audit`; `benchmark/resultsdoc.py` has no E8 section; the
  edatool and edabench paths, the ruff exclude and the frozen `edabench/legacy/` prefix are gone.
- Check: no Python, TOML, JSON or text file outside the docs names edatool, edabench or smolagents;
  fast suite 184 passed; `ruff check --select F` clean.

## 2026-10-19 · P14.6 · Small additions to modelling and evaluation

- Modelling had no LLM role at all. `agents/modeller.py` (one call, `prompts/modeller.md`) now
  chooses which of `linear`, `lightgbm` and `catboost` the engine searches and how the search time
  is split. It sees only facts code computed (task, metric, rows, column types, categorical columns
  with more than 50 levels, classes and the smallest class share, the fixed run budget). Code checks
  the answer (at least one boosting family, each family once, positive shares), keeps it only inside
  the configured candidates and their order, and normalizes the shares. Cross-validation still
  chooses the final model, so the LLM proposes and code verifies, as in the cleaning stages. The
  total budget stays the one from settings (design rule 5). Only the LLM variants use it; the
  others keep the engine defaults (`SHARES` in `modelling/engine.py`).
- The prompt gets the run budget from settings, not the time left: a fixed prompt can be cached, so a
  run with the same seed and a warm cache still repeats exactly (`tests/testreproducible.py`).
- Left out on purpose: class weights for imbalanced classes (they distort the probabilities that
  log-loss scores, and the engine has no check for them), feature engineering, stacking, SHAP and
  calibration.
- Evaluation was one test score. `modelling/evaluator.py` adds a permutation importance of the raw
  columns on the locked test set (demo mode only): at most 2,000 rows, two shuffles per column,
  stopped by the time left in the run (5 to 30 s). `report.md` shows the top 10 columns. Benchmark
  mode has no test set, so benchmark runs do not change.
- Check: a `full` run on ibmchurn with a fake provider (modeller answer lightgbm 2 : linear 1) searched
  dummy, lightgbm and linear, chose lightgbm + linear (CV 0.8478, test 0.8545); the importance ranks
  tenure, Contract and InternetService first (15 s for 20 columns). Fast suite 187 passed.

## 2026-10-19 · P14.7 · Variants (11 to 6)

- Kept: `generic` (no audit, no LLM), `rules` (audit and translator, no LLM), `full` (all LLM
  stages), `full-nogate` (RQ5: what the gate prevents), `freecode` (RQ5: typed operations vs free
  code) and `full+ag` (the plug-in comparison with AutoGluon). Together they answer RQ1 to RQ5:
  value of auditing (`generic` vs `rules`), value of the LLM (`rules` vs `full`), the gate, and
  typed operations.
- Removed: `full-noreview`, `full-norepair`, `full-nosamples`, `full-noleakguard` and `rules+ag`.
  With them the variant options `samples`, `repair`, `leakguard` and `operations` became constant
  and were removed. Value samples in prompts stay a setting (`cleaner.show_values`), and repair can
  still be switched off with `cleaner.max_repairs=0`; `repair_plan`, `gate_with_refine` and the
  leak guard lost their on/off parameters.
- `benchmark/systems/registry.json` lists the six variants; the grids follow in P14.8.

## 2026-10-19 · P14.8 · Benchmark at MSc core

- Kept: the benchmark data (OpenML pools, corruptor with 12 mess types and 4 levels, leak
  generator, natural-mess group), the resumable runner, scoring, the five baselines (dummy,
  LightGBM, skrub, FLAML, AutoGluon), the statistics (normalized score, average ranks,
  Friedman/Nemenyi, Wilcoxon with Holm, bootstrap intervals), figures, `docs/results.md` generation,
  one-command reproduction and the showcase.
- Experiments kept: E1 (score vs mess level, RQ1 and RQ2), E2 (`full+ag` vs AutoGluon), E3 (leaks,
  RQ3), E4 (reliability and cost, from the same runs), E5 (ablations: `rules`, `freecode`,
  `full-nogate` next to `full`, RQ5) and E10 (natural mess). Removed: E6 (per mess type, with the
  repair-accuracy check), E6b (reading robustness), E7 (framer accuracy on 40 datasets), E8 (audit
  quality, gone with edabench in P14.5) and E9 (budget sensitivity), and the power projection with
  the held-out cells. They answer side questions; the five research questions do not need them.
- Deleted: `benchmark/projection.py`, `framereval.py`, `performance.py`, `repairs.py`, their tests,
  `docs/projection.md`, the grids `pilot-seed`, `pilot-types`, `pilot-budget`, `reading`, `types`,
  `budgets` and their result files. The runner lost the held-out cells and the budget lists; the
  orchestrator lost `stop_after` and `prepared.parquet` (both only served the repair check).
- Grids now: `smoke` 36 cells, `pilot-main` 216, `pilot-leaks` 20, `pilot-ablations` 8,
  `pilot-natural` 12; full grids (written, not run) `main` 1,440, `leaks` 180, `ablations` 160.
- `benchmark/results/runs.csv` is renamed to `runs-before-p14.csv`: the runner resumes a grid from
  the rows in `runs.csv`, so the old pilot rows would make the re-run skip every cell. The old rows
  stay for comparison. `docs/results.md`, `tables.md` and the figures keep the old numbers with a
  notice at the top of `docs/results.md` until the re-run.
- Check: `figures.build` and `resultsdoc.build` run on a copy of the old rows (5 figures, sections
  RQ1 to RQ5 and E10); fast suite 184 passed.

## 2026-10-19 · P14.9 · Documentation

- `README.md` and `docs/architecture.md` describe the restructured pipeline: the package table, the
  12 stages with their LLM calls and fallbacks, the six variants and the 29 audit codes with their
  main thresholds.
- `docs/protocol.md` replaces `docs/preregistration.md` (still in the git history). It keeps the
  hypotheses, metrics, statistics and exclusion rules, adapts RQ5 and the grids to the six variants,
  and drops the projection and the held-out cells. It says openly that it is written after a first
  pilot of the earlier code, so it is not a pre-registration in the strict sense.
- `docs/deviations.md` records the method change and that every pilot cell is re-run;
  `docs/buildplan.md` has a note at the top and a P14 section; `docs/progress.md` has the re-run
  steps for the user, with a warning that the experiments cap leaves $1.13.
- The study (`study/`, git-ignored) was rebuilt for the new code with the code-study skill: 19 pages,
  every source file covered, the check reports it in sync. The old edatool pages were removed, and
  new pages explain the audit package, the orchestrator and CLI, and the benchmark.
