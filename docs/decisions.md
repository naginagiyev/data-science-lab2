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
