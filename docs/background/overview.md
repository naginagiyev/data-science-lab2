# DS Lab 1 → DS Lab 2 / MSc Thesis: Critical Review

Review of the DS Lab 1 codebase (agentic ML automation pipeline) as a starting point for the DS Lab 2 / MSc thesis. It lists what is amateur, incorrect, or not thesis-grade, and how to fix each point.

The sections are ordered by how much they would hurt in front of a thesis committee. **Sections A and B matter most**: an examiner forgives rough engineering, but not a missing research question or flawed ML methodology.

---

## Table of Contents

- [A. Thesis framing (the biggest gap)](#a-thesis-framing-the-biggest-gap)
- [B. ML methodology errors](#b-ml-methodology-errors)
- [C. LLM-agent design](#c-llm-agent-design)
- [D. Security](#d-security)
- [E. Software engineering](#e-software-engineering)
- [F. Repo hygiene and documentation](#f-repo-hygiene-and-documentation)
- [Suggested priorities for DS Lab 2](#suggested-priorities-for-ds-lab-2)

---

## A. Thesis framing (the biggest gap)

### A1. It's a tool, not a study

**Problem.** The project is a pipeline that "takes a CSV and does everything". A thesis has to answer a question: *does* it work, *how well*, *compared to what*, and *why*. Nothing in the repo measures any of that. There is no benchmark, no baseline, and no results table. The six CSVs in `data/` are demo inputs, not an evaluation.

**Fix.**
- Write down a **research question**, e.g. *"Can an LLM multi-agent system match classical AutoML on tabular benchmarks, and which agent components actually contribute?"*
- Compare against **baselines**:
  - Classical AutoML: AutoGluon, auto-sklearn, FLAML, TPOT, H2O.
  - A trivial baseline: default Random Forest / LightGBM.
  - Other LLM data-science agents: AIDE, DS-Agent, Data Interpreter, AutoKaggle.
- Evaluate on a **standard benchmark**: the OpenML-CC18 suite, the AutoML Benchmark (AMLB), or a subset of MLE-bench / Kaggle.
- Report **more than one number**:
  - task success rate (did the pipeline finish?)
  - predictive score
  - wall-clock time
  - LLM tokens and cost
  - number of fix-loop iterations
- Run every setting with **several seeds** and report mean ± std. Use **statistical tests** (Wilcoxon signed-rank, critical-difference diagrams).
- Run **ablations**: remove the EDA report, the fix loop, the tuning agent, or the consultant, and measure what each one adds. This is what makes agent research publishable.
- Write a **related-work chapter** covering AutoML and LLM-for-data-science agents. Right now the novelty is unclear.

### A2. The README promises more than the code delivers

**Problem.** The README claims support for 6 task types (binary, multi-class, regression, anomaly detection, time series, clustering), but:
- `inferTaskTypeFromTarget` can only return binary, multi-class, or regression.
- `configuration/modeloptions.md` lists ARIMA/SARIMA, Prophet, LSTM, and Autoencoder. The training plan requires ONNX export, but Prophet and ARIMA can't be exported to ONNX, and `prophet` / `torch` are not even in `requirements.txt`.
- Anomaly detection runs with `targetCol = null`, so its metrics are computed against the feature matrix (see B4).

**Fix.** An examiner will test such a claim. Either **scope down** to what you can prove (tabular classification + regression is enough for a thesis), or implement *and evaluate* the other task types properly.

---

## B. ML methodology errors

### B1. Data leakage in preprocessing

**Problem.** The processing plan (`prompts/processingplanprompt.md`, "Fit and Save") fits the preprocessor on the **full dataset**. Only afterwards does `tools/datasplitter.py` split the *already processed* data. Imputers, scalers, and encoders have therefore seen the test rows.

**Fix.** Split the **raw** data first. Fit the preprocessor on train only, then apply (`transform`) it to validation and test.

### B2. The test set is used for tuning

**Problem.** `agents/evaluationagent.py` (`tune`) scores on the test set, feeds that score to the LLM to choose new hyperparameters, and repeats until the test score passes the threshold. After that, the "test score" is really a validation score, and there is no unbiased final estimate. Also, `constants.json` sets `trainSplit=0.8` and `testSplit=0.2`, so a validation set is never created.

**Fix.** Use train / validation / test (or cross-validation on train plus a locked test set). Tune only on validation. Touch the test set **exactly once**, at the very end.

### B3. The split ignores the task type

**Problem.** `tools/datasplitter.py` uses a plain shuffled `train_test_split` for every task:
- No **stratification** for imbalanced classification. `creditcardfraud.csv` is the textbook case where this matters.
- **Time series gets shuffled**, which leaks the future into training.

**Fix.** Use a stratified split for classification and a chronological split (e.g. `TimeSeriesSplit`, or cut by date) for time series.

### B4. The metrics are computed wrongly

**Problem.** In `tools/metrics.py` and `EvaluationAgent.getPredictions`:
- Only ONNX `output[0]` is used, which for sklearn classifiers is the predicted **label**. The probabilities are in `output[1]`. So **ROC-AUC, Log-Loss, and Average-Precision are computed from hard labels** and are meaningless.
- `pred.round()` is applied to class predictions.
- F1 / Precision / Recall are always `average='weighted'`, even for binary fraud detection. There, positive-class F1 or PR-AUC is what matters.
- MAPE divides by zero when a true value is 0.
- Anomaly detection / clustering pass `true=df.values`, i.e. the features, into supervised metrics.
- Scores are rounded to 2 decimals, so "did this change help?" comparisons are too coarse.

**Fix.** Use probability outputs for ranking and probabilistic metrics. Choose the averaging mode per task (binary vs. multi-class). Use a safe MAPE (or sMAPE / MAE instead). Store scores at full precision and round only for display.

### B5. The tuning loop has logic bugs

**Problem.**
- The output of `act()` (the re-run of the LLM-edited training script) is **never checked**. If the edited `train.py` crashes, the old `model.onnx` stays in place and the history records a change that never happened.
- The model from the **last iteration is never evaluated**. Whatever the final tuning step produced is what gets deployed, even if it's worse.
- Three rounds of "the LLM guesses new hyperparameters" is not tuning. There is no search space, no CV, and no comparison with random search / Optuna.

**Fix.** Use real hyperparameter optimisation (Optuna or random search with CV). If you keep LLM-guided tuning, make **"LLM-guided vs. Optuna" an experiment**. It fits the research question well.

### B6. There is no model selection or baseline

**Problem.** The LLM picks one model from a list of three and never compares it against anything. There is no trivial baseline (majority class / mean predictor) to show that the model learned something at all.

**Fix.** Train several candidates, select on validation, and always report a dummy baseline next to the chosen model.

### B7. The success thresholds are invented by the LLM

**Problem.** When the user doesn't give `minScoreRequirement`, the LLM makes one up (`prompts/configfiller.md`), and that invented number decides when tuning stops.

**Fix.** Use a fixed optimisation budget (time or trials) as the stopping criterion. Treat any user threshold as a report item, not a stopping rule.

### B8. Task and target detection are fragile

**Problem.**
- The target detector sees only **2 sample values per column**.
- The task-type heuristic calls any 2-valued column binary and anything else non-regression multi-class. A 0–10 wine-quality score becomes 11-class classification.

**Fix.** Give the detector richer column statistics (dtype, cardinality, distribution summary, position, name). This is the supposedly "intelligent" part of the system, so **evaluate it**: measure target and task-type detection accuracy on N labelled datasets.

---

## C. LLM-agent design

### C1. The LLM judges things code can check deterministically

**Problem.**
- The training "pass/fail" decision is made by an LLM reading stdout (`TrainAgent.reason`).
- The processing check asks an LLM whether dtypes are numeric and non-null. That's a two-line pandas check.

**Fix.** Use hard checks for what is checkable: exit code, model file exists, ONNX loads, it predicts on validation with the right shape, no NaNs, all numeric. Keep the LLM for **diagnosis and repair only**. A reviewer will call the current approach unreliable and wasteful.

### C2. Input columns for deployment are guessed by the LLM

**Problem.** `tools/featuredecider.py` asks the LLM which raw columns the API needs by reading the preprocessing code.

**Fix.** The fitted pipeline already knows this: read `feature_names_in_` from the preprocessor.

### C3. Runs are not reproducible and have no observability

**Problem.**
- No temperature or seed is set.
- Prompts, responses, tokens, and costs are not logged.
- There are no retries or rate-limit handling around API calls.
- The model name `gpt-4.1-mini` is hardcoded in two places.

**Fix.**
- Log **every LLM call** per run (prompt, response, tokens, latency, cost) so you can report cost and debug failures.
- Add retries with backoff.
- Make the LLM configurable so you can compare models (small vs. large, or across providers) as another experiment.

### C4. The prompts are inconsistent and copy-pasted

**Problem.**
- `processingplanprompt.md` says "Do not use comments in the code", while `codexcodeprompt.md` says to add section comments.
- Both reasoning prompts end with "Do not change the length or logic of the original prompt… restructure for clarity". This doesn't make sense in context and looks pasted from another prompt.
- The metric lists are duplicated in three prompts **and** in `metrics.json`, so they will drift apart.
- There are typos ("purpse", "there metric", "the followings").

**Fix.** Prompts are part of your method. Treat them like code: keep a single source of truth (inject metric lists from `metrics.json`), version them, and review them.

### C5. The target label mapping is lost

**Problem.** Text targets (Yes/No) are mapped to integers outside the pipeline, and the mapping is not saved. The API returns `0`/`1` with no way back to the original labels.

**Fix.** Fit and persist a label encoder alongside the preprocessor, and inverse-transform predictions in the API.

---

## D. Security

These are serious for anything demoed publicly.

### D1. Secrets are uploaded to the VM

**Problem.** `.env` lives in `vm/` (`paths.py`: `envPath = vmDir / ".env"`), and `tools/transfer.py` uploads **everything** under `vm/`. Your OpenAI key, VM password, and ngrok token end up on the server.

**Fix.** Keep `.env` at the project root (outside any upload folder) and upload an explicit allow-list of files.

### D2. LLM-generated code runs unsandboxed

**Problem.** Generated scripts run with your user's full permissions and network access. `sandbox/` is just a folder name.

**Fix.** Run generated code inside a Docker container (or similar) with no secrets, restricted file-system access, and CPU/memory/time limits.

### D3. The deployment is insecure

**Problem.**
- It logs in as **root with a password**.
- `AutoAddPolicy` skips host-key verification, which allows man-in-the-middle attacks.
- The public ngrok endpoint has **no authentication** and no input validation (fields are typed `Any`).
- It runs `kill -9` on whatever process is using port 5000.
- `rm -rf '\root\vm'` in `transfer.py` uses Windows-style backslashes on a Linux host, so it doesn't delete what was intended.

**Fix.** Use SSH keys with a non-root user and known hosts. Put an API key on the endpoint and give the input schema real types. Better still, containerise the API (Docker image with the model baked in) and deploy that.

### D4. `clean.py` can delete real data

**Problem.** It removes any file in `data/` whose name contains "train", "test", "val", or "processed". `evaluation.csv` or `intervals.csv` would be deleted.

**Fix.** Keep raw data read-only and write all generated files to a per-run output directory (see E2). Cleaning then means deleting a run folder.

---

## E. Software engineering

### E1. There are three near-duplicate entry points

**Problem.** `main.py`, `main2.py`, and `app.py` are near copies. Inside `main.py`, the whole pipeline is copy-pasted twice (auto vs. interactive mode). `inferTaskTypeFromTarget` exists three times, and `loadAllowedMetrics` / `normalizeMetric` twice.

**Fix.** Use one orchestrator with a pipeline definition (a list of steps) and one CLI with flags (`--auto`, `--verbose`, `--config`).

### E2. One JSON file is shared global state

**Problem.**
- Every component reads and writes `configuration/configuration.json`. Components are coupled only through that file, and the same dict is aliased as `consultation`, `runtimeConfig`, and `configuration`.
- Every run overwrites `sandbox/`, `models/`, and `data/train.csv`, so **two experiment runs can't coexist**. A thesis needs exactly that.

**Fix.** Pass a typed run context (Pydantic model) between steps. Write each run to its own `runs/<timestamp>_<dataset>/` directory containing config, artifacts, logs, and metrics. Consider MLflow (or similar) for experiment tracking.

### E3. Scripts execute at import time

**Problem.** `deployer.py`, `transfer.py`, and `featuredecider.py` do all their work at module level and are invoked through `runpy`. `exit(1)` inside the deployer silently kills the entire pipeline.

**Fix.** Turn them into functions with explicit arguments, return values, and proper exceptions.

### E4. There are zero tests

**Fix.**
- Unit tests for the deterministic parts: metrics, splitter, EDA type inference, data reader.
- An integration test of the whole pipeline with a mocked LLM.
- CI via GitHub Actions.

### E5. Error handling is weak

**Problem.** A `None` metric or threshold crashes (`self.metric.upper()`, comparisons with `None`). Bare `except Exception: pass` blocks hide failures in the deployer.

**Fix.** Validate the configuration once, up front (Pydantic with required fields and allowed values). Fail loudly, with clear messages.

### E6. The code style is inconsistent

**Problem.** Python code uses camelCase (not PEP 8), mixed with snake_case (`custom_style`, `item_path`). Type hints are partial, and some are wrong (`reason(self, output: str) -> dict` receives a dict and returns a `ReasoningResult`).

**Fix.** Adopt PEP 8, run `ruff` (lint + format), and add type hints consistently. Optionally check with `mypy`.

---

## F. Repo hygiene and documentation

- **Stale and corrupted README.** It still says "DS LAB I" and ends with corrupted UTF-16 text (`#   d a t a - s c i e n c e - l a b 2`).
- **UTF-16 requirement files.** Both `requirements.txt` files are UTF-16 encoded, which usually means they came from `pip freeze >` in PowerShell.
- **Conflicting dependency declarations.** They live in three places that disagree:
  - `pyproject.toml` lists no dependencies.
  - `uv.lock` is effectively empty.
  - `requirements.txt` is a full freeze including transitive packages.

  → Declare direct dependencies in `pyproject.toml` and let `uv lock` pin them.
- **Build artifacts committed.** `__pycache__/*.pyc` and `dslab1.egg-info/` are in git. `sys.dont_write_bytecode = True` is a workaround for a missing `.gitignore` rule. `pyproject.toml` also packages `data`, `sandbox`, and `models` as Python packages.
- **Git history is one dump commit.** For the thesis, commit incrementally with meaningful messages; your supervisor may look at the history.
- **Missing docs.** There is no architecture diagram, no LICENSE, and no "how to reproduce the results" section.

---

## Suggested priorities for DS Lab 2

1. **Define the research question and evaluation protocol** (benchmark, baselines, metrics, seeds) *before* adding features.
2. **Fix the methodology**: split first, keep a locked test set, use stratified and time-aware splits, and compute metrics from probabilities.
3. **Isolate runs and log every LLM call**, so experiments are reproducible and comparable.
4. **Replace LLM judgments with deterministic checks** wherever possible, and sandbox code execution.
5. **Narrow the scope** to what you can evaluate rigorously (tabular classification + regression), then run ablations.
6. **Clean up the engineering**: single entry point, tests, dependencies, secrets.
