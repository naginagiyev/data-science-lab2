# Architecture (one page)

The design history is in `buildplan.md`; this page is the map of the code as it is after the P14
restructuring.

## Idea: the LLM proposes, code verifies

The LLM never writes the pipeline. At six decision points it gives a structured answer that code can
check: the target and task, the columns known only after the event, a cleaning plan from a fixed
catalog of tested, typed operations, repairs and refinements of that plan, the model families with
their time shares, and a summary paragraph. Code checks every answer: unknown columns, bad
parameters and steps on the target are refused and sent back; every step must pass data checks on
cross-validation folds; a cross-validation gate removes any step that makes the score worse; and
cross-validation picks the final model. When an LLM call fails, a rule-based fallback takes its
place, so every run ends with a model and a report.

## Packages

| Package | Role |
|---|---|
| `tools/` | reader (raw text view), column types and value patterns, glance, splitter, run folder, metrics, report, sandbox |
| `audit/` | the code-only audit: 28 checks, 29 issue codes with default fixes (`configuration/audit.json`) |
| `operations/` | 22 typed cleaning operations, the condition language, the translator and the compiler |
| `checks/` | data checks per step, leak guard, validation gate |
| `modelling/` | engine with Optuna, CV helpers, evaluator with permutation importance, fitted model, backends |
| `agents/` | framer, leak review, cleaner (plan, repair, refine), free coder, modeller, summarizer |
| `llms/` | the only door to the LLM: cache, budgets, retries, validation, logging |

## The 12 stages (`orchestrator.py`)

| Stage | What it does | Main code | Fallback |
|---|---|---|---|
| 01 read | any CSV, Excel, JSON or parquet file to a raw text view; title rows, delimiters, encodings | `tools/reader.py` | hard stop with a clear message |
| 02 glance | per-column types, missing tokens, patterns (units, packed, dates) | `tools/glance.py`, `tools/columntypes.py` | empty glance |
| 03 frame | target, task, metric, id columns: flags, then rules, then the LLM | `agents/framer.py` | rules |
| 04 split | locked test set (demo mode) and CV folds: grouped, by time, stratified or random | `tools/splitter.py` | random split |
| 05 audit | 28 checks on the training rows | `audit/` | empty report |
| 06 leak guard | LLM leak review, then drops by evidence (structural, statistical, semantic) | `agents/leakreview.py`, `checks/leakage.py` | framer ids only |
| 07 plan | translator plan from the issues; the LLM cleaner writes a full plan | `operations/translator.py`, `agents/cleaner.py` | leak guard steps only |
| 08 compile | data checks per step on folds, LLM repair of failing steps | `checks/datachecks.py`, `agents/cleaner.py` | protected steps only |
| 09 gate | P0 (protected steps) vs P1 (all steps), step tests, LLM refine rounds | `checks/gate.py` | P0 |
| 10 model | LLM modeller (families, time shares), then the own engine or a backend | `agents/modeller.py`, `modelling/engine.py`, `modelling/backends.py` | dummy model |
| 11 test | the locked test set, used once (demo mode), and permutation importance | `modelling/evaluator.py` | none |
| 12 report | `report.md`, `scores.json`, an optional checked LLM summary | `tools/report.py`, `agents/summarizer.py` | template only |

Every stage has a time limit from the run budget; every decision goes to `decisions.jsonl`; every
LLM call goes to `llm.jsonl` and the global spend ledger (`benchmark/results/spend.json`), which
enforces the caps.

## The audit codes

| Group | Codes | Checks (file) |
|---|---|---|
| A structure | A02 saved index, A04 duplicate rows, A05 conflicting duplicates | `audit/structure.py` |
| B formats | B01 numbers as text, B02 dates as text, B04 numeric codes, B05 label spellings, B06 boolean tokens, B07 units, B08 packed values, B09 date formats | `audit/formats.py` |
| C missing | C02 hidden tokens, C03 sentinels, C04 informative missingness, C05 co-missing blocks, C06 almost empty columns | `audit/values.py` |
| D validity | D01 impossible values, D02 extreme values, D03 placeholder dates, D05 rare categories | `audit/values.py` |
| E columns | E01 constant, E02 quasi-constant, E03 id, E04 many levels, E05 free text, E06 one-to-one, E08 high correlation | `audit/structure.py` |
| F, G target | F03 class spellings, G01 one feature predicts the target (tree or purity) | `audit/target.py` |

The thresholds are in `configuration/audit.json`: for example B07 needs 90% of at least 20 values
with a unit, D02 a robust z above 10 or a gap of 5 IQRs on at most 1% of the values, E08 a rank
correlation of 0.95, and G01 an AUC of 0.98 or an R² of 0.95 for a depth-3 tree on one feature.

## The fitted model

`model.joblib` holds the agent steps, a typing step, a generic encoding stage per model family and
the trained models (or the backend folder). It predicts from raw input, so the same messy file format
works at prediction time; the code of a `freecode` step is re-run in the sandbox.

## The benchmark (`benchmark/`)

Datasets (`datasets.py`), the corruptor with ground truth (`corruptor.py`, 10 mixed mess types at 4
levels), the leak generator (`leaks.py`), fixed splits and text files (`splits.py`,
`materialize.py`), one adapter per system (`systems/`, `envs/`), the resumable runner with hard
time-outs (`runner.py`), the harness scorer (`scoring.py`), and the analysis (`analysis.py`,
`figures.py`, `resultsdoc.py`). Grids are JSON files in `benchmark/grids/`.
