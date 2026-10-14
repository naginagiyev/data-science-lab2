# Architecture (one page)

The full design is in `buildplan.md` (§2 architecture, §3 stage contracts, Appendix B the operation
catalog, Appendix C the audit-to-step translation). This page is the map.

## Idea: the LLM proposes, code verifies

The LLM never writes the pipeline. It chooses operations from a fixed catalog of tested, typed
operations (`operations/`) and fills in their parameters as structured output. Code checks every
answer: unknown columns, bad parameters and steps on the target are refused and sent back; every
step must pass data checks on cross-validation folds; edatool verifies that the issues it was meant
to fix are gone; a cross-validation gate removes any step that makes the score worse. When an LLM
call fails, a rule-based fallback takes its place, so every run ends with a model and a report.

## The 12 stages (`orchestrator.py`)

| Stage | What it does | Main code | Fallback |
|---|---|---|---|
| 01 read | any CSV, Excel, JSON or parquet file to a raw text view; title rows, delimiters, encodings | `tools/reader.py`, `edatool` loader | hard stop with a clear message |
| 02 glance | per-column types, missing tokens, patterns (units, packed, dates) | `tools/glance.py` | minimal glance |
| 03 frame | target, task, metric, id columns: flags, then rules, then the LLM | `agents/framer.py` | rules |
| 04 split | locked test set (demo mode) and CV folds: stratified, grouped or by time | `tools/splitter.py` | random folds |
| 05 audit | edatool: about 50 detectors, then LLM agent sessions that review unclear issues | `edatool/`, `llms/agentmodel.py` | deterministic audit |
| 06 leak guard | ids and leaks dropped by evidence (structural, statistical, semantic) | `checks/leakage.py` | no drops |
| 07 plan | translator plan from the audit; the LLM cleaner writes a full plan | `operations/translator.py`, `agents/cleaner.py` | translator plan |
| 08 compile | data checks per step on folds, LLM repair of failing steps, edatool verify | `checks/datachecks.py` | protected steps only |
| 09 gate | P0 (protected steps) vs P1 (all steps), step tests, LLM refine rounds | `checks/gate.py` | P0 |
| 10 model | own engine (LightGBM, CatBoost, linear, Optuna, ensemble) or a backend | `modelling/engine.py`, `modelling/backends.py` | dummy model |
| 11 test | the locked test set, used once (demo mode) | `modelling/evaluator.py` | none |
| 12 report | `report.md`, scores, an optional checked LLM summary | `tools/report.py`, `agents/summarizer.py` | template only |

Every stage has a time limit from the run budget; every decision goes to `decisions.jsonl`; every
LLM call goes to `llm.jsonl` and the global spend ledger (`benchmark/results/spend.json`), which
enforces the caps.

## The fitted model

`model.joblib` holds the agent steps, a typing step, a generic encoding stage per model family and
the trained models (or the backend folder). It predicts from raw input, so the same messy file format
works at prediction time; generated code (custom steps, free code) is re-run in the sandbox.

## The benchmark (`benchmark/`)

Datasets (`datasets.py`), the corruptor with ground truth (`corruptor.py`, 12 mess types), the leak
generator (`leaks.py`), fixed splits and text files (`materialize.py`), one adapter per system in its
own environment (`systems/`, `envs/`), the resumable runner with hard time-outs (`runner.py`), the
harness scorer (`scoring.py`), and the analysis (`analysis.py`, `figures.py`, `resultsdoc.py`,
`projection.py`). Grids are JSON files in `benchmark/grids/`.
