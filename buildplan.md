# Agentic AutoML v2: build plan

This file is the complete work order for DS LAB 2. You (Claude Code) will carry it out on your own, from the first commit to the final report. The user will not answer questions while you work. When this plan leaves a detail open, choose the simplest option that keeps the research honest, and write the choice into `docs/decisions.md`.

- Date of this plan: 10 October 2026.
- DS LAB 2 deliverables: due around the end of January 2027. This plan builds only the code; the user writes the report.
- Thesis (next semester, same project): due at the end of May 2027.

## Contents

- [0. Read me first](#0-read-me-first)
- [1. Context](#1-context)
- [2. Target architecture](#2-target-architecture)
- [3. Component specifications](#3-component-specifications)
- [4. Benchmark and experiments](#4-benchmark-and-experiments)
- [5. Testing](#5-testing)
- [6. Phases and tasks](#6-phases-and-tasks)
- [7. Definition of done](#7-definition-of-done)
- [8. Risks and gotchas](#8-risks-and-gotchas)
- [Appendix A. Prompt skeletons](#appendix-a-prompt-skeletons)
- [Appendix B. Operation catalog](#appendix-b-operation-catalog)
- [Appendix C. edatool fix to operation translation](#appendix-c-edatool-fix-to-operation-translation)
- [Appendix D. Dataset pools](#appendix-d-dataset-pools)
- [Appendix E. Mess types and leak specs](#appendix-e-mess-types-and-leak-specs)
- [Appendix F. Project CLAUDE.md](#appendix-f-project-claudemd)
- [Appendix G. progress.md template](#appendix-g-progressmd-template)

---

## 0. Read me first

### 0.1 Mission

Build **v2** of the user's agentic AutoML system and the evidence that it is useful:

1. **The system.** An agent for tabular binary classification, multi-class classification and regression. The LLM proposes typed data-preparation operations. Deterministic code verifies, compiles and validates them. A standard engine trains the model. The test set is used once. Every run ends with a model and a report.
2. **The auditor.** The user's new EDA tool (`edatool`, in `../eda`) becomes the "understand" stage. You copy it into this project and improve it.
3. **The benchmark code and a small pilot.** Build the complete benchmark (messy tables with known ground truth, baselines, ablations, statistics), but run only the **pilot** (about 18% of the full benchmark, §4.6) and a **projection** of what the full benchmark would show (§4.10). The full grids run only if the user asks later. The user's money and computer time are limited: **the whole plan must cost about $4 of LLM calls, never more than $5.**
4. **Documentation and demo.** A README with one reproduction command, short code documentation and a demo. **Do not write a report or a paper** (no LaTeX, no paper text, no reference list). The user writes those. Your deliverable is working, well-written, tested code plus the result files it produces.

The thesis niche is fixed: **messy, real-world tables**. The main hypothesis: on messy tables, "LLM proposes, code verifies" builds better models than classical AutoML, and it is more reliable and cheaper than agents that write free code. On clean tables it is not worse. Your job is to test this hypothesis fairly, not to prove it.

### 0.2 Read these first, in this order

1. This file, completely.
2. `architecture/v2-plan.html` (the research rationale: niche, RQ1 to RQ5, design principles, benchmark, ablations). Read Parts 1, 2 and 4. This plan replaces it where they differ (see §1.4).
3. `architecture/overview.md` (the problems of v1; this plan uses its codes, for example B1 or C3).
4. `../eda/README.md`, then `../eda/architecture.html` sections 1 to 11, then skim the code in `../eda/edatool/`.
5. `../II/tools/edaadapter.py` and the first 120 lines of `../II/sandbox/run_wine_check.log` (an earlier attempt to plug edatool into v1; lessons in §1.3).

Never modify `../eda`, `../I` or `../II`. Copy what you need into this folder.

### 0.3 Working protocol

1. **Phases in order.** Do the phases of §6 in order and the tasks inside a phase in order, unless a task says "parallel". Every task has a "done when" line. A task is done only when that line is true.
2. **`progress.md`** (project root) is your memory. Create it from Appendix G. After every task, update the status, date, commit hash and a one-line note. At the start of every session, and after any context reset, read `progress.md` first, then the current phase of this file.
3. **`docs/decisions.md`** records every judgment call: date, decision, reason, alternatives. **`docs/deviations.md`** records every change to a pre-registered experiment (from P9.2 on).
4. **Git.** Commit after every task. Message: `P<phase>.<task>: <what changed>`, plus any attribution lines your environment requires. Never push, never force, never rewrite history. Never commit `.env`, data files, caches or run folders.
5. **Tests.** Before every commit that touches code, `uv run pytest -q` must pass (fast suite: under 3 minutes, no network, no LLM).
6. **Long jobs** (more than 10 minutes, for example benchmark grids) run in the background. They must be resumable and log to a file. Work on other tasks while they run, then check back. Never run a benchmark grid in the foreground, and never start the full grids of §4.6 (pilot grids only).
7. **Phase summary.** At the end of each phase, add a short summary to `progress.md` in simple English (B2 level): files created, edited and deleted, and what changed. This is the user's "After a Task" rule.
8. **When blocked.** Try the documented fallback. If you are still blocked after the timebox of the task, mark the task `[!]` with the reason, and continue with the next task that does not depend on it. Never fake progress.
9. **No questions to the user.** Use the defaults in §0.5 and record judgment calls in `docs/decisions.md`.

### 0.4 Integrity rules (never break these)

- Never invent numbers, results or datasets. Every number in the documentation is produced by code from `benchmark/results/` (generated tables).
- Write hypotheses as hypotheses until the data supports them. Report negative results and failed experiments.
- Benchmark test labels are read only by the harness scorer. Develop and tune on the **dev pool** only (§4.1). After the freeze (P9.3), do not change the method because of pilot results. Bug fixes are allowed only with an entry in `docs/deviations.md` and a re-run of every affected cell.
- In every comparison: same time budget, same machine, same LLM for every LLM system, same train and test files.
- Secrets: never print, log, copy or commit `.env` or the API key. Remove the key from the environment of every sandbox subprocess.
- LLM spend: respect the cap in §0.5. The LLM client enforces it (§3.15).

### 0.5 Defaults

The user may change these before you start. Otherwise use them as they are.

| Setting | Default | Where it lives |
|---|---|---|
| Python | 3.12, **x86_64 build** (runs emulated on this ARM laptop) | `.python-version` |
| LLM | OpenAI `gpt-4.1-nano` for development (P1 to P7.8), `gpt-4.1-mini` for the final dev check (P7.9), the pilot and the demo; temperature 0, seed 0 | `configuration/settings.json` → `llm` |
| LLM spend for the whole plan | **target about $4, hard maximum $5**: development $1.50, pilot and projection $2.75, the rest is a safety buffer | `settings.json` → `spend` |
| Time budget of one pilot run | 120 s wall clock, the same for every system (300 s in the budget check) | `benchmark/grids/*.json` |
| Seeds | 1 for LLM variants, 2 for free systems in the pilot | grids |
| Full benchmark grids | built and tested with a tiny smoke run, **not run** | only if the user asks |
| Report or paper | None. Code, tests, result files and README only | — |

Check the model names and prices at the start (P0.5). If `gpt-4.1-mini` or `gpt-4.1-nano` is not available any more, use the closest current OpenAI model in price that supports structured outputs and tool calling, and record the change.

**Money rules** (the user has $5 in total, and the target is $4):

- Every real LLM call goes through the client, which writes it to the spend ledger. The client refuses calls when the ledger reaches the cap of the current category or **$5 in total**.
- At **$4 in total**, stop all optional LLM work (dev iteration, extra ablations), write a note in `progress.md`, and finish the remaining tasks with the deterministic variants and the LLM cache.
- Tests never call the real API (fake provider), except the few `llm`-marked checks named in the tasks.
- Never pay twice for the same answer: the LLM cache and the edatool agent cache are always on.
- Use small data while developing (samples of at most 5,000 rows).
- The `rules` and `generic` variants and all baselines except v1 and AIDE cost $0. Use them wherever they answer the question.
- v1 and AIDE need many LLM calls each. Build their adapters only if there is spare time, and never run them in this plan.

### 0.6 Coding rules

The user's global rules (`~/.claude/CLAUDE.md`) load automatically and apply to all Python you write. The ones that matter most:

- Double quotes. Lines around 100 characters.
- Exactly **one** empty line between imports, functions, classes and main code. Never two. No empty line at the end of any file; the last line ends with one newline.
- All imports at the top, in one block, **sorted by line length** (shortest first), `from x import y` lines included.
- `snake_case`, `PascalCase`, `UPPER_CASE`. Short but informative names; no abbreviations (except `np`, `pd`, `plt`, `i`, `e`).
- New Python files and folders: lowercase letters and digits only, **no underscores** (`__init__.py` and `__main__.py` are fine). Other files: short, alphanumeric, `-` allowed.
- Paths: never plain strings. Build them with `pathlib` from the project root and keep them all in the root `paths.py`. (`edatool/paths.py` stays, because edatool is a self-contained package.)
- Keep it simple. No frameworks, no plugin systems, no async, no web UI, no database.

**Do not run `ruff format` or `black`.** They insert two empty lines between functions and reorder imports, which breaks the user's rules. Lint only with `uv run ruff check --select F` (unused imports, undefined names).

This plan is the user's request, and the user's rules say that a request wins over a rule. So these defaults are **overridden** for this project:

1. Restructuring and rewriting existing files is required (this is a rebuild). Apply the style rules to the whole codebase.
2. Error handling is required at every stage boundary (fallbacks, §3.16) and inside operations. Never use a bare `except: pass`; every caught exception is logged to the run folder.
3. Logging is required: `run.log`, `decisions.jsonl`, `llm.jsonl`. This is research data.
4. Validation with Pydantic is required for settings, LLM outputs and plans.
5. Tests with pytest are required (§5).
6. `if __name__ == "__main__":` guards are required in every entry point (Windows starts subprocesses with "spawn").
7. Generated code runs only in the sandbox (§3.18).
8. Comments: write one short `#` comment above every function and class, in the user's style: lowercase (code names keep their case), simple English, about 80 characters per line, continue with `...` at the end of the line and on the next `#` line. No docstrings. Copy the style of `../eda/edatool/` (it already follows this rule).
9. No type hints in function signatures (edatool style). Pydantic model fields need annotations; that is fine.

---

## 1. Context

### 1.1 The thesis in one paragraph

AutoML systems are strong on clean tables, but real tables are messy: numbers stored as text with units, dates as text, categories with many spellings, hidden missing values, ID columns and leaky columns. LLM agents can read such tables, but agents that write free code crash, leak and cost a lot. v2 lets the LLM **choose and parameterize tested operations**, and lets **code verify** every step: data checks, issue-level verification with edatool, and a cross-validation gate. The evidence is a benchmark of controlled and natural mess.

| RQ | Question | Hypothesis (not a result) |
|---|---|---|
| RQ1 messy data | On messy tables, does v2 beat classical AutoML and code-writing agents? Does the gap grow with the mess level? | Yes, and the gap grows with the level. |
| RQ2 no harm | On clean tables, is v2 as good as the same engine without the agent? | No significant difference. |
| RQ3 leakage | Does v2 find leaky columns that AutoML uses? | High recall, few false alarms; AutoML collapses at deployment. |
| RQ4 reliability and cost | Success rate, variance, time, cost? | About 100% success; lower cost and variance than code agents. |
| RQ5 design | Which parts matter (typed operations, gate, repair, audit review, value samples)? | Typed operations beat free code; the gate prevents harm. |

Planned contributions: (1) the messy-table benchmark with ground truth, (2) the method, (3) the empirical study, (4) the plug-in result "v2 preparation + AutoGluon beats AutoGluon alone" if the data supports it, (5) the audit-quality study of edatool (its own benchmark).

### 1.2 What exists and what happens to it

| Thing | Where | What you do |
|---|---|---|
| v1 (DS LAB 1): 17 stages, the LLM writes whole scripts, ONNX, leakage (B1, B2), wrong metrics (B4) | this folder (a copy of `../I` without `.git`) | Freeze a copy as the "v1" baseline (P0.4), then delete it from the root (P1.2). |
| edatool: deterministic auditor (53 detectors, 43 in audit mode) + smolagents review agent with a done gate + `presplit` + `verify` | `../eda/edatool` | Copy into `edatool/` (P3.1) and improve (P4). |
| edatool benchmark: injector, evaluation, baselines (ydata-profiling, cleanlab, old eda.py) | `../eda/benchmark` | Copy into `edabench/` (P3.1). It measures the auditor alone. |
| v2 research plan | `architecture/v2-plan.html` | Rationale. This file is the build plan. |
| Integration attempt "master 2.0" | `../II` | Read-only. Lessons below. |
| Local datasets: `ibmchurn.csv`, `kinghousedata.csv`, `winequality.csv`, `creditcardfraud.csv` | `data/` (git-ignored) | Dev pool (§4.1). |

### 1.3 Facts measured before writing this plan

- edatool audit **without** the agent (audit time / process wall clock): winequality 2.2 s / 4 s, kinghousedata 3.3 s / 5 s, ibmchurn 3.6 s / 16 s (the first Python start is slow under emulation), credit_score 6.3 s / 9 s, creditcardfraud (284,807 rows) 20.8 s / 23 s. The single-feature leakage check (G01) is the slowest detector on small data.
- With the agent (from `../II` logs): about 36,000 tokens for winequality (2 sessions), and 126 s for the whole EDA stage on creditcardfraud.
- `../II` logging crashed with `UnicodeEncodeError` (Windows console code page cp1252) on the symbols `✓ → ·`. Fix this (§8, P4.2).
- edatool's fixes are **recommendations**, sometimes with missing parameters and sometimes unsafe for ML. Examples from real runs:
  - `replace_values {"column": "bedrooms"}` without a mapping (bedrooms = 33 outlier);
  - drop `waterfront` because it is quasi-constant, although it is a strong price feature;
  - `cap_values` on the regression target `price`, and `group_rare` on the classes of the target `quality`; both would change what is evaluated;
  - `drop_rows` for conflicting duplicates (label noise) without saying "training rows only".
  This is why v2 needs a translation layer, a policy for the target, and a validation gate.
- creditcardfraud produces 20 "extreme isolated values" review items on the anonymized columns V1 to V28. They cost agent calls, and the extreme values are exactly the fraud signal. Anonymized columns need a cheap path (P4.5).
- The local `winequality.csv` is red wine only: 1,599 rows, 240 duplicate rows, target `quality` from 3 to 8, class 3 has 10 rows.
- `../eda/data/credit_score.csv` is already label-encoded with an index column. It is not a messy dataset. Use it only as a "processed data" torture case.

### 1.4 What this plan changes compared with v2-plan.html

1. Stages "Profile" and "Audit columns and leaks" are replaced by **edatool** (`audit` with agent review). `tools/eda.py` is deleted; its old version lives on as a baseline inside `edabench/legacy/`.
2. The split uses edatool's **`presplit` advice** (group, time, stratified or random).
3. A new rule: **raw text in**. The pipeline input is edatool's raw text view (every value is a string). Every parse is an explicit, logged operation. A deterministic **typing step** (pandas-like) runs after the agent operations (§3.11).
4. The operation toolbox is extended and **mapped to edatool's fix vocabulary** (Appendix C), and edatool's `verify` checks the plan at the issue level.
5. New variants: `generic` (no audit, no operations), `rules` (deterministic audit and translator, no LLM anywhere) and `full`. This splits "value of auditing" from "value of the LLM".
6. Time budget 300 s per run (laptop), with a budget-sensitivity check (E9).
7. Base datasets come from **TabArena-v0.1** on OpenML (curated, licensed, readable names), with a separate dev pool.
8. Not used in semester 1: MLflow, LiteLLM, Docker sandbox, AutoGluon Assistant, feature engineering (CAAFE-style ideas are a stretch goal).

### 1.5 The machine

- Windows 11 on **ARM64** (Snapdragon X, 8 cores, 16 GB RAM, about 5 GB free when idle, no CUDA GPU). `uv` 0.12 is installed. Python 3.10 and 3.11 x86_64 builds are installed; 3.12 x86_64 is available for download (`cpython-3.12.14-windows-x86_64-none`).
- Use **x86_64 Python** (it runs emulated): LightGBM, CatBoost and PyTorch do not publish Windows-ARM64 wheels. After every environment creation, check that `platform.machine()` returns `AMD64`.
- WSL2 Ubuntu (arm64) and Docker Desktop are installed but stopped. Use WSL only as a fallback for a baseline that fails on Windows, and record it.
- Memory is tight: run one heavy system at a time per worker, at most 2 workers, 4 threads per run.
- The project path contains spaces (`DS LAB II`). Quote paths in shell commands.
- Console encoding: every entry point calls `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` (and the same for `stderr`). Subprocesses get `PYTHONIOENCODING=utf-8`. File log handlers use `encoding="utf-8"`.

---

## 2. Target architecture

### 2.1 Design rules

Rules 1 to 8 come from v2-plan.html; 9 to 12 are new.

1. **LLM proposes, code verifies.** The LLM never decides what code can check.
2. **Typed operations first, free code last.** Free code is a sandboxed escape hatch.
3. **Leak-free by construction.** Split first; everything that learns is fitted on training rows only; the test set is used once.
4. **Every stage has a fallback.** A run always ends with a model and a report.
5. **Fixed rules decided in advance.** Metric, budget, seeds and splits come from settings, never from the LLM.
6. **One run = one folder.** Runs never overwrite each other.
7. **Budgets everywhere.** Time per stage, LLM calls and dollars per run, and a global spend cap.
8. **Same code for demo and benchmark.** One orchestrator function.
9. **Raw text in.** Input is the raw text view; every parse is an operation.
10. **edatool is the only reader and the only auditor.**
11. **Training rows only for row filters.** Row filters never touch validation folds or test rows.
12. **The target label space never changes.** No class merging, no capping of evaluated targets.

### 2.2 The pipeline

```
raw file
  01 read          edatool loader -> raw text view (all strings)            code
  02 glance        types, missing, uniques, samples, pattern stats          code
  03 frame         target, task, metric, id and time columns                rules + LLM (flags override)
  04 split         presplit advice -> locked test (demo) + CV folds         code
  ---------------- training rows only from here --------------------------------
  05 audit         edatool: detectors, links, agent review                  code + LLM agent
  06 leak guard    structural + statistical + semantic evidence             code
  07 plan          translator default plan -> LLM cleaner plan              code + LLM
  08 compile/check plan -> sklearn Pipeline; data checks; edatool verify;   code + LLM repair
                   repair <= 3 per operation
  09 gate          CV: protected-only vs full plan; per-operation effects;  code + LLM refine
                   refine <= 2
  10 model         dummy, linear, LightGBM, CatBoost + Optuna under budget  code
                   (or AutoGluon backend on the prepared data)
  11 test          locked test set, once                                    code
  12 report        model.joblib, report.md, decision and LLM logs           code + optional LLM summary
```

LLM call sites: framer (1 call), edatool agent (a few sessions), cleaner (1 call + up to 2 schema retries), repair (up to 3 per failing operation), refine (up to 2), summary (1). All of them sit behind budgets and fallbacks.

### 2.3 Repository layout (final state)

```
./                              project root (this folder)
├── CLAUDE.md                   project instructions for every Claude Code session (Appendix F)
├── buildplan.md                this plan
├── progress.md                 task checklist and status (Appendix G)
├── README.md                   what it is, install, quick start, reproduce
├── pyproject.toml, uv.lock, .python-version, .gitignore, .env (ignored)
├── main.py                     CLI: run, predict, report (demo and benchmark use the same orchestrator)
├── paths.py                    every project path
├── settings.py                 loads configuration/settings.json + overrides into a Settings model
├── schemas.py                  shared Pydantic contracts (§2.4)
├── orchestrator.py             stages, budgets, fallbacks, checkpoints
├── configuration/
│   ├── settings.json           all knobs (§2.5)
│   └── metrics.json            primary and reported metrics per task (injected into prompts)
├── edatool/                    vendored auditor (from ../eda), improved in P4
├── agents/
│   ├── framer.py               target, task and id columns
│   ├── cleaner.py              plan, repair, refine
│   ├── freecoder.py            ablation: the LLM writes the cleaning code
│   └── summarizer.py           optional report paragraph
├── llms/
│   ├── client.py               ask(): structured output, retries, cache, logging, budget, spend
│   ├── providers.py            OpenAI provider and a fake provider for tests
│   └── agentmodel.py           smolagents model wrapper that logs and budgets edatool's agent calls
├── operations/
│   ├── base.py                 operation base class (pandas in, pandas out, same rows)
│   ├── registry.py             catalog: names, parameter models, examples, prompt text
│   ├── parsing.py              parse_number, parse_units, parse_datetime, parse_boolean
│   ├── values.py               replace_values, set_missing_where, clip_values, log_transform, impute_constant
│   ├── columns.py              drop_columns, extract_pattern, date_difference, as_category, add_missing_indicator
│   ├── categories.py           map_categories, group_rare
│   ├── text.py                 text_features
│   ├── rows.py                 training-only row filters
│   ├── target.py               allowed target operations
│   ├── custom.py               sandboxed custom operation
│   ├── translator.py           edatool issues -> default plan
│   ├── generic.py              typing step + generic stage
│   └── compiler.py             plan -> pipeline, row and target operations
├── checks/
│   ├── datachecks.py           deterministic checks of compiled operations
│   ├── leakage.py              leak guard decisions
│   └── gate.py                 validation gate
├── modelling/
│   ├── cv.py                   folds, quick model, cross-validated scoring with row filters
│   ├── engine.py               candidates + Optuna under a deadline + ensemble + refit
│   ├── backends.py             AutoGluon and FLAML backends (subprocess, own environments)
│   ├── fittedmodel.py          the saved artifact (pipeline + label encoder + metadata)
│   └── evaluator.py            locked test set, scores.json
├── tools/
│   ├── reader.py               read_table(), to_raw_view()
│   ├── glance.py               quick look
│   ├── splitter.py             split.json
│   ├── metrics.py              fixed metric functions
│   ├── sandbox.py              runs generated code safely
│   ├── runfolder.py            run folder, JSONL writers, context checkpoints, logging setup
│   └── report.py               report.md
├── prompts/                    framer.md, cleaner.md, repair.md, refine.md, customop.md, freecode.md, summary.md
├── benchmark/
│   ├── datasets.py             pools and loaders (OpenML cache, local files)
│   ├── corruptor.py            mess generator with ground truth
│   ├── leaks.py                leak generator (Group D)
│   ├── splits.py               fixed outer splits per dataset and seed
│   ├── materialize.py          writes train, test and truth files
│   ├── runner.py               grid runner (resumable, timeouts, workers)
│   ├── scoring.py              harness scoring and failure handling
│   ├── spend.py                global LLM spend ledger
│   ├── analysis.py             normalized scores, ranks, tests, confidence intervals
│   ├── projection.py           pilot projection: intervals, power simulation, held-out check (§4.10)
│   ├── figures.py              all figures and tables
│   ├── framereval.py           framer accuracy on OpenML (E7)
│   ├── showcase.py             demo: AutoGluon vs v2 vs v2 + AutoGluon on one file
│   ├── reproduce.py            one command: materialize, run a grid, analyse
│   ├── systems/                one adapter per system (dummy, lightgbm, skrub, flaml, autogluon, v2, v1, aide, tabpfn)
│   ├── envs/                   requirement files of the separate environments + setup script
│   ├── grids/                  smoke, dev and pilot-*.json (run); main, leaks, ablations, types, subsets, budgets .json (not run)
│   ├── legacy/v1/              frozen v1 code with its own pyproject and lock file
│   └── results/                committed: results CSVs, figures, tables, spend.json
├── edabench/                   edatool's own benchmark (from ../eda/benchmark)
├── tests/                      unit, torture, integration (fake LLM), fixtures
├── docs/                       decisions.md, deviations.md, preregistration.md, datasets.md, results.md, showcase.md, architecture.md
├── data/                       ignored: local datasets
├── cache/                      ignored: LLM cache, OpenML cache, edatool agent cache, benchmark files, workdirs
└── runs/                       ignored: one folder per run
```

### 2.4 Contracts (`schemas.py`)

Use edatool's `Task` literal (`"binary-classification"`, `"multi-class-classification"`, `"regression"`) everywhere. The CLI accepts the aliases `binary`, `multiclass` and `regression` and maps them once.

```python
from typing import Any, Literal
from pydantic import BaseModel
from edatool.schemas import Task

class TaskSpec(BaseModel):
    target: str
    task: Task
    metric: Literal["roc_auc", "log_loss", "rmse"]
    id_columns: list[str] = []
    time_column: str | None = None
    confidence: float = 1.0
    reason: str = ""
    source: Literal["flags", "rules", "llm"] = "rules"

class SplitInfo(BaseModel):
    strategy: Literal["random", "stratified", "group", "time"]
    key: str | None = None
    test_rows: list[int] = []
    folds: list[list[int]] = []
    reason: str = ""
    warnings: list[str] = []

class Step(BaseModel):
    op: str
    params: dict[str, Any]
    issues: list[str] = []
    reason: str = ""
    protected: bool = False
    source: Literal["leakguard", "translator", "llm", "repair", "refine", "custom"] = "translator"

class Plan(BaseModel):
    version: int = 0
    steps: list[Step] = []
    row_steps: list[Step] = []
    target_steps: list[Step] = []
    notes: str = ""

class Decision(BaseModel):
    stage: str
    kind: str
    subject: str
    proposal: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    accepted: bool
    reason: str
    source: Literal["rules", "llm", "edatool", "gate", "checks", "user"]

class StageStatus(BaseModel):
    name: str
    ok: bool
    seconds: float
    fallback: str | None = None
    error: str | None = None

class RunContext(BaseModel):
    run_id: str
    mode: Literal["demo", "benchmark"]
    variant: str
    data_path: str
    settings_hash: str
    task: TaskSpec | None = None
    split: SplitInfo | None = None
    audit_dir: str | None = None
    leaks: list[dict[str, Any]] = []
    plans: list[Plan] = []
    final_plan: Plan | None = None
    gate: dict[str, Any] = {}
    model: dict[str, Any] = {}
    scores: dict[str, Any] = {}
    stages: list[StageStatus] = []
    llm: dict[str, float] = {}
```

LLM answer models live next to their agents. Important: OpenAI strict structured outputs **do not accept free-form dictionaries** (`dict[str, X]` becomes `additionalProperties`, which strict mode rejects). So the cleaner returns each operation as `{"op": str, "params_json": str, "issues": [str], "reason": str}`. Code parses `params_json` and validates it with the operation's own Pydantic parameter model, and sends validation errors back to the LLM. Mappings inside parameters are lists of pairs (for example `[{"value": "?", "new": null}]`), never dictionaries. This also works with providers without strict schema support (semester 2, RQ6).

### 2.5 Settings (`configuration/settings.json`)

Load into a nested Pydantic `Settings` model. CLI overrides: `--settings file.json` (deep merge) and `--set gate.enabled=false` (dotted keys). Save the final settings and their hash in every run folder.

```json
{
  "seed": 0,
  "split": {"test_size": 0.2, "cv_folds": 5, "min_rows": 20},
  "budget": {
    "total_seconds": 300,
    "min_model_seconds": 60,
    "stage_seconds": {"audit": 90, "plan": 45, "compile": 45, "gate": 60, "report": 20}
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4.1-nano",
    "pilot_model": "gpt-4.1-mini",
    "temperature": 0,
    "seed": 0,
    "timeout_seconds": 60,
    "api_retries": 4,
    "schema_retries": 2,
    "cache": true,
    "max_calls_per_run": 40,
    "max_cost_per_run_usd": 0.1,
    "prices_per_million_tokens": {"gpt-4.1-mini": {"input": 0.4, "output": 1.6}, "gpt-4.1-nano": {"input": 0.1, "output": 0.4}}
  },
  "spend": {"category": "dev", "dev_cap_usd": 1.5, "experiments_cap_usd": 2.75, "stop_optional_usd": 4.0, "total_cap_usd": 5.0},
  "audit": {"use_agent": true, "sample_rows": 50000, "session_seconds": 60, "max_sessions": 8, "n_jobs": 4},
  "leakage": {"drop_semantic_only": true},
  "cleaner": {"enabled": true, "show_values": true, "max_values_per_column": 15, "max_repairs": 3, "max_refines": 2},
  "checks": {"max_parse_failure": 0.05, "max_column_growth": 10, "op_seconds": 30},
  "verify": {"enabled": true},
  "gate": {
    "enabled": true,
    "folds": 3,
    "max_rows": 20000,
    "per_op": true,
    "max_per_op_tests": 10,
    "tolerance": {"roc_auc": {"absolute": 0.002}, "log_loss": {"relative": 0.005}, "rmse": {"relative": 0.005}}
  },
  "engine": {"candidates": ["dummy", "linear", "lightgbm", "catboost"], "optuna": true, "ensemble": true, "n_jobs": 4, "patience_trials": 30},
  "backend": "own",
  "report": {"llm_summary": true}
}
```

Check the prices in P0.5 against the provider's pricing page and update them.

### 2.6 Run folder

```
runs/<yyyymmdd-hhmmss>-<dataset>-<6 hex>/
├── settings.json        final settings of this run (+ hash)
├── context.json         RunContext, saved after every stage
├── run.log              everything, UTF-8
├── glance.json          stage 02
├── task.json            stage 03
├── split.json           stage 04 (row positions, not CSV copies)
├── audit/               edatool report.json, issues.md, trace.jsonl, timings.json
├── leaks.json           stage 06
├── plans.json           every plan version: translator, llm, repaired, refined, final
├── checks.json          per-operation data checks
├── verify/              edatool verify report of the compiled plan
├── gate.json            CV scores per fold, per-operation effects, decision
├── decisions.jsonl      one line per decision (proposal, evidence, accepted, reason, source)
├── llm.jsonl            one line per LLM call (stage, prompt, hash, model, tokens, cost, seconds, cached, ok)
├── errors.jsonl         every caught exception with stage and traceback
├── custom/              sandboxed code and logs, if any
├── optuna.csv           model search trials
├── model.joblib         FittedModel (preparation + model + label encoder + metadata)
├── scores.json          CV and test scores, dummy scores, time per stage, LLM usage
├── predictions.csv      benchmark mode
├── prepared.parquet     benchmark mode: training frame after the agent operations (for repair accuracy)
└── report.md            human-readable report
```

### 2.7 Variants

These names are used in code, settings, grids, results and the report.

| Variant | Audit | Agent review | Operations | LLM cleaner | Gate | Engine |
|---|---|---|---|---|---|---|
| `generic` | no | no | none | no | no | own |
| `rules` | yes | no | translator plan | no | yes | own |
| `full` | yes | yes | translator + LLM plan | yes | yes | own |
| `full-noreview` | yes | no | translator + LLM plan | yes | yes | own |
| `full-nogate` | yes | yes | LLM plan, all accepted | yes | no | own |
| `full-norepair` | yes | yes | failing operations dropped at once | yes | yes | own |
| `full-nosamples` | yes | yes (no example values) | LLM sees names, types, stats only | yes | yes | own |
| `full-noleakguard` | yes | yes | leak guard off | yes | yes | own |
| `freecode` | yes | yes | LLM writes a cleaning class (sandbox) | yes | yes | own |
| `full+ag` | as `full` | | | | | AutoGluon on the prepared data |
| `rules+ag` | as `rules` | | | | | AutoGluon on the prepared data |

---

## 3. Component specifications

### 3.1 Read (`tools/reader.py`, stage 01)

- `read_table(path)` calls `edatool.load_raw(path)` and returns the raw text view and a read report (encoding, delimiter, skipped title or footer rows, bad lines, rows, columns).
- Hard stops (the only ones in the system), each with a clear message: unsupported extension, unreadable file, no data rows, fewer than `split.min_rows` labeled rows, a classification target with one class.
- `to_raw_view(frame)` turns a DataFrame into the same contract: `frame.astype(str).where(frame.notna(), "")`. Prediction accepts a path (preferred) or a DataFrame.
- Column names stay as in the file in the raw view. Safe names are made only at the generic stage output (§3.11).
- Missing target: drop those rows before anything else and log the count.
- Exact duplicate rows: in demo mode, drop them before the split (so copies cannot sit in train and test) and log the count. In benchmark mode, never (the harness owns the data).

### 3.2 Glance (`tools/glance.py`, stage 02)

For each column: name, inferred semantic type and confidence (edatool `infer_schema` on a sample of at most 50,000 rows), missing share (blank and hidden tokens), unique count, top 5 values with counts, 5 random distinct values, min and max for numbers, and **pattern stats**: share of plain numbers, numbers with a unit or suffix (with the top suffixes), currency, percent, date-like (with the top formats), boolean-like, values with two or more numbers inside (packed), long text. Implement the pattern stats once in `edatool/core/patterns.py` (P4.4) and use them in glance and in the new detectors.

Output `glance.json` and a compact text view for prompts (about 60 tokens per column). For tables with more than 60 columns, show the 40 most "interesting" columns (issues, non-numeric types, packed or unit patterns) in full and the rest as one line each.

### 3.3 Framer (`agents/framer.py`, stage 03)

- Flags `--target` and `--task` win. The benchmark always passes both.
- Rules rank target candidates: name hints (target, label, class, y, outcome, result, churn, price, survived, fraud, default, status, score, quality, ...; the list lives in settings), last-column bonus, never an id, constant, empty or long-text column. Rules also propose a task: 2 values → binary; non-numeric with 3 or more values → multi-class; numeric with more than 20 values or non-integer values → regression; integer with 3 to 20 levels → ambiguous (the LLM decides and gives a reason).
- The LLM (`prompts/framer.md`) sees the compact glance, the top 5 candidates and the rule proposal. It returns `{target, task, confidence, reason, id_columns, time_column}`. Invalid answers are retried with the error (up to 2 times), then the rules win.
- `--ask` (interactive only): when confidence is below 0.6, ask the user with `questionary`.
- Metric from the task, never from the LLM: binary → `roc_auc`, multi-class → `log_loss`, regression → `rmse` (as in AMLB and TabArena).

### 3.4 Split (`tools/splitter.py`, stage 04)

- `edatool.presplit(raw, target, task)` returns a `SplitAdvice` (group, time, stratified or random, with a key).
- Demo mode: hold out `test_size` by the advice: `GroupShuffleSplit` on the key, a time cut on the sorted time key, stratified if every class has at least 2 rows (else random with a warning).
- Benchmark mode: the harness gives only the training file; the system never splits off a test set.
- CV folds on the training rows: `StratifiedKFold` (folds = min(cv_folds, smallest class count), at least 2; if a class has fewer than 2 rows, `KFold` with a warning), `GroupKFold` for a group key, ordered folds for a time key, `KFold` for regression. Shuffle with the seed. Save positions to `split.json`.
- The group key and time key are used for splitting. An id-like group key is dropped from the features by the leak guard.

### 3.5 Audit (stage 05)

- `edatool.audit(train_raw, target, task, use_agent=..., model=<logged smolagents model>, settings={...}, run_dir=<run>/audit)`. Agent cache: `cache/edatool/` (shared between runs, so re-runs are free).
- In benchmark mode, ignore the task-type doubt issue F02 (the task is given) and log the decision.
- The stage budget caps the agent (per-session wall-clock limit added in P4.5).
- Fallbacks: agent error, time-out or budget → run `audit(..., use_agent=False)`. Deterministic audit crash (detectors already isolate their own crashes) → an empty report and a warning.
- Token usage of the agent goes into `llm.jsonl` and the spend ledger through `llms/agentmodel.py`.

### 3.6 Leak guard (`checks/leakage.py`, stage 06)

Evidence from the edatool report:

- **Structural**: confirmed id columns (E03), saved index columns (A02), framer id columns.
- **Statistical**: G01 (one feature predicts the target), G02 (pure categories), G04 (missingness predicts the target).
- **Semantic**: G03 (the agent's name review says "known only after the event") and agent confirmations of G01, G02 or G04 with a drop fix.

| Evidence | Action |
|---|---|
| Structural | Drop (protected step). |
| Semantic + statistical | Drop. Report "leak removed". |
| Semantic only | Drop if `leakage.drop_semantic_only`, else warn. Report. |
| Statistical only (open or dismissed by the agent) | Keep. Report "suspiciously strong, please check". |
| None | Keep. |

Without the agent (`rules`, `generic`) only structural drops and statistical warnings remain. The gate never decides about leaks: removing a leak lowers CV scores. Output `leaks.json` and one decision per column.

### 3.7 Planning (stage 07)

**Translator** (`operations/translator.py`, no LLM). Turns edatool issues into the default plan with Appendix C:

- Takes all `must_fix` issues and the `should_fix` codes listed in Appendix C. Confirmed issues are translated as Appendix C says. Review items that stayed `open` (no agent, or the session failed) are translated too, but always as gated steps. Dismissed and accepted issues are skipped.
- Never translates target-changing fixes (policy rule 12), `split_strategy` (the split stage handles it) or `no_action`.
- Fills missing parameters from the evidence (for example D01 negatives → `set_missing_where` with `below: 0`). If it cannot, it skips the fix and logs why.
- Labels steps: **protected** (leak guard drops, saved index drop, hidden missing tokens to missing, exact duplicate rows in training) or **gated** (everything else).

**Cleaner** (`agents/cleaner.py`, `prompts/cleaner.md`). Input: task, protected decisions, compact `issues.md`, the columns view (types, missing, value counts of up to 15 values, pattern stats), the default plan and the catalog text from `operations/registry.py`. Output: a full plan. Rules for the LLM are in Appendix A. Code validates the plan in order, tracking the columns that exist after each step (operations create and drop columns): unknown operation, bad parameters, unknown column, a step on the target, or an operation outside its allowed kind → errors sent back (up to 2 retries) → then the translator plan → then an empty plan. Code (not the LLM) sets `protected` on every step.

### 3.8 Operations (`operations/`)

Every operation is a scikit-learn transformer built on `operations/base.py`:

- `fit(frame, y=None)` learns state from training rows only (most operations learn nothing); `transform(frame)` returns a DataFrame with **the same index and number of rows**, works on unseen rows and unseen categories, keeps NaN, never touches columns it does not own, and fills `stats_` (values changed, parse failures with up to 10 examples, seconds).
- Parameters are a Pydantic model registered in `registry.py`, with a one-line purpose, a JSON example, the edatool actions it implements, and `kind` (`column`, `row`, `target`).
- Picklable (no lambdas), deterministic, no I/O.

The catalog (full parameters in Appendix B): `drop_columns`, `replace_values`, `set_missing_where`, `parse_number`, `parse_units`, `parse_datetime`, `date_difference`, `parse_boolean`, `map_categories`, `group_rare`, `as_category`, `extract_pattern`, `add_missing_indicator`, `impute_constant`, `clip_values`, `log_transform`, `text_features`, `custom`. Row operations (training rows only): `drop_duplicate_rows`, `drop_conflicting_rows`, `drop_rows_where` (condition in edatool's safe rule language, `edatool/agent/rules.py`, never `eval`). Target operations: `map_target_labels` (classification, normalizes spellings of existing classes; never merges classes) and `log_target` (regression, through `TransformedTargetRegressor`, scored on the original scale).

`registry.catalog_text()` renders the catalog for prompts. Prompts never list operations by hand (C4).

### 3.9 Compile, check, verify, repair (stage 08)

- `operations/compiler.py` builds `Pipeline([("agent", AgentSteps(steps)), ("typing", TypingStep()), ("generic", GenericStage(...))])` for the features (the model is added by the engine). Row steps become a `RowFilter` applied to training rows only. Target steps go into the label encoder or `TransformedTargetRegressor`.
- `checks/datachecks.py` runs on every CV fold of a sample (fit on the training part, transform both parts): parse failure share of each operation at most `checks.max_parse_failure` of its non-missing inputs; declared output types; no infinite values; no new all-missing column; same number of rows; unseen rows transform without error; time per operation at most `checks.op_seconds`; column growth at most `checks.max_column_growth`; no feature equals the target or is named like the target.
- **edatool verify**: after the checks, fit the agent steps on all training rows and call `edatool.verify(report, [(step name, frame), ...], step_issues)` on the output **before** the typing and generic steps. Add the target column back to every snapshot frame (verify needs it). `step_issues` maps each step name to the issue ids in its `issues` list. Ignore P01 and P02 at that point (they are expected before the generic stage). Feed back `still_open` must-fix issues and new must-fix issues created by a step. Run verify on the first compiled plan and on the final plan only (it reruns all detectors).
- **Repair** (`prompts/repair.md`): a failing operation gets a structured error, for example `parse_units on Weight: 4.1% of values could not be parsed, examples: "1.3 kgs", "?"`, plus the column's value counts. The LLM returns a corrected operation or `drop` with a reason. At most `cleaner.max_repairs` attempts per operation, then the operation is dropped. Every attempt is a decision.

### 3.10 Validation gate (`checks/gate.py`, stage 09)

- Quick model: LightGBM with fixed parameters (300 trees, learning rate 0.05, 31 leaves, `n_jobs` from settings, `verbose=-1`), behind the same typing and generic stages. Folds: `gate.folds` folds from the split (same strategy), on a stratified sample of at most `gate.max_rows` training rows.
- Row filters apply to the training part of each fold only. **Validation parts are never filtered**, or the score would be inflated.
- P0 = protected steps only. P1 = protected + all gated steps. Accept P1 if its mean CV score is not worse than P0 within the tolerance of the metric (`gate.tolerance`).
- Per-operation effects (if `gate.per_op` and at most `gate.max_per_op_tests` gated steps): for each gated step, score P1 without that step; effect = score(P1) − score(P1 without step), with the sign flipped for loss metrics so that positive means "helps". Remove steps that hurt by more than the tolerance, then score the reduced plan once more. The final plan is the better of P0 and the reduced P1.
- **Refine** (`prompts/refine.md`): if a step was rejected or the full plan did not beat P0, show the LLM the per-fold scores and per-step effects; it may revise the plan (up to `cleaner.max_refines` rounds). Every revision is compiled, checked and gated. Keep the best CV score.
- Leak removals are protected and never gated. Output `gate.json` and one decision per gated step (effect, accepted or rejected).

### 3.11 Typing step and generic stage (`operations/generic.py`)

- **TypingStep** mimics `pandas.read_csv` defaults on string columns that no operation produced: the pandas default NA tokens (`""`, `"NA"`, `"N/A"`, `"NaN"`, `"null"`, `"None"`, `"nan"`, `"n/a"`, `"#N/A"`, `"<NA>"` and the other pandas defaults; **not** `"?"`) become missing; a column whose non-missing values all parse as numbers becomes numeric; other columns stay strings. `fit` decides the type of each column on training rows; `transform` applies the same decision and turns unparseable values in numeric columns into NaN. This makes `generic` equal to "what a normal AutoML pipeline sees after `pd.read_csv`".
- **GenericStage**: use `skrub.TableVectorizer` if it works with the installed scikit-learn: numbers pass through; dates become parts (year, month, day, weekday, total seconds); low-cardinality strings (at most 30 levels) become one-hot for linear models or ordinal codes for tree models; high-cardinality strings become `StringEncoder` features (about 30 components). Linear models also get a median imputer and a standard scaler. If skrub causes trouble, build the same with a `ColumnTransformer` and record it.
- Output feature names are sanitized for LightGBM (no `[ ] { } " : ,` and no duplicates), with a name map saved in the model.

### 3.12 Model engine (`modelling/engine.py`, stage 10)

- Candidates: dummy (always; prior or mean), linear (logistic regression or ridge), LightGBM, CatBoost (`allow_writing_files=False`, or it creates `catboost_info/` folders).
- Optuna TPE (seeded) on the CV folds within the remaining budget. Time shares: LightGBM 50%, CatBoost 35%, linear 5%, ensemble and refit 10%. Stop a family early after `engine.patience_trials` trials without improvement. Search spaces (LightGBM): leaves 15 to 255 (log), learning rate 0.01 to 0.2 (log), minimum child samples 5 to 100, subsample 0.5 to 1, column sample 0.5 to 1, L2 0.001 to 10 (log), trees up to 2,000 with early stopping on a 10% inner split. CatBoost: depth 4 to 8, learning rate 0.02 to 0.2, L2 1 to 10, up to 2,000 iterations with early stopping. Linear: C or alpha 0.001 to 100 (log).
- Keep out-of-fold predictions. If the average of the best two families beats the best single family on out-of-fold predictions, use the average.
- Refit on all training rows (row filters applied). Trial log → `optuna.csv`.
- Any candidate that crashes or times out is skipped. Optuna errors → default parameters. The dummy model always exists.
- **Backends** (`modelling/backends.py`): `autogluon` and `flaml` run in their own environments as subprocesses on the prepared data (agent steps + typing, written as parquet), with the remaining budget, so that "v2 preparation + AutoGluon" has the same total budget as AutoGluon alone. If a backend fails, fall back to the own engine and log it.

### 3.13 Metrics and evaluator (`tools/metrics.py`, `modelling/evaluator.py`, stage 11)

- Probabilities for ROC-AUC, PR-AUC and log-loss. Binary: ROC-AUC (primary), log-loss, PR-AUC, accuracy, balanced accuracy, F1 of the positive class. Multi-class: log-loss (primary, `labels=` all training classes, probabilities clipped at 1e-15), ROC-AUC one-vs-rest macro, accuracy, balanced accuracy, macro F1. Regression: RMSE (primary), MAE, R2, MAPE on non-zero targets only.
- Labels: a label encoder fitted on training labels. Probabilities are always aligned to the full class list; a class missing from a training fold gets a column of zeros before renormalizing with a small epsilon (this prevents the common multi-class crash).
- Full precision in files; round only for display.
- The locked test set is used once, after the final refit. Nothing may change the model afterwards.

### 3.14 Artifact and report (`modelling/fittedmodel.py`, `tools/report.py`, stage 12)

- `FittedModel`: the feature pipeline, the model, the row-filter and target-step records, the label encoder, the raw input columns, the task, the metric, versions and the settings hash. `predict(path_or_frame)` and `predict_proba(...)` accept raw input, add missing input columns as empty and ignore extra columns (with warnings). Save with joblib. The free-code variant stores its code and predicts through the sandbox.
- `report.md` from a template: task and split; read report; top audit issues; leak decisions; plan steps accepted and rejected with their CV effects and reasons; repairs; CV and test scores next to the dummy; time and LLM cost per stage; warnings and fallbacks. An optional LLM summary paragraph (`prompts/summary.md`, at most 120 words, only facts given in the prompt); if it fails, the template is enough.

### 3.15 LLM layer (`llms/`)

- `client.ask(prompt_name, inputs, schema, stage, ctx)`: renders `prompts/<name>.md` (placeholders like `{catalog}` filled from code), calls the provider, validates against the Pydantic `schema`, and returns the object or `None`.
- Model, temperature and seed come from settings (C3). If the provider rejects a parameter (some newer models do not accept `temperature`), retry once without it and log it.
- API errors (rate limits, connection errors, 5xx, time-outs) are retried with exponential backoff (`tenacity`, `llm.api_retries`). Schema failures are retried with the validation error (`llm.schema_retries`).
- Disk cache in `cache/llm/`, keyed by a hash of model, prompt text and rendered inputs. A cache hit costs nothing and is logged as cached.
- Every call is one line in `llm.jsonl`: stage, prompt name, prompt hash, model, input and output tokens, cost, seconds, cached, ok, error.
- Budgets: per run (`max_calls_per_run`, `max_cost_per_run_usd`) and global (`benchmark/spend.py` ledger in `benchmark/results/spend.json`, split into `dev` and `experiments`). The benchmark runner sets `spend.category` to `experiments`; everything else counts as `dev`. When a category cap or `total_cap_usd` is reached, `ask` returns `None` and the fallbacks take over. At `stop_optional_usd` (total), write a warning in `progress.md` and follow the money rules of §0.5. Before each real call, the client estimates its cost from the prompt length; if that estimate would cross `total_cap_usd`, it does not make the call.
- `providers.py`: `OpenAIProvider` (chat completions with structured outputs) and `FakeProvider` (answers from fixtures; used in every test).
- `agentmodel.py`: a subclass of smolagents' `OpenAIModel` whose `generate` logs each call (tokens, cost) into the run's `llm.jsonl` and the spend ledger, and stops the session when the budget is gone. Pass it to `edatool.audit(model=...)`.

### 3.16 Orchestrator, budgets and fallbacks (`orchestrator.py`)

- `run(data_path, settings, target=None, task=None, mode="demo", variant="full")` returns the `RunContext` and saves `FittedModel`. Each stage is `stage(ctx, data, settings)` wrapped by `run_stage(name, function, fallback)`: it times the stage, catches exceptions, writes the traceback to `errors.jsonl` and `run.log`, applies the fallback, appends a `StageStatus` and saves `context.json`.
- A `Budget` object holds the run deadline and the stage limits; stages ask it for their remaining time. The model stage gets everything left, at least `budget.min_model_seconds`.
- Fallbacks:

| Stage | Typical failure | Fallback |
|---|---|---|
| 01 read | encoding, delimiter, layout | try alternatives; else hard stop with a clear message |
| 02 glance | strange column | skip that column's stats, log |
| 03 frame | LLM error, invalid answer, low confidence | rules; `--ask` asks the user |
| 04 split | tiny class; group or time split impossible | non-stratified or random split + warning |
| 05 audit | agent error, time-out, budget | deterministic audit |
| 06 leak guard | missing evidence | structural only |
| 07 plan | invalid JSON, unknown operation, budget | retry with errors (2×) → translator plan → empty plan |
| 08 compile | an operation fails checks | repair (3×) → drop the operation |
| 08 verify | verify crashes | skip verify, log |
| 09 gate | CV error | P0 (protected steps only) |
| 10 model | a candidate crashes or times out | skip it; dummy always exists |
| 10 backend | AutoGluon or FLAML subprocess fails | own engine |
| 11 test | metric undefined (for example one class in test) | NaN with a reason + secondary metrics |
| 12 report | LLM summary fails | template only |
| any | LLM budget spent | agents return `None`; rules take over |
| any | unexpected exception | log; use the stage fallback; continue from the last good context |

### 3.17 CLI (`main.py`)

```
uv run main.py run data/file.csv [--target T] [--task binary|multiclass|regression] [--variant full]
                                 [--budget 300] [--seed 0] [--backend own|autogluon] [--ask]
                                 [--out runs] [--settings file.json] [--set key=value ...]
uv run main.py predict runs/<run>/model.joblib data/new.csv --out predictions.csv
uv run main.py report runs/<run>
```

Without arguments, `main.py` starts the interactive mode (questionary + rich, like v1). Console output is one short line per stage, ASCII-safe, with full details in `run.log`.

### 3.18 Sandbox (`tools/sandbox.py`)

Used by `custom` operations and by the `freecode` variant.

- Before running: an AST check. Allowed imports: `pandas`, `numpy`, `re`, `math`, `datetime`, `string`, `collections`, `itertools`, `functools`, `statistics`, `unicodedata`. Forbidden: `os`, `sys`, `subprocess`, `socket`, `shutil`, `pathlib`, `importlib`, `ctypes`, `pickle`, `marshal`, `requests`, `urllib`, `http`, and the names `open`, `exec`, `eval`, `compile`, `__import__`, `input`, `globals`, `locals`, `breakpoint`.
- Runs `python` in a new temporary folder with a minimal environment (`SYSTEMROOT`, `PATH`, `TEMP`, `TMP`, `PYTHONIOENCODING=utf-8`, `PYTHONHASHSEED=0`); **no API keys**. Data goes in and out as parquet. Time limit per call (60 s). On a time-out, kill the whole process tree (`taskkill /T /F /PID <pid>` on Windows).
- Generated code must define one class with `fit(frame)` and `transform(frame)`, keep the row count and index, and never touch the target.
- This is a safety net, not a security boundary; write that in `README.md` under "Limitations" (D2).

---

## 4. Benchmark and experiments

### 4.1 Dataset pools (Appendix D has the IDs)

- **Dev pool** (build and tune on these only): the 4 local datasets and 2 TabArena datasets. Never report them as main results.
- **Group A, clean base** (main results): 15 TabArena-v0.1 datasets with readable column names, 1,000 to 54,000 rows (two large ones are sampled down to a fixed 30,000 rows), 6 binary, 3 multi-class, 6 regression.
- **Group B, controlled mess**: Group A + the corruptor (§4.2) at levels 0 to 3.
- **Group C, natural mess**: 5 datasets that are messy in their original form. Reported separately.
- **Group D, leakage set**: 10 Group A datasets with one realistic post-event leak column each (Appendix E2).

For every dataset, write a data card in `docs/datasets.md`: source, OpenML IDs, license, rows, columns, task, target, why it was chosen, known mess. Check every license; drop a dataset you cannot use and pick a spare.

### 4.2 Corruptor (`benchmark/corruptor.py`)

- Start from the clean raw text view (reuse `edabench` loading and `column_as_text`), the target, the task, a seed and a level (0 = clean, 1 = low, 2 = medium, 3 = high). Reuse and extend the injector code from `edabench/injector.py`.
- Mess types and level parameters: Appendix E1. The mixed-mess levels of Group B use types 1 to 9 and 12. Type 10 (file layout) is a separate reading test (E6b). Type 11 (leakage) is Group D only.
- Rules: never corrupt the target; corrupt at most 80% of the features; never destroy information except where the type requires it (fake missing values, impossible values); same corruption process for training and test rows (corrupt the full table with the seed, then split with the fixed indices).
- Ground truth `truth.json` per file: type, columns, affected rows, parameters, and the original values of corrupted cells (parquet next to it) for repair accuracy.
- Tests: determinism (same seed → same bytes), level monotonicity (more cells at higher levels), recoverability (the inverse of each corruption restores the original values), target untouched.

### 4.3 Splits and files (`benchmark/splits.py`, `benchmark/materialize.py`)

- One fixed outer split per dataset and seed: 80/20, stratified for classification, saved as indices.
- Files: `cache/benchmark/files/<group>/<dataset>/level<k>/seed<s>/train.csv`, `test_features.csv`, `test_labels.csv` (harness only), `truth.json`. Write all values as text (`na_rep=""`), UTF-8 without BOM, except where the file-layout type says otherwise.
- Group D also writes `test_features_naive.csv` (leak values kept), to show how good AutoML looks before deployment.

### 4.4 Systems and environments

| Tier | System | Environment | Notes |
|---|---|---|---|
| must | `dummy` | main | class prior or mean |
| must | `lightgbm` | main | `pd.read_csv` defaults, object columns → `category`, default parameters |
| must | `skrub` | main | `skrub.tabular_pipeline(...)` with `HistGradientBoosting` (older skrub: `tabular_learner`) |
| must | `flaml` | `benchmark/envs/flaml` | `AutoML.fit(..., time_budget, metric, seed, n_jobs=4)` |
| must | `autogluon` | `benchmark/envs/autogluon` | `TabularPredictor(label, problem_type, eval_metric).fit(time_limit, presets="medium_quality", num_cpus=4)`; if Ray fails on Windows, use sequential fold fitting |
| must | v2 variants (§2.7) | main | `python -m benchmark.systems.v2 --variant ...` |
| later | `v1` | `benchmark/legacy/v1/.venv` (Python 3.10) | frozen DS LAB 1 code; adapter only if time allows (P2.4); **never run in this plan** (about 60 LLM calls per run) |
| later | `aide` | `benchmark/envs/aide` | `aideml`; adapter only if time allows; **never run in this plan** |
| could | `tabpfn` | `benchmark/envs/tabpfn` | free; at most 10,000 training rows; skip if the weights need a gated download; not in the pilot |
| could | AutoGluon Assistant, CAAFE | — | semester 2 |

Adapter protocol: `python <adapter> --train <csv> --test <csv>[,<csv>] --target <name> --task <task> --metric <metric> --budget <seconds> --seed <n> --workdir <dir> --out <dir>`. The adapter writes `predictions*.csv` (classification: one probability column per class label plus `label`; regression: `prediction`) and `meta.json` (seconds, ok, error, LLM calls, tokens, cost). `benchmark/systems/registry.json` maps each system to its Python executable and adapter. Every system reads the CSV files itself, as a user would.

Heavy baselines live in separate `uv` environments (Python 3.12 x86_64, except v1), so their pinned versions cannot break the main environment. If one fails to install on Windows, try WSL Ubuntu; if that also fails within the timebox, mark it skipped and write the reason in `docs/decisions.md` and `README.md`.

### 4.5 Protocol (fairness rules)

- Every system gets the same train file, test features, target name, task type, metric and time budget (120 s wall clock in the pilot; 300 s in the budget check and in the full grids), on the same machine, with 4 threads (`n_jobs=4`, `OMP_NUM_THREADS=4`) and at most 2 runs in parallel.
- The runner's hard time-out is budget × 1.25 + 60 s; on a time-out, kill the process tree.
- No system sees test labels; the harness scores.
- Label keys: the harness normalizes class labels to strings, and integer-valued numbers to `str(int(value))` (so `"1"`, `"1.0"` and `1` match), for predictions and truth alike.
- A failure (no file, wrong shape, NaN predictions, time-out, crash) gets the dummy model's score and is counted as a failure (as in AMLB).
- Every LLM system uses the same model. The LLM cache is allowed (it only replays identical calls).
- Delete system workdirs after scoring (AutoGluon models are large); keep predictions and `meta.json`. Keep workdir paths short (`cache/w/<8 hex>`) because of the Windows 260-character path limit.

### 4.6 Grids and compute

The runner (`benchmark/runner.py`) reads a grid JSON (systems × datasets × levels × seeds × budget), skips cells already in the results file, appends one line per cell to `benchmark/results/<grid>.csv`, and stops LLM cells when the experiments spend cap is reached.

Results columns: `grid, system, variant, dataset, group, level, mess_types, seed, budget_s, status, error, metric, score, dummy_score, secondary (JSON), seconds, llm_calls, input_tokens, output_tokens, cost_usd, run_dir, git_hash, settings_hash, started_at`.

**Grids that run in this plan** (the pilot is about 18% of the runs of the full benchmark below; LLM model `gpt-4.1-mini`):

| Grid | Content | Runs | Paid LLM runs | Computer time (2 workers) | LLM cost |
|---|---|---|---|---|---|
| `smoke` | 2 dev datasets × levels 0, 3 × seed 0 × all installed systems, budget 60 s, `full` with `gpt-4.1-nano` | ~40 | 4 | < 1 h | < $0.05 |
| `pilot-main` (E1, E2) | 8 Group A datasets (Appendix D, marked P) × levels 0, 2, 3 × seed 0 × `dummy`, `lightgbm`, `skrub`, `flaml`, `autogluon`, `generic`, `rules`, `full`, `full+ag` | 216 | 24 (`full+ag` reuses the cached answers of `full`) | ~4 h | ~$1.0 |
| `pilot-seed` | the same 8 datasets × levels 0, 3 × seed 1 × free systems only (`dummy`, `lightgbm`, `skrub`, `autogluon`, `generic`, `rules`) | 96 | 0 | ~1.5 h | $0 |
| `pilot-leaks` (E3) | 5 Group D datasets × seed 0 × `full`, `full-noleakguard`, `rules`, `autogluon`, `lightgbm` | 25 | 10 | ~0.5 h | ~$0.4 |
| `pilot-ablations` (E5) | 4 of the pilot datasets × level 3 × seed 0 × `freecode`, `full-nogate`, `full-norepair`, `full-noreview`, `full-nosamples` (`generic`, `rules`, `full` come from `pilot-main`) | 20 | 20 | ~0.5 h | ~$0.8 |
| `pilot-types` (E6) | repair accuracy of the cleaning stage only (no model training) on all 15 Group A datasets × 12 types at level 3, with `rules` (free) and `full` on 4 datasets × 4 key types (units, mixed units, packed, spellings) × `full`, `generic`, `autogluon` | ~50 trained + cleaning-only checks | 16 | ~1 h | ~$0.4 |
| `pilot-natural` (E10) | 3 Group C datasets × seed 0 × `full`, `generic`, `autogluon`, `lightgbm` | 12 | 3 | ~0.3 h | ~$0.1 |
| `pilot-budget` | 2 pilot datasets × level 3 × 300 s × `full`, `autogluon` (LLM answers cached) | 4 | 0 | ~0.3 h | $0 |
| **Total** | | **~460** | **~77** | **~8 h** | **~$2.7** |

If a pilot `full` run costs more than $0.06 on average after the first 10 runs, reduce `pilot-ablations` to 2 datasets and `pilot-types` to 2 datasets, and log it in `docs/deviations.md`. Never cross the experiments cap.

**Full grids** (write the grid files and test them with the `smoke` grid, but **do not run them**; they exist so the user can run them later on a stronger machine or with a larger budget):

| Grid | Content | Rough time and cost |
|---|---|---|
| `main` (E1, E2) | 15 Group A datasets × levels 0, 1, 2, 3 × seeds 0, 1, 2 × `dummy`, `lightgbm`, `skrub`, `flaml`, `autogluon`, `generic`, `full`, `full+ag`, budget 300 s | ≈ 38 h with 2 workers; about $10 |
| `leaks` (E3) | 10 Group D datasets × seeds 0, 1, 2 × `full`, `full-noleakguard`, `rules`, `autogluon`, `flaml`, `skrub`, `lightgbm` | about 12 h; about $2 |
| `ablations` (E5) | 8 Group A datasets × levels 0, 3 × seeds 0, 1 × the 9 v2 variants | about 15 h; about $8 |
| `types` (E6) | 8 Group A datasets × each applicable single mess type at level 3 × seed 0 × `full`, `generic`, `autogluon`, `skrub` | about 8 h; about $4 |
| `subsets` (E4) | 6 Group A datasets × levels 0, 3 × seed 0 × `v1`, `aide` | about 6 h; about $8 |
| `budgets` (E9) | 5 Group A datasets × level 2 × seed 0 × budgets 60, 300, 900 s × `full`, `autogluon`, `flaml` | about 6 h; about $1 |

**Priority** inside the pilot if time or money runs short: `pilot-main` → `pilot-leaks` → `pilot-ablations` (`freecode` and `full-nogate` first) → `pilot-types` → `pilot-seed` → `pilot-natural` → `pilot-budget`.

### 4.7 Metrics and statistics (`benchmark/analysis.py`)

- Per cell: primary metric on the test set, secondary metrics, seconds, success, LLM calls, tokens, dollars.
- **Normalized score** per (dataset, level, seed): (score − dummy) / (best − dummy), where best is the best score of any system in that cell; for log-loss and RMSE flip the sign first. Average over seeds, then over datasets.
- **Average rank** per level; **critical-difference diagram** (Friedman test + Nemenyi post-hoc) per level, with `autorank` or your own implementation of Demšar (2006).
- **Wilcoxon signed-rank tests** of `full` against each baseline, per level and pooled over levels 2 and 3, with Holm correction. Win, tie and loss counts. Bootstrap 95% confidence intervals of the mean normalized-score difference.
- **Decision quality** (research data from `decisions.jsonl`): framer accuracy; leak detection precision, recall and F1 per leak strength; false leak drops on clean data; repair accuracy per mess type (share of corrupted cells restored to the true value: equal within 1e-6 relative for numbers, equal labels for categories, NaN recall and false-NaN rate for fake missing values); share of LLM steps accepted by the gate; repair-loop success; fallback counts per stage; failure taxonomy.
- **Reliability and cost table**: success rate, median seconds, mean tokens, mean dollars, standard deviation over seeds, per system.

### 4.8 Experiments and research questions

| ID | Experiment | Grid in this plan (full grid for later) | Answers |
|---|---|---|---|
| E1 | Messy benchmark: score vs mess level | `pilot-main` + `pilot-seed` (`main`) | RQ1, RQ2 |
| E2 | Plug-in: `full+ag` vs `autogluon` | `pilot-main` (`main`) | contribution 4 |
| E3 | Leakage set: deployment score (leak masked), naive score, detection precision and recall | `pilot-leaks` (`leaks`) | RQ3 |
| E4 | Reliability and cost (v1 and AIDE only in the full grid) | all pilot grids (`subsets`) | RQ4 |
| E5 | Ablations | `pilot-ablations` (`ablations`) | RQ5 |
| E6 | Per mess type (repair accuracy, gain over `generic`); E6b reading robustness (type 10, free systems only) | `pilot-types` (`types`) | RQ1 depth |
| E7 | Framer accuracy, rules vs rules + LLM, on 40 OpenML datasets from TabArena and OpenML-CC18 (≤ 100,000 rows; about $0.05) | `framereval.py` | stage evaluation |
| E8 | Audit quality: edabench before/after P4, vs ydata-profiling, cleanlab, old eda.py (deterministic, $0) | `edabench` | contribution 5 |
| E9 | Budget sensitivity | `pilot-budget` (`budgets`) | robustness of conclusions |
| E10 | Natural mess (Group C) | `pilot-natural` | external validity |

Figures: (1) normalized score vs mess level per system, with 95% intervals; (2) CD diagram at level 3 (and level 0 in the appendix); (3) gain over `generic` per mess type; (4) leak experiment: naive vs deployment score; (5) ablation bar chart; (6) cost vs score scatter. Tables: reliability and cost; Wilcoxon results; per-dataset scores (appendix); framer accuracy; audit benchmark.

### 4.9 Pre-registration

Before the pilot grids, write `docs/preregistration.md`: research questions and hypotheses (copy §1.1), datasets and pools (including which datasets are in the pilot and how they were chosen), systems and versions, budgets, seeds, metrics, analysis plan (§4.7), the projection method (§4.10), what counts as support for each hypothesis in the full benchmark (for example "H1 supported if `full` beats `autogluon` in a Holm-corrected Wilcoxon test at levels 2 and 3 pooled, p < 0.05"), the held-out cells of §4.10, and exclusion rules. Commit it and tag `prereg`. Every later change goes to `docs/deviations.md`.

### 4.10 Pilot projection (`benchmark/projection.py`)

The pilot does not give final results. It estimates what the full benchmark would show, with honest uncertainty, so the user can decide whether to run it. The estimate must be more than a guess:

1. **Choose the pilot datasets before seeing any result**: 8 of the 15 Group A datasets (Appendix D, marked P), with all three task types and small and large sizes. Record the choice in the pre-registration.
2. **Effect estimates with intervals.** For each comparison (`full` vs `autogluon`, `skrub`, `lightgbm`, `generic`; `full+ag` vs `autogluon`), compute the mean normalized-score difference per level, with bootstrap 95% intervals over datasets (10,000 resamples). An interval that crosses 0 means "not known yet".
3. **Variance from the free seed.** Use the seed-1 runs of the free systems to estimate how much scores move between seeds. Add that variance to the `full` estimates (which have only one seed), and say so.
4. **Power estimate.** Simulate the full grid (15 datasets, 3 seeds): draw dataset effects from the pilot's estimated mean and spread, add the seed noise, run the pre-registered Wilcoxon test with Holm correction, and repeat 2,000 times. Report the share of simulations with a significant result (the power) for each hypothesis.
5. **Held-out check.** Before running them, write predictions (with intervals) for 4 held-out pilot cells into `docs/preregistration.md`: 2 datasets × level 2 for `full` and `autogluon`. Run them last. Report how many fell inside their predicted intervals. If fewer than 3 of 4 did, the projection report says the projection is unreliable.
6. **Short budget check.** Compare the ranking of `full` and `autogluon` at 120 s and 300 s (`pilot-budget`). If the order changes, the report warns that the 120 s results may not carry over.
7. **Mechanism check.** Link the score gains to repair accuracy per mess type (`pilot-types`). A gain that appears where the repairs worked is explained. A gain without repairs is flagged as suspicious.
8. **Output**: `docs/projection.md`, generated by code. For each RQ: the estimated effect and its interval, the power for the full benchmark, the result of the held-out and budget checks, and a recommendation in one line: "run the full benchmark", "run a reduced benchmark (which grids)" or "not worth running yet (why)". Also the estimated time and cost of each full grid, from the measured pilot runs.

---

## 5. Testing

- `uv run pytest -q`: the fast suite (no network, no real LLM, under 3 minutes). Markers: `slow` (big files, full pipeline on dev data) and `llm` (real API, opt-in, costs cents). Configure `pythonpath = ["."]` and the markers in `pyproject.toml`.
- **Unit tests**: every operation (fit and transform on a training frame; unseen rows; unseen categories; NaN; empty strings; unicode; all-missing column; one row; index and row count kept; pickle round trip); translator mappings; compiler column tracking; data checks; metrics against scikit-learn (including a class missing from a fold); splitter (stratify fallback, groups never overlap, time order); framer rules; leak decision table; LLM client with the fake provider (retries, schema retry, cache hit, budget stop, cost); typing step (pandas parity); label-key normalization; corruptor determinism and recoverability.
- **Gate tests** with synthetic data: a helpful operation (parsing a number with units that drives the target) is accepted; a harmful one (dropping the strongest feature, or a row filter that removes one class) is rejected; validation folds are never filtered.
- **Integration tests**: the whole pipeline on small dev samples (at most 2,000 rows) with the fake provider and fixture answers, for `generic`, `rules`, `full` and `freecode`; each run under 90 s; run folders complete; `FittedModel` reloads and predicts.
- **Torture tests** (`tests/torture.py` writes the files at test time): one column; one data row; header only; all values missing; target with one class; a class with one row; unicode headers (Azerbaijani, Cyrillic, emoji); duplicate column names; semicolon CSV with decimal commas; tab and pipe delimiters; Excel with 3 title rows and a footer note; 500 columns; 1,000,000 rows (`slow`); UTF-8 BOM; cp1252 file with "€"; quoted fields with newlines; numbers and words in one column; only constant features; only date features; only long-text features; a high-cardinality id; numbers with units and packed values; `inf` values; column names with `[ ] { } " :`; JSON and parquet input. Each must finish with a model and a report, or stop with the clear message of §3.1.
- **edatool regression**: the deterministic edabench results after P4 must keep recall and precision within 1 point of the P4.1 baseline.
- **Fault injection**: monkeypatch the provider to raise, an operation to raise, Optuna to raise, a candidate model to raise, verify to raise; every run still ends with a model and a report.
- **Reproducibility**: same seed and warm LLM cache → identical plan and identical scores.

---

## 6. Phases and tasks

Each task: what to do → done when. Commit after each task.

### P0. Setup and safety net

- **P0.1 Read the inputs** of §0.2. → You can explain the v2 pipeline, edatool's stages and fix vocabulary, and the v1 problems B1 to B8 and C1 to C5.
- **P0.2 Git.** If this folder has no `.git`, run `git init` (do not copy `../I/.git`; the old history stays there). Add to `.gitignore`: `runs/`, `cache/`, `*.parquet`, `catboost_info/`, `AutogluonModels/`. Keep the existing ignores (`data/`, `models/`, `sandbox/`, `architecture/`, `.env`, `.venv/`). Note that these folder patterns match at every depth: `.venv/` also covers the baseline environments, and no code or test fixture folder may be called `data`, `models` or `sandbox`. Commit everything as `P0.2: v1 snapshot (DS LAB 1)` and tag `v1-dslab1`. → `git log` shows the snapshot; `git status` is clean; `git check-ignore .env` confirms `.env` is ignored.
- **P0.3 Memory files.** Create `CLAUDE.md` (Appendix F), `progress.md` (Appendix G, with every task of this section), `docs/decisions.md`, `docs/deviations.md`. → Files committed.
- **P0.4 Freeze v1.** Copy `agents/`, `llms/`, `tools/`, `configuration/`, `prompts/`, `main.py`, `main2.py`, `paths.py`, `schemas.py`, `pyproject.toml`, `uv.lock` and `.python-version` into `benchmark/legacy/v1/` unchanged (no `.env`, no data files, no `__pycache__`). Git does not keep empty folders and `.gitignore` ignores `data/`, `models/` and `sandbox/`, so the headless runner (P2.4) creates those three folders in each temporary workdir. → Copy committed; `diff` with the root shows no differences.
- **P0.5 Check the LLM.** With a throwaway script in your scratchpad (not committed), make one tiny structured-output call with `gpt-4.1-nano` and one with `gpt-4.1-mini`, temperature 0 and seed 0 (cost < $0.001 together). Check the current prices of both on OpenAI's pricing page and update `prices_per_million_tokens`. Start the spend ledger with these calls. → Model availability, accepted parameters and prices recorded in `docs/decisions.md`.

### P1. Foundation

- **P1.1 Environment.** New `pyproject.toml`: `requires-python = ">=3.12,<3.13"`, no build system (`[tool.uv] package = false`), dependencies: pandas (`<3`, edatool needs object-dtype raw views), numpy, scipy, scikit-learn, lightgbm, catboost, optuna, pydantic, openai, python-dotenv, tenacity, joblib, pyarrow, openpyxl, rich, questionary, tabulate, `smolagents[openai]==1.26.0`, skrub, matplotlib, autorank (drop it if it conflicts; §4.7 allows your own implementation), openml; dev group: pytest, ruff, psutil. `.python-version` = `cpython-3.12-windows-x86_64-none`. Run `uv python install cpython-3.12-windows-x86_64-none`, then `uv sync`. If `uv` does not accept the key in `.python-version`, create the environment with `uv venv --python cpython-3.12-windows-x86_64-none` and sync again. → `uv run python -c "import platform, sys; print(platform.machine(), sys.version)"` prints `AMD64` and 3.12; all packages import; versions recorded in `docs/decisions.md`.
- **P1.2 Remove v1 from the root** (it is frozen in P0.4): `agents/*.py`, `llms/*.py`, `tools/*.py`, `main2.py`, `clean.py`, `configuration/constants.json`, `configuration/modeloptions.md`, `configuration/edaconfig.py`, all files in `prompts/`, the empty `models/` and `sandbox/` folders, `agentic_automl.egg-info/`. Keep `main.py`, `paths.py`, `schemas.py` and `configuration/metrics.json` for rewriting. → Root contains no v1 logic; commit message lists the removed files.
- **P1.3 Core files.** `paths.py` (every path from §2.3), `configuration/settings.json` (§2.5), `settings.py` (load, deep merge, dotted overrides, hash), `configuration/metrics.json`, `schemas.py` (§2.4). → Unit tests for settings loading and overrides pass.
- **P1.4 Run folder, logging, metrics.** `tools/runfolder.py` (create folder, JSONL writers, context checkpoint, UTF-8 logging setup with ASCII-safe console), `tools/metrics.py` (§3.13). → Tests pass, including the multi-class missing-class case.
- **P1.5 LLM layer.** `llms/client.py`, `llms/providers.py`, `benchmark/spend.py` (§3.15). `llms/agentmodel.py` waits until P7.2. → Fake-provider tests pass; one `llm`-marked test with the real API (`gpt-4.1-nano`) passes (cost < $0.002); a test proves that the client refuses a call that would cross `total_cap_usd`.
- **P1.6 Orchestrator and CLI skeleton.** `orchestrator.py` with `run_stage`, `Budget`, checkpoints and stub stages; `main.py` with the commands of §3.17. → `uv run main.py run data/ibmchurn.csv --target Churn --variant generic` creates a run folder with `context.json` and `run.log`, even though stages are stubs.

### P2. Baselines first (installation problems are the most common delay)

- **P2.1 Harness skeleton.** `benchmark/systems/` protocol and `registry.json`, `benchmark/runner.py` (resumable, time-outs, process-tree kill, workers), `benchmark/scoring.py` (label keys, dummy fallback), `benchmark/splits.py` and a first `materialize.py` for the 4 local datasets (level 0, seeds 0 to 2; creditcardfraud as a fixed stratified sample of 50,000 rows). → Files exist; the runner scores a hand-made predictions file correctly.
- **P2.2 Main-environment baselines**: `dummy`, `lightgbm`, `skrub`. → Each scores on the 4 local datasets.
- **P2.3 Separate environments**: `flaml` and `autogluon` (`benchmark/envs/<name>.txt` + setup script; timebox 2 h each; WSL fallback). → Each scores on the 4 local datasets with a 60 s budget, or is marked `[!]` with the reason.
- **P2.4 v1 adapter (optional, never run).** Do this task only if P2.1 to P2.3 went smoothly; otherwise mark it skipped. Create `benchmark/legacy/v1/.venv` with its own lock (Python 3.10 x86_64). Write `benchmark/legacy/v1/headless.py` (its own file; the frozen code stays unchanged): copy the frozen folder to a temporary workdir per run, put the train file in its `data/`, skip `TargetDetector` and the task inference (use the given target and task), run the remaining v1 steps in order, then load `models/preprocessor.pkl` (cloudpickle) and `models/model.onnx`, transform the test features and predict (use the ONNX probability output when it exists; otherwise labels, which is v1's real behaviour, B4). Hard time-out 20 minutes. The API key comes from the environment, never from a copied `.env`. **Do not run v1 with the real LLM** (it makes about 60 calls per run). → The adapter is tested with a fake `OPENAI_BASE_URL` that refuses connections: it must fail cleanly and report the failure.
- **P2.5 Optional baseline**: `tabpfn` (free; timebox 1 h). Do not install AIDE in this plan. → Installed and smoke-tested, or marked skipped with the reason.
- **P2.6 Baseline smoke table.** 4 local datasets × seed 0 × every installed baseline, budget 60 s → `benchmark/results/p2-smoke.csv` and a short table in `progress.md`.

### P3. v2 core without LLM (variant `generic`)

- **P3.1 Vendor edatool.** Copy `../eda/edatool` to `edatool/` and `../eda/benchmark` to `edabench/`. Fix imports (`benchmark.` → `edabench.`), move edabench paths into the root `paths.py` (data from `data/`, runs in `cache/edabench/`, results in `edabench/results/`), remove the `EDATOOL_DATA_DIR` dependency. Add a short `edatool/README.md` (copied, adapted). → `uv run python -m edatool audit data/ibmchurn.csv --target Churn --task binary-classification --no-agent --out cache/edatool-runs` works for the 4 local datasets.
- **P3.2 Reader and glance** (§3.1, §3.2; the pattern stats can start inside `tools/glance.py` and move to edatool in P4.4). → Tests pass; glance of the 4 local datasets looks right (TotalCharges shows a blank token; `date` in kinghousedata is date-like).
- **P3.3 Splitter** (§3.4). → Tests pass; kinghousedata gets a group split on `id`.
- **P3.4 Typing step and generic stage** (§3.11). → Pandas-parity tests pass; LightGBM accepts the output of every torture header.
- **P3.5 Engine** (§3.12, own engine only). → Respects the deadline within 10%; tests pass.
- **P3.6 Evaluator, FittedModel, report, predict** (§3.13, §3.14). → `uv run main.py run ... --variant generic` produces a complete run folder for the 4 local datasets; `predict` works on the test features.
- **P3.7 v2 adapter** (`benchmark/systems/v2.py` with `--variant`). → `generic` is within a few points of `lightgbm` and `skrub` on the 4 local datasets (sanity check); table in `progress.md`.

### P4. edatool improvements

- **P4.1 Baseline numbers before changing anything**: run edabench `evaluate types`, `evaluate benchmark` (deterministic) and `experiments baselines` → `docs/edatool-baseline.md`. (The previous measurement in `../eda/architecture.html`: recall 0.986, precision 0.896 on 358 injected issues.) The comparison tools (ydata-profiling, cleanlab) go into a separate environment `benchmark/envs/edabench` if they conflict with the main environment. No LLM calls in this task.
- **P4.2 Windows-safe output.** Library mode must never crash on a cp1252 console: ASCII fallbacks for `✓ ✗ · → └─` when the stream encoding is not UTF-8, and `errors="replace"` handlers. → A test with a cp1252 stream passes.
- **P4.3 Loader robustness**: encodings (BOM via `utf-8-sig`, UTF-16 by BOM, then UTF-8, cp1252, latin-1); delimiter fallback (`, ; \t |`, choose the most consistent field count above 1) when the sniffer fails; title rows above the header and notes below the table (find the header as the first row with the modal field count and mostly non-numeric text; drop rows after the table whose field count differs or that have one non-empty cell) for CSV and Excel; report everything in `attrs` and in A01 evidence; clear errors for empty and header-only files. → Torture files for these cases load correctly.
- **P4.4 Semantic-mess detectors** (`edatool/core/patterns.py` + detectors):
  - **B07 numbers with units or suffixes** on categorical and text columns: at least 90% of non-missing values match number + unit (`1.37kg`, `8GB`, `1,200 sqft`, `3.2k`); evidence: unit counts and examples; fix `convert_type {"to": "numeric", "units": [...]}`; needs review when there are 2 or more units (the agent must give conversion factors).
  - **B08 packed values**: replace digits by `#` and count templates; if the top 3 templates cover at least 80% and have at least 2 numeric slots (`#x#`, `# bd / # ba`); evidence: templates and examples; fix `extract_pattern` (new action in the vocabulary).
  - **B09 mixed date formats**: two or more date patterns in one column, with a day/month ambiguity check; fix `convert_type {"to": "datetime", "formats": [...]}`; needs review if ambiguous.
  - Add more hidden-missing tokens to `tokens.json` (`n.a.`, `not available`, `--`, `#N/A`, `nil`).
  - Inject the fix vocabulary into the agent prompts from `FixAction` instead of listing it by hand.
  → Unit tests per detector; the 4 local datasets give no new false alarms.
- **P4.5 Policy and cost.** Catalog defaults: F05 (rare classes) and F07 (target outliers) become `no_action` with a reason ("the target label space never changes"). Anonymized column names (`^(V|X|f|col|feature)_?\d+$`) skip agent review of D02 and E09 (accepted with "no meaning available"). Group review items of the same code into batched sessions (at most 8 items each) and cap the number of sessions (`audit.max_sessions`). Add a per-session wall-clock limit to the guard. Make the G01 `n_jobs` a setting. Put the agent's token usage into `report.meta`. → On a fixed 5,000-row stratified sample of creditcardfraud, with `gpt-4.1-nano`, the agent uses at most 1/3 of the tokens it used before the change (measure both; total cost < $0.05); tests pass.
- **P4.6 Benchmark the auditor again.** Extend `edabench/injector.py` with units, mixed units, packed values and mixed date formats (and their accepted codes in `evaluate.py`). Re-run P4.1. → Old kinds: recall and precision within 1 point of P4.1; new kinds: recall ≥ 0.9; table in `docs/edatool-baseline.md` (before/after).

### P5. Toolbox and verification (variant `rules`)

- **P5.1 Operations**: `base.py`, `registry.py` and all column operations of Appendix B. → Unit tests per operation pass.
- **P5.2 Row and target operations.** → Tests: row filters never run at prediction time; the target label space is unchanged.
- **P5.3 Translator** (Appendix C). → On the audit reports of the 4 local datasets the default plans look right (churn: TotalCharges blank → missing (protected), parse TotalCharges, drop customerID (protected), drop duplicate rows (protected, training only); kinghousedata: parse `date`, `zipcode` as category, `id` dropped and used as the group key, `waterfront` drop gated; wine: no target class merging).
- **P5.4 Compiler, data checks, verify integration** (§3.9). → Tests pass; a plan with a deliberately wrong unit map fails the checks with a structured error.
- **P5.5 Leak guard** (§3.6). → Decision-table tests pass.
- **P5.6 Gate** without refine (§3.10). → Synthetic helpful and harmful tests pass.
- **P5.7 Variant `rules` end to end** on the 4 local datasets. → Complete run folders; `rules` ≥ `generic` on ibmchurn and kinghousedata, or a written explanation.

### P6. Benchmark data

- **P6.1 Dataset registry** (`benchmark/datasets.py`): pools of Appendix D, OpenML cache in `cache/openml/`, a manifest with dataset versions and file hashes, readable-name check, licenses. Download everything. → `docs/datasets.md` has a data card per dataset; spares replace unusable datasets.
- **P6.2 Corruptor** (§4.2, Appendix E1). → Tests pass; a printed sample of each type at each level looks realistic (paste one example per type into `docs/datasets.md`).
- **P6.3 Leak generator** (Appendix E2). → Tests pass; G01 fires for strong leaks.
- **P6.4 Materialize the dev pool** (levels 0 to 3, seeds 0 to 2). → Files and `truth.json` exist.
- **P6.5 Group C**: check that each Group C dataset (Appendix D) is really messy in the version you downloaded; if the TabArena version was cleaned, use the original OpenML version or a spare. Document the mess per dataset. → Section "Natural mess" in `docs/datasets.md`.

### P7. LLM agents (variant `full`)

- **P7.1 Framer** (§3.3) with fake-provider tests and `--ask`. → Tests pass; a real-LLM check with `gpt-4.1-nano` on the 4 local datasets picks the right targets.
- **P7.2 edatool agent integration** (`llms/agentmodel.py`, §3.5). → Agent calls appear in `llm.jsonl` and the spend ledger; the fallback works when the provider raises.
- **P7.3 Cleaner** (§3.7, Appendix A). → Fake-provider tests pass; a real run with `gpt-4.1-nano` on a level-3 dev file (at most 5,000 rows) produces a valid plan with parse operations.
- **P7.4 Repair loop.** → A fixture with a wrong unit map gets repaired.
- **P7.5 Refine loop.** → A fixture plan with a harmful step gets revised.
- **P7.6 Sandbox and `custom` operation** (§3.18). → AST-check tests and time-out tests pass; no API key reaches the subprocess (test).
- **P7.7 Free-code variant** (`agents/freecoder.py`, `prompts/freecode.md`). → Runs end to end on 2 dev datasets (`gpt-4.1-nano`, at most 5,000 rows); success and failure are logged.
- **P7.8 Summary paragraph.** → Appears in `report.md`; disappears gracefully when the provider fails.
- **P7.9 Dev iteration** (dev pool only, samples of at most 5,000 rows, levels 0 and 3, seed 0). Use `gpt-4.1-nano` and at most 40 paid `full` runs; read the decision logs of the worst cells, improve prompts, operations and policies, and re-run (unchanged prompts come from the cache for free). Find most problems with the free `rules` variant first. Then run one final check with `gpt-4.1-mini` on 6 dev cells. Stop when: success 100%; mean cost per `full` run with `gpt-4.1-mini` ≤ $0.05; `full` ≥ `rules` ≥ `generic` on the messy dev files, or the reasons are understood and written down; or the development cap ($1.50) is reached. → `docs/decisions.md` has the iteration notes and the measured cost per run.

### P8. Robustness and performance

- **P8.1 Torture suite** (§5). → All files finish or stop with the clear message.
- **P8.2 Fault injection** (§5). → All tests pass.
- **P8.3 Performance.** creditcardfraud (full 284,807 rows) and the 1,000,000-row torture file finish inside a 300 s budget with the free `rules` variant (and `full` only with cached LLM answers); peak memory stays under 6 GB (measure with `psutil`). → Numbers in `progress.md`.
- **P8.4 Reproducibility test.** → Passes.
- **P8.5 Fresh-clone test**: `git clone` this repo into a temporary folder, `uv sync`, `uv run pytest -q`. → Passes without the original working copy.

### P9. Pilot experiments (no full benchmark)

- **P9.1 Materialize** Group A (levels 0 to 3, seeds 0 to 2, for later use), the single-type files for `pilot-types`, Group C and Group D. → Files exist; a manifest lists them with hashes.
- **P9.2 Pre-registration** (§4.9), including the pilot datasets and the predictions for the 4 held-out cells (§4.10). → `docs/preregistration.md` committed and tagged `prereg`.
- **P9.3 Smoke grid and freeze.** Run `smoke` (it also checks that every full grid file loads and that its first cell starts), fix what breaks (dev data only), then tag `v2-freeze`. → All systems produce scores on the smoke grid.
- **P9.4 E7 framer accuracy** (`benchmark/framereval.py`, 40 datasets, about $0.05). → `benchmark/results/framer.csv`.
- **P9.5 `pilot-main`** in the background (2 workers), in chunks of at most 2 hours (the runner resumes). Check the log and the spend after the first 10 `full` runs and apply the reduction rule of §4.6 if needed. Leave the 4 held-out cells for the end. → `benchmark/results/pilot-main.csv`.
- **P9.6 `pilot-leaks`.** → `pilot-leaks.csv`.
- **P9.7 `pilot-ablations`.** → `pilot-ablations.csv`.
- **P9.8 `pilot-types`** and E6b reading robustness (free systems only). → `pilot-types.csv`, `reading.csv`.
- **P9.9 `pilot-seed`.** → `pilot-seed.csv`.
- **P9.10 `pilot-natural`.** → `pilot-natural.csv`.
- **P9.11 `pilot-budget`.** → `pilot-budget.csv`.
- **P9.12 E8 audit benchmark (final, deterministic, $0).** → edabench results copied into `benchmark/results/`.
- **P9.13 Held-out cells** (last). → Their results are next to the predictions of P9.2.

Never start the full grids of §4.6. They are for the user to run later.

### P10. Analysis

- **P10.1 `benchmark/analysis.py`** (§4.7) with tests on a small synthetic results file. → Tests pass.
- **P10.2 `benchmark/figures.py`**: all figures (PDF and PNG) to `benchmark/results/figures/`, tables (`.csv` and `.md`) to `benchmark/results/tables/`. → One command regenerates everything.
- **P10.3 `docs/results.md`** (generated by code, not written by hand): the pilot tables per RQ, clearly labeled as pilot results (small sample). No paper text.
- **P10.4 `benchmark/projection.py` and `docs/projection.md`** (§4.10): effect intervals, seed noise, power simulation, held-out check, budget check, mechanism check, recommendation, and the estimated time and cost of each full grid. → Tests on a synthetic results file pass; `docs/projection.md` is generated from the real pilot files.

### P11. Documentation and demo (no report, no paper)

- **P11.1 Code quality pass**: read every module once more; remove dead code and duplication; make names and the one-line comments clear; check the style rules (§0.6). → `uv run pytest -q` passes; no unused code (`ruff check --select F` clean).
- **P11.2 Documentation**: `README.md` (what it is, install, quick start, CLI, variants, how to reproduce: `uv run python -m benchmark.reproduce --grid smoke`, `--grid pilot` and, for later, `--grid main` and the other full grids with their estimated time and cost from `docs/projection.md`; folder map), `docs/architecture.md` (one page, links to this plan), `docs/showcase.md`. → A new reader can run a demo in 10 minutes.
- **P11.3 Demo.** `benchmark/showcase.py` runs `autogluon` on the raw file, `full`, and `full+ag` on one messy file (a level-3 pilot file with units or packed columns, so the LLM answers come from the cache and the demo costs $0) with the same budget, prints the three scores, and prints the main decisions from the run folder ("Weight parsed from text to kg", "customer_id dropped (id)", "refund_issued dropped (leak)"). → Output pasted into `docs/showcase.md` from a real run.

### P12. Final quality check and handover

- **P12.1 Checks**: full test suite (including `slow`); `uv run python -m benchmark.reproduce --grid smoke`; style check script (one empty line between definitions, imports sorted by length, file names, no empty line at the end of files); secret scan (`git grep -n "sk-"` finds nothing); `git status` clean. → All pass.
- **P12.2 Handover**: final summary in `progress.md` and as your final message: what was built, the pilot results (from `docs/results.md`), the projection and its recommendation (from `docs/projection.md`), the total LLM spend, what failed or was skipped and why, limitations, and the plan for the thesis semester. List the files created, edited and deleted in the last phase. Simple English.

### Stretch goals (only after P12, in this order)

- S1. Feature ideas in the CAAFE style: the LLM proposes safe expressions (ratios, differences, date gaps, group statistics) written in the rule language; each one is gated.
- S2. LLM swap (RQ6): a smaller OpenAI model and one open model through an OpenAI-compatible endpoint.
- S3. Only if the user asks: run the full grids of §4.6, then scale up to 30 Group A datasets and 5 seeds.
- S4. AutoGluon Assistant (MLZero) baseline.

---

## 7. Definition of done

- [ ] v2 finishes on 100% of benchmark files and torture files, or stops with the clear message of §3.1.
- [ ] No LLM-written training or tuning code. The test set is used once. All fitting happens inside pipelines on training rows.
- [ ] Every run has its own folder with settings, decisions, LLM calls, cost, scores and a report.
- [ ] edatool is vendored, improved (P4) and re-benchmarked (before/after table).
- [ ] Benchmark code complete for the full design (15 Group A datasets, 12 mess types, 4 levels, Group C, Group D); full grid files written and smoke-tested, not run.
- [ ] Pilot grids ran (§4.6), including all must-have baselines (or their failure is documented).
- [ ] `docs/projection.md` generated, with intervals, power, held-out check and a recommendation.
- [ ] Results: normalized scores, ranks, CD diagram, Wilcoxon tests against AutoGluon (Holm), reliability and cost table, score-vs-mess figure, plug-in result, leak experiment, at least 3 ablations (`generic`, `rules`, `freecode` or `full-nogate`), framer accuracy.
- [ ] Pre-registration committed before the pilot grids; deviations logged.
- [ ] No report or paper was written; results exist as generated files in `benchmark/results/` and `docs/results.md`.
- [ ] README with one reproduction command; a working demo; clean, readable, tested code.
- [ ] Tests pass; style rules hold; no secrets in git; total LLM spend about $4 and never above $5 (`benchmark/results/spend.json`).

---

## 8. Risks and gotchas

| Risk or gotcha | What to do |
|---|---|
| A baseline does not install on Windows ARM (emulated x86_64) | Separate `uv` environment; WSL Ubuntu fallback; then skip and document |
| AutoGluon uses too much memory or Ray fails | `presets="medium_quality"`, `num_cpus=4`, sequential fold fitting, one AutoGluon run at a time |
| The OpenAI model or its parameters change | Model and prices in settings; the client retries without unsupported parameters |
| Strict structured outputs reject dictionaries | `params_json` strings and lists of pairs (§2.4) |
| LightGBM rejects feature names with special characters | Sanitize names in the generic stage (§3.11) |
| Multi-class log-loss crashes when a class is missing from a fold | `labels=` + aligned, padded probabilities (§3.13) |
| CatBoost writes `catboost_info/` everywhere | `allow_writing_files=False` |
| Console `UnicodeEncodeError` (cp1252) | Reconfigure stdout and stderr to UTF-8; ASCII fallbacks (P4.2) |
| Windows subprocesses and joblib need `spawn` | `if __name__ == "__main__":` in every entry point |
| Time-outs leave child processes | Kill the process tree (`taskkill /T /F /PID`) |
| Windows 260-character path limit (AutoGluon model folders) | Short workdirs `cache/w/<8 hex>` |
| `ruff format` or `black` breaks the user's style | Never run them; `ruff check --select F` only |
| pandas 3 changes string dtypes and breaks the raw view | Keep `pandas<3` in the main environment |
| edatool agent cost explodes on wide or anonymized tables | P4.5: anonymized-name skip, batched sessions, session caps, wall-clock caps |
| The gate overfits the CV folds | Few decisions, a tolerance, the final score only from the locked test set |
| The method gets tuned to the benchmark | Dev pool vs Group A separation, pre-registration, freeze tag |
| The LLM memorized famous datasets | Prefer less famous datasets; mess changes the surface; discuss it as a limitation |
| AutoGluon already handles some mess types | Report it honestly; the per-type analysis (E6) shows where meaning is needed |
| Context loss during a long autonomous run | `progress.md`, `docs/decisions.md`, project `CLAUDE.md`, small commits |
| The pilot costs more than estimated | Measure after 10 `full` runs; reduction rule and priority order in §4.6; hard cap $5 in the client |

---

## Appendix A. Prompt skeletons

Keep prompts short. Lists (operations, fix vocabulary, metrics) are injected from code. Every prompt file is hashed; the hash is logged with every call.

**`prompts/framer.md`**

```
You frame a supervised learning task for one table. You see a compact profile of every column and a
ranked list of target candidates from rules.
Return the target (one name from the candidate list), the task (binary-classification,
multi-class-classification or regression), a confidence from 0 to 1, a one-sentence reason, the id
columns (they identify rows) and the time column (it orders rows in time) or null.
Rules:
- Choose the column a business user would most likely want to predict. Follow the rule ranking unless
  the meaning of the columns clearly says otherwise.
- Two distinct target values: binary-classification. An ordered integer score with few levels: decide
  by meaning (a category to predict: multi-class; a measured amount: regression) and say why.
- Never invent column names.
Profile:
{profile}
Rule candidates:
{candidates}
Rule task proposal: {rule_task}
```

**`prompts/cleaner.md`**

```
You plan data-preparation operations for one table before an AutoML engine trains a model. Code has
already measured the data (issues below). Code will verify every operation you propose: it compiles
them into a scikit-learn pipeline that is fitted on training folds only, checks the output, and keeps
an operation only if cross-validation shows that it does not hurt.
Your job: choose operations from the catalog and fill in their parameters.
Rules:
- Use only catalog operations. params_json must be a JSON object that matches the operation.
- Give every operation a reason and the issue ids it addresses ([] for a column-level reason).
- Never touch the target column `{target}`, except with the target operations of the catalog. Never
  merge or remove target classes.
- Ids and leaks are already decided by code (see "Already decided"). Do not repeat them.
- Read the values before deciding: units, currencies, dates, spellings, packed values, special codes.
  Prefer operations that keep information (parse, extract, map) over dropping columns.
- Do not add imputation or encoding for ordinary columns: a generic stage does that after your
  operations. Use impute_constant only when the meaning gives the value (for example 0 when tenure is 0).
- Start from the default plan. Keep, change or remove its operations, and add missing ones.
Task: {task}
Already decided by code: {protected}
Audit issues: {issues}
Columns: {columns}
Default plan: {default_plan}
Operation catalog: {catalog}
```

**`prompts/repair.md`**

```
One operation failed the data checks. Fix its parameters, replace it by another catalog operation, or
drop it.
Operation: {operation}
Error: {error}
Examples of failed raw values: {examples}
Value counts of the column: {value_counts}
Catalog entries: {catalog_entries}
Return the corrected operation, or action "drop" with a reason.
```

**`prompts/refine.md`**

```
The validation gate measured your plan with cross-validation ({direction}).
Score with protected operations only (P0): {baseline}
Score of the full plan, per fold: {full}
Effect of each operation (score with it minus score without it; positive helps): {effects}
Rejected operations: {rejected}
Revise the plan: remove operations that hurt, fix parameters that may be wrong, or add an operation
that the evidence supports. Do not repeat a rejected operation unchanged. Return the full plan.
```

**`prompts/customop.md`** and **`prompts/freecode.md`**

```
Write one Python class named {class_name} with fit(self, frame) and transform(self, frame).
frame is a pandas DataFrame of raw text values (strings, "" means empty) without the target column.
transform must return a DataFrame with the same index and the same number of rows.
Allowed imports: {allowed_imports}. No files, no network, no printing, no randomness.
{task_specific_instructions}
Return only the code.
```

**`prompts/summary.md`**

```
Write 4 to 6 plain sentences for a data scientist about this run. Use only the facts below. Do not
add numbers that are not in the facts.
Facts: {facts}
```

---

## Appendix B. Operation catalog

Mappings are lists of pairs. Every operation also has `reason` and `issues` (set by the plan step, not in the parameters).

| Operation | Kind | Parameters | Learns in fit | Example |
|---|---|---|---|---|
| `drop_columns` | column | `columns: [str]` | no | customerID |
| `replace_values` | column | `column`, `items: [{value: str, new: str or null}]` | no | `" "` → missing; `"0"` → missing for `yr_renovated` |
| `set_missing_where` | column | `column`, `below: float or null`, `above: float or null`, `values: [str]` | no | negative counts → missing; bedrooms above 20 → missing |
| `parse_number` | column | `column`, `remove: [str]`, `decimal: "." or ","`, `thousands: str or null`, `percent: bool`, `suffixes: [{suffix: str, factor: float}]` | no | `"$1,250.00"` → 1250.0; `"3.2k"` → 3200 |
| `parse_units` | column | `column`, `units: [{unit: str, factor: float}]`, `default_factor: float or null` | no | `"1.37kg"` → 1.37; `"800 g"` → 0.8 |
| `parse_datetime` | column | `column`, `formats: [str]` (or `["mixed"]`), `dayfirst: bool`, `parts: [year, month, day, weekday, hour, epoch_days]`, `keep_original: bool` | no | `"20141013T000000"` → 2014, 10, 13, 0, ... |
| `date_difference` | column | `start`, `end`, `unit: days or years`, `name` | no | sale year − `yr_built` → house age |
| `parse_boolean` | column | `column`, `true_values: [str]`, `false_values: [str]` | no | `"Y"`, `"yes"`, `"TRUE"` → 1 |
| `map_categories` | column | `column`, `items: [{value: str, canonical: str}]`, `case_fold: bool`, `strip: bool` | no | `"NY"`, `"new york"` → `"New York"` |
| `group_rare` | column | `column`, `min_count: int`, `label: str` | yes (frequent levels) | levels under 10 rows → `"Other"` |
| `as_category` | column | `column` | no | `zipcode` numbers → category |
| `extract_pattern` | column | `column`, `pattern: regex with groups`, `into: [str]`, `types: [number or text]`, `keep_original: bool` | no | `"2560x1600"` → 2560, 1600 |
| `add_missing_indicator` | column | `columns: [str]` | no | `x_missing` = 1 where `x` is missing |
| `impute_constant` | column | `column`, `value: str or float` | no | TotalCharges missing → 0 (tenure 0) |
| `clip_values` | column | `column`, `lower_quantile: float or null`, `upper_quantile: float or null` | yes (quantiles) | feature winsorizing (never the target) |
| `log_transform` | column | `column`, `offset: float` | no | `sqft_lot` → log1p |
| `text_features` | column | `column`, `max_features: int`, `components: int` | yes (TF-IDF + SVD) | description → 10 numeric columns |
| `custom` | column | `columns: [str]`, `into: [str]`, `instructions: str` (the code is generated by `customop.md`) | sandbox | `"128GB SSD + 1TB HDD"` → ssd_gb, hdd_gb |
| `drop_duplicate_rows` | row | none | — | exact copies in training rows |
| `drop_conflicting_rows` | row | none | — | same features, different target |
| `drop_rows_where` | row | `condition` (edatool rule language) | — | `` `sqft_living` > 13000 `` |
| `map_target_labels` | target | `items: [{value: str, canonical: str}]` (spellings of existing classes only) | — | `"yes "` → `"Yes"` |
| `log_target` | target | `offset: float` | — | log1p(price), scored on the original scale |

Operation order rules (checked by the validator): a step may only use columns that exist at that point; `drop_columns` cannot be followed by steps on the dropped columns; operations on the target column are rejected unless their kind is `target`.

---

## Appendix C. edatool fix to operation translation

| edatool code(s) | edatool fix | Default step | Protected? |
|---|---|---|---|
| A02 | `drop_column` | `drop_columns` | yes |
| E03 (and framer id columns) | `set_role {"role": "id"}` | `drop_columns` (decided by the leak guard; a group key is used for splitting first) | yes |
| G01–G04 | `drop_column` | decided by the leak guard (§3.6) | yes |
| C02 | `replace_values {mapping}` | `replace_values` (tokens → missing) | yes |
| A04 | `drop_rows {"condition": "duplicated"}` | `drop_duplicate_rows` (training rows) | yes |
| F01 | `drop_rows` | rows with a missing target are removed at read time | — |
| B01 | `convert_type {"to": "numeric", "cleaning": [...]}` | `parse_number {remove: cleaning}` | gated |
| B07 (new) | `convert_type {"to": "numeric", "units": [...]}` | `parse_units` (factors from the agent or the LLM; if unknown, skip and let the cleaner decide) | gated |
| B02, B09 (new) | `convert_type {"to": "datetime", ...}` | `parse_datetime {parts: [year, month, day, weekday]}` | gated |
| B08 (new) | `extract_pattern` | `extract_pattern` | gated |
| B03 | `convert_type` | none by default (the cleaner decides) | — |
| B04 | `encode {"method": "onehot"}` | `as_category` | gated |
| B05 | `normalize_labels {mapping}` | `map_categories` | gated |
| B06 | `normalize_labels` | `parse_boolean` | gated |
| C03 | `replace_values {mapping}` | `replace_values` (sentinel → missing) + `add_missing_indicator` | gated |
| C04, C05 | `add_indicator` | `add_missing_indicator` | gated |
| C01 | `impute` | none (the generic stage imputes) | — |
| C06 | `drop_column` | `drop_columns` | gated |
| D01 | `replace_values` (no mapping) | `set_missing_where` from the evidence (for example `below: 0`) | gated |
| D02 | `replace_values` (no mapping) | `set_missing_where {above or below}` from the flagged values; skipped for anonymized names | gated |
| D03 | `replace_values` | `replace_values` (placeholder dates → missing) | gated |
| D04, A05, A06 | `drop_rows` | `drop_rows_where`, `drop_conflicting_rows` (training rows only) | gated |
| D05, E04 | `group_rare` | `group_rare {min_count}` | gated |
| E01 | `drop_column` | `drop_columns` | gated |
| E02, E06, E07, E08 | `drop_column` | `drop_columns` (the gate decides; `waterfront` must survive if it helps) | gated |
| E05 | `drop_column` | `text_features` | gated |
| E09 (consider) | `log_transform` | none by default (no effect on trees; the cleaner may add it) | — |
| F02, F04, H01–H05 | `no_action`, `split_strategy` | handled by the framer or the split stage | — |
| F03 | `normalize_labels` | `map_target_labels` | yes |
| F05, F06, F07 | `group_rare`, `log_transform`, `cap_values` (target) | not translated (rule 12); `log_target` only if the cleaner proposes it, gated | — |
| A07 | `no_action` (rename) | none (names are sanitized at the generic stage) | — |

---

## Appendix D. Dataset pools

OpenML dataset IDs (did) and task IDs (tid) of TabArena-v0.1 (Erickson et al., NeurIPS 2025 Datasets and Benchmarks; Appendix B.2 of arXiv:2506.16791). Use `openml.datasets.get_dataset(did)` and the dataset's `default_target_attribute`. Verify every ID, the target and the column names after download; if anything differs, fix the registry and log it.

**Dev pool**

| Dataset | Source | Task | Rows | Notes |
|---|---|---|---|---|
| ibmchurn | `data/ibmchurn.csv` | binary (`Churn`) | 7,043 | blank TotalCharges, customerID |
| kinghousedata | `data/kinghousedata.csv` | regression (`price`) | 21,613 | date text, `yr_renovated = 0`, repeated `id` |
| winequality | `data/winequality.csv` | multi-class (`quality`) | 1,599 | red wine only; 240 duplicates |
| creditcardfraud | `data/creditcardfraud.csv` | binary (`Class`) | fixed stratified 50,000-row sample | anonymized "no harm" control |
| houses | did 46934, tid 363678 | regression | 20,640 | California housing |
| Is-this-a-good-customer | did 46938, tid 363682 | binary | 1,723 | |

**Group A (main, clean base)**

| Dataset | did | tid | Task | Rows | Features | Pilot |
|---|---|---|---|---|---|---|
| credit-g | 46918 | 363626 | binary | 1,000 | 21 | P |
| Fitness_Club | 46927 | 363671 | binary | 1,500 | 7 |  |
| Marketing_Campaign | 46940 | 363684 | binary | 2,240 | 26 |  |
| Bank_Customer_Churn | 46911 | 363619 | binary | 10,000 | 11 | P |
| E-CommereShippingData | 46924 | 363632 | binary | 10,999 | 11 | P |
| online_shoppers_intention | 46947 | 363691 | binary | 12,330 | 18 |  |
| maternal_health_risk | 46941 | 363685 | multi-class | 1,014 | 7 | P |
| students_dropout_and_academic_success | 46960 | 363704 | multi-class | 4,424 | 37 | P |
| SDSS17 | 46955 | 363699 | multi-class | 78,053 → fixed stratified 30,000 | 12 |  |
| concrete_compressive_strength | 46917 | 363625 | regression | 1,030 | 9 | P |
| healthcare_insurance_expenses | 46931 | 363675 | regression | 1,338 | 7 |  |
| airfoil_self_noise | 46904 | 363612 | regression | 1,503 | 6 |  |
| Another-Dataset-on-used-Fiat-500 | 46907 | 363615 | regression | 1,538 | 8 | P |
| miami_housing | 46942 | 363686 | regression | 13,776 | 16 |  |
| diamonds | 46923 | 363631 | regression | 53,940 → fixed 30,000 | 10 | P |

Spares (same criteria): churn (46915), credit_card_clients_default (46919), blood-transfusion-service-center (46913), wine_quality (46964, regression), Food_Delivery_Time (46928).

**Group C (natural mess; verify in P6.5)**

| Dataset | did | Expected mess |
|---|---|---|
| HR_Analytics_Job_Change_of_Data_Scientists | 46935 | experience `">20"`, `"<1"`; company size ranges; `last_new_job` `"never"`, `">4"` |
| in_vehicle_coupon_recommendation | 46937 | income ranges as text; age `"50plus"`, `"below21"`; time `"10AM"`; expiration `"1d"`, `"2h"` |
| heloc | 46932 | special values −7, −8, −9 |
| bank-marketing | 46910 | `"unknown"` tokens; `pdays = -1`; `duration` is a known post-event leak |
| Food_Delivery_Time | 46928 | check whether the TabArena version kept the text formats |

Spares for Group C: adult (1590, `"?"` tokens), titanic (40945), employee_salaries (42125), house prices (42563), credit-approval (29).

**Group D (leakage set)**: credit-g, Fitness_Club, Marketing_Campaign, Bank_Customer_Churn, E-CommereShippingData, maternal_health_risk, concrete_compressive_strength, healthcare_insurance_expenses, Another-Dataset-on-used-Fiat-500, diamonds. Pilot (`pilot-leaks`): credit-g, Bank_Customer_Churn, maternal_health_risk (strong leaks) and concrete_compressive_strength, healthcare_insurance_expenses (moderate leaks).

Group C in the pilot (`pilot-natural`): HR_Analytics_Job_Change_of_Data_Scientists, in_vehicle_coupon_recommendation, heloc (or their replacements from P6.5).

---

## Appendix E. Mess types and leak specs

### E1. Mess types (corruptor)

Levels: 1 = 25% of eligible columns per type (at least one if any is eligible), 5% of cells; 2 = 50% of columns, 15% of cells; 3 = 75% of columns, 30% of cells. For format types (1, 2, 3, 7, 8) the whole column gets the format, and the cell share is the share of minority variants (second unit, other date formats, other spellings).

| # | Type | Eligible columns | What the corruptor does | Expected operation |
|---|---|---|---|---|
| 1 | Units and symbols | continuous numeric, ≥ 20 unique values | `1250` → `"$1,250.00"`, `"12.5%"`, `"3.2k"` | `parse_number` |
| 2 | Mixed units | continuous numeric; unit family from name hints (weight, length, time, volume, money), else generic `k` and `M` | `"1.37kg"` and `"1370 g"` in one column | `parse_units` |
| 3 | Date formats | date columns, or numeric dates (for example MJD in SDSS17); not applicable otherwise | `"2014-10-13"`, `"13/10/2014"`, `"Oct 13 2014"` | `parse_datetime` |
| 4 | Category spellings | categorical, 3 to 50 levels | case, extra spaces, abbreviations, small typos, synonyms from a small dictionary | `map_categories` |
| 5 | Fake missing values | any feature | NaN → `-999`, `9999`, `"?"`, `"unknown"`, `"N/A"`, `"-"` (numeric sentinels only where they are outside the natural range) | `replace_values` |
| 6 | Impossible values | numeric with a natural range | ×10 or ×100 scale errors, sign flips, digit doubling | `set_missing_where` |
| 7 | Boolean spellings | 2-level columns | `"Y"`, `"yes"`, `"TRUE"`, `"1"`, `"t"` vs their negatives, mixed | `parse_boolean` |
| 8 | Packed columns | pairs of numeric columns | `"3 bd / 2 ba"`, `"{w}x{h}"`, `"L:12.3 W:4.5"`; originals removed | `extract_pattern` |
| 9 | Id and index columns | added | random customer ids; `Unnamed: 0` index | `drop_columns` |
| 10 | File layout | the file | title rows, footer notes, `;` delimiter, decimal comma, BOM, cp1252 | reader (E6b only) |
| 11 | Target leakage | added | see E2 | leak guard (Group D only) |
| 12 | Plain missing (control) | any feature | blank cells at random (MCAR) | none (no-harm control) |

### E2. Leak specs (Group D)

One leak column per dataset, with a post-event name, written in `benchmark/leaks.py` with a table in `docs/datasets.md`. Five datasets get a strong leak, five a moderate one.

- Classification: the leak agrees with the positive class in 98% of rows (strong) or 80% (moderate). Example names: `refund_issued`, `account_closed_flag`, `collections_case_opened`, `campaign_response_date`, `final_risk_grade`.
- Regression: target × U(0.97, 1.03) (strong) or target + noise so that the correlation is about 0.8 (moderate). Example names: `appraised_value_after_sale`, `measured_strength_28d`, `claim_paid_amount`.
- Test time (deployment): the leak column is empty in `test_features.csv`. The naive file keeps it.
- Metrics: detection precision, recall and F1 per strength; false leak drops on Group A at level 0; deployment score vs naive score per system.

---

## Appendix F. Project CLAUDE.md

Create this file in P0.3 (keep it under 60 lines):

```markdown
# Agentic AutoML v2 (DS LAB 2, MSc thesis)

- The work order is `buildplan.md`. The status is `progress.md`. Read `progress.md` first in every
  session, then the current phase of `buildplan.md`.
- Judgment calls go to `docs/decisions.md`; changes to pre-registered experiments go to
  `docs/deviations.md`.
- Environment: Windows on ARM, x86_64 Python 3.12 through uv. Use `uv run ...`. The path has spaces:
  quote it. Reconfigure stdout to UTF-8 in entry points.
- Never: push, rewrite git history, print or commit `.env`, invent results or citations, call a real
  LLM in the fast tests, run a full benchmark grid in the foreground, run `ruff format` or `black`.
- Style: the global coding rules apply (one empty line between definitions, imports sorted by line
  length, double quotes, pathlib paths in `paths.py`, lowercase file names without underscores).
  `buildplan.md` §0.6 lists the overrides (error handling, logging, tests, comments).
- LLM spend: target about $4, hard maximum $5 for the whole plan (`configuration/settings.json` → `spend`; the
  client enforces it). Use `gpt-4.1-nano` while developing. Never start the full benchmark grids.
- Commit after every task: `P<phase>.<task>: <what changed>`.
```

---

## Appendix G. progress.md template

```markdown
# Progress

Legend: [ ] to do · [~] in progress · [x] done · [!] blocked · [-] skipped (reason)

## Spend
- development: $0.00 of $1.50
- pilot: $0.00 of $2.75
- total: $0.00 (target about $4, hard maximum $5)

## P0. Setup and safety net
- [ ] P0.1 Read the inputs
- [ ] P0.2 Git
- [ ] P0.3 Memory files
- [ ] P0.4 Freeze v1
- [ ] P0.5 Check the LLM

(... every task of buildplan.md §6, one line each: status, date, commit, note ...)

## Open problems
-

## Phase summaries
-
```
