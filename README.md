# Agentic AutoML v2

AutoML for messy tables. Real tables hold numbers written as text with units, dates as text, many
spellings of one category, hidden missing values, id columns and leaky columns. v2 lets an LLM
**choose and set up tested cleaning operations**, and lets **code check every step**: data checks,
issue-level verification with the auditor `edatool`, and a cross-validation gate that keeps only
steps that do not hurt. The model is trained by an own engine (LightGBM, CatBoost, linear, with
Optuna) or, as a plug-in, by AutoGluon or FLAML.

This repository also holds the messy-table benchmark used to test it (controlled mess with ground
truth, natural mess, a leakage set) and the results of a pilot study. The build plan is
`docs/buildplan.md`; the work log is `docs/progress.md`; every judgment call is in `docs/decisions.md`.

## Install

Windows, macOS or Linux with [uv](https://docs.astral.sh/uv/). The project uses Python 3.12
(x86_64 on Windows on ARM, see `.python-version`).

```
uv sync
```

The LLM variants need an OpenAI key in a file `.env` at the project root (never committed):

```
OPENAI_API_KEY=sk-...
```

Optional baseline and backend environments (each in its own folder, so their pins never touch the
main environment): `uv run python -m benchmark.envs.install flaml autogluon edabench`.

## Quick start

Train on one CSV file, Excel sheet, JSON or parquet file:

```
uv run python main.py run data/ibmchurn.csv --target Churn --variant full --budget 120
```

The run prints one line per stage and writes a run folder in `runs/`: `report.md` (task, split, audit
issues, leak decisions, the plan with each step's gate effect, scores next to a dummy model, time
and LLM cost per stage), `decisions.jsonl`, `llm.jsonl`, `plans.json`, `model.joblib` and more.
Predict new rows from the raw file format:

```
uv run python main.py predict runs/<run folder>/model.joblib new_rows.csv --out predictions.csv
uv run python main.py report runs/<run folder>
```

Without arguments, `uv run python main.py` starts an interactive mode. A five-minute demo that
compares AutoGluon, `full` and `full+ag` on one messy file (LLM answers from the cache, $0):

```
uv run python -m benchmark.showcase
```

## Command line

`main.py run <data>` options: `--target`, `--task` (binary, multiclass, regression), `--variant`,
`--budget` (seconds, default 300), `--seed`, `--backend` (own, autogluon, flaml), `--ask` (confirm
the target when the framer is unsure), `--mode` (demo or benchmark), `--out`, `--settings` (a JSON
file merged over `configuration/settings.json`) and `--set key=value` (for example
`--set gate.enabled=false`).

## Variants

| Variant | Audit | LLM review | Cleaning plan | Gate | Repair | Engine |
|---|---|---|---|---|---|---|
| `generic` | no | no | none (typing and encoding only) | no | no | own |
| `rules` | yes | no | rules translate the audit issues | yes | no | own |
| `full` | yes | yes | LLM plan from the operation catalog | yes | yes | own |
| `full+ag`, `rules+ag` | as `full`, `rules` | | | | | AutoGluon on the prepared data |
| `freecode` | yes | yes | LLM writes a cleaning class (sandbox) | yes | trial runs | own |
| `full-noreview`, `full-nogate`, `full-norepair`, `full-nosamples`, `full-noleakguard` | ablations of `full` | | | | | own |

Generated code (the `custom` operation and `freecode`) runs in a separate Python process with an AST
check of imports and names, a time limit, a temporary folder and no API key. This is a safety net,
not a security boundary: do not run it on untrusted data or prompts.

## Reproduce the benchmark

One command per grid; finished cells are skipped, so a stopped run can be started again:

```
uv run python -m benchmark.reproduce --grid smoke    # about 25 min, < $0.01
uv run python -m benchmark.reproduce --grid pilot    # about 8 h, about $1.7
```

`pilot` writes the benchmark files (OpenML downloads on first use), runs every pilot grid, the repair
checks and the framer test, and generates `benchmark/results/figures/`, `benchmark/results/tables.md`,
`docs/results.md` and `docs/projection.md`. The dev pool needs four local files in `data/` (git-ignored):
`ibmchurn.csv` (IBM Telco Customer Churn), `kinghousedata.csv` (King County house sales),
`winequality.csv` (Wine Quality) and `creditcardfraud.csv` (Credit Card Fraud Detection), all public
Kaggle datasets; the smoke grid uses ibmchurn. Every other dataset comes from OpenML.

The full grids are written and checked but were not run in this project; estimates from the pilot
(2 workers, `gpt-4.1-mini`; see `docs/projection.md`):

| Grid | Runs | Time | LLM cost |
|---|---|---|---|
| `main` | 1,440 | about 36 h | about $5.2 |
| `leaks` | 210 | about 6 h | about $0.9 |
| `ablations` | 288 | about 12 h | about $2.1 |
| `types` | 320 | about 9 h | about $1.9 |
| `budgets` | 45 | about 2.5 h | about $0.4 |

Run them with `uv run python -m benchmark.reproduce --grid main` (and so on).

## Results

Every benchmark run is one row of `benchmark/results/runs.csv` (column `grid` names the
experiment; the `p2-*` to `p7-*` grids are build checkpoints). `docs/results.md` (pilot results,
small sample) and `docs/projection.md` (what the full benchmark would show, with power estimates
and a recommendation). The design and the protocol are fixed in
`docs/preregistration.md`; every later change is in `docs/deviations.md`.

## Folder map

```
main.py, orchestrator.py   CLI and the 12-stage pipeline (one function for demo and benchmark)
settings.py, schemas.py    settings (configuration/settings.json) and typed records
paths.py                   every path of the project
agents/                    framer, cleaner (plan, repair, refine), free-code writer, summary
llms/                      LLM client (structured output, cache, retries, budgets, spend ledger)
operations/                the typed cleaning operations, translator, compiler
checks/                    data checks and verify, leak guard, validation gate
modelling/                 engine, cross-validation, fitted model, AutoGluon/FLAML backends
tools/                     reader, glance, splitter, run folder, metrics, report, sandbox
prompts/                   the LLM prompts
edatool/, edabench/        the auditor and its own benchmark (vendored and improved)
benchmark/                 datasets, corruptor, leaks, runner, systems, grids, analysis, results
tests/                     fast suite (uv run pytest -q) and slow tests (-m slow)
docs/                      build plan, progress log, decisions, deviations, data cards,
                           pre-registration, results, architecture, study.html, background/
cache/                     git-ignored: LLM answer cache, OpenML downloads, benchmark files and
                           run folders (report, decisions, LLM log of every benchmark run)
```
