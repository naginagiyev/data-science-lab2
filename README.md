# Agentic AutoML v2

AutoML for messy tables. Real tables hold numbers written as text with units, dates as text, many
spellings of one category, hidden missing values, id columns and leaky columns. v2 lets an LLM
**choose and set up tested operations** at each decision point (framing, leak review, cleaning plan,
repair, model families, summary), and lets **code check every answer**: data checks on
cross-validation folds, a leak decision table, a validation gate that keeps only cleaning steps that
do not hurt, and cross-validation for the final model. The model is trained by an own engine
(LightGBM, CatBoost, linear, with Optuna) or, as a plug-in, by AutoGluon or FLAML.

This repository also holds the messy-table benchmark used to test it (controlled mess with ground
truth, natural mess, a leakage set). The build plan is `docs/buildplan.md`; the work log is
`docs/progress.md`; every judgment call is in `docs/decisions.md`.

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
main environment): `uv run python -m benchmark.envs.install flaml autogluon`.

## Quick start

Train on one CSV file, Excel sheet, JSON or parquet file:

```
uv run python main.py run data/ibmchurn.csv --target Churn --variant full --budget 120
```

The run prints one line per stage and writes a run folder in `runs/`: `report.md` (task, split, top
audit issues, leak decisions, the plan with each step's gate effect, scores next to a dummy model,
permutation importance, time and LLM cost per stage), `audit.json`, `decisions.jsonl`, `llm.jsonl`,
`plans.json`, `model.joblib` and more. Predict new rows from the raw file format:

```
uv run python main.py predict runs/<run folder>/model.joblib new_rows.csv --out predictions.csv
uv run python main.py report runs/<run folder>
```

Without arguments, `uv run python main.py` starts an interactive mode. A demo that compares
AutoGluon, `full` and `full+ag` on one messy file:

```
uv run python -m benchmark.showcase
```

## Command line

`main.py run <data>` options: `--target`, `--task` (binary, multiclass, regression), `--variant`,
`--budget` (seconds, default 300), `--seed`, `--backend` (own, autogluon, flaml), `--ask` (confirm
the target when the framer is unsure), `--mode` (demo or benchmark), `--out`, `--settings` (a JSON
file merged over `configuration/settings.json`) and `--set key=value` (for example
`--set gate.enabled=false`).

## The pipeline

| Stage | What it does | Code | LLM |
|---|---|---|---|
| 01 read | any file to a raw text view (encodings, delimiters, title and footer rows) | `tools/reader.py` | |
| 02 glance | column types, missing tokens, value patterns | `tools/glance.py`, `tools/columntypes.py` | |
| 03 frame | target, task, metric, id columns | `agents/framer.py` | when unclear |
| 04 split | locked test set (demo) and CV folds: group, time, stratified or random | `tools/splitter.py` | |
| 05 audit | 28 checks on the training rows: structure, formats, values, target, leak signals | `audit/` | |
| 06 leak guard | ids and leaks dropped by evidence; the LLM names after-the-event columns | `checks/leakage.py`, `agents/leakreview.py` | yes |
| 07 plan | translator plan from the issues; the LLM cleaner writes a full plan | `operations/translator.py`, `agents/cleaner.py` | yes |
| 08 compile | data checks per step on folds; LLM repair of failing steps | `checks/datachecks.py` | yes |
| 09 gate | P0 (protected steps) vs P1 (all steps), step tests, LLM refine rounds | `checks/gate.py` | yes |
| 10 model | the LLM chooses families and time shares; Optuna and CV decide | `agents/modeller.py`, `modelling/engine.py` | yes |
| 11 test | the locked test set, once; permutation importance | `modelling/evaluator.py` | |
| 12 report | `report.md` with a checked LLM summary | `tools/report.py`, `agents/summarizer.py` | yes |

Every stage has a time limit and a fallback, so a failed LLM call never ends a run.

## Variants

| Variant | Audit | LLM | Cleaning plan | Gate | Engine |
|---|---|---|---|---|---|
| `generic` | no | no | none (typing and encoding only) | no | own |
| `rules` | yes | no | rules translate the audit issues | yes | own |
| `full` | yes | yes | LLM plan from the operation catalog, repaired and refined | yes | own |
| `full-nogate` | yes | yes | as `full` | no | own |
| `freecode` | yes | yes | LLM writes one cleaning class (sandbox) | yes | own |
| `full+ag` | yes | yes | as `full` | yes | AutoGluon on the prepared data |

Generated code (`freecode`) runs in a separate Python process with an AST check of imports and
names, a time limit, a temporary folder and no API key. This is a safety net, not a security
boundary: do not run it on untrusted data or prompts.

## Reproduce the benchmark

One command per grid; finished cells are skipped, so a stopped run can be started again:

```
uv run python -m benchmark.reproduce --grid smoke    # all 11 systems on two dev samples
uv run python -m benchmark.reproduce --grid pilot    # the four pilot grids, figures, docs/results.md
```

`pilot` writes the benchmark files (OpenML downloads on first use), runs `pilot-main` (E1, E2, E4),
`pilot-leaks` (E3), `pilot-ablations` (E5) and `pilot-natural` (E10), and generates
`benchmark/results/figures/`, `benchmark/results/tables.md` and `docs/results.md`. The dev pool needs
four local files in `data/` (git-ignored): `ibmchurn.csv` (IBM Telco Customer Churn),
`kinghousedata.csv` (King County house sales), `winequality.csv` (Wine Quality) and
`creditcardfraud.csv` (Credit Card Fraud Detection), all public Kaggle datasets; the smoke grid uses
ibmchurn. Every other dataset comes from OpenML.

The full grids `main` (1,440 cells), `leaks` (180) and `ablations` (160) are written and checked but
not run; start one with `uv run python -m benchmark.reproduce --grid main`.

## Results

Every benchmark run is one row of `benchmark/results/runs.csv` (the column `grid` names the
experiment). `docs/results.md` is generated from it. The pilot results of the code before the P14
restructuring are kept in `benchmark/results/runs-before-p14.csv`; the experiment protocol is
`docs/protocol.md`, and changes to it are in `docs/deviations.md`.

## Folder map

```
main.py, orchestrator.py   CLI and the 12-stage pipeline (one function for demo and benchmark)
settings.py, schemas.py    settings (configuration/settings.json) and typed records
paths.py                   every path of the project
configuration/             settings, metrics, audit catalog and thresholds, tokens
tools/                     reader, column types, glance, splitter, run folder, metrics, report, sandbox
audit/                     the code-only audit of the training rows
operations/                the typed cleaning operations, translator, compiler
checks/                    data checks, leak guard, validation gate
modelling/                 engine, cross-validation, evaluator, fitted model, backends
agents/                    framer, leak review, cleaner, free-code writer, modeller, summary
llms/                      LLM client (structured output, cache, retries, budgets, spend ledger)
prompts/                   the LLM prompts
benchmark/                 datasets, corruptor, leaks, runner, systems, grids, analysis, results
tests/                     fast suite (uv run pytest -q) and slow tests (-m slow)
docs/                      build plan, progress log, decisions, deviations, data cards, protocol,
                           results, architecture, showcase, background/
cache/                     git-ignored: LLM answer cache, OpenML downloads, benchmark files and
                           run folders of every benchmark run
```
