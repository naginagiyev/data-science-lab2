# Progress

Legend: [ ] to do · [~] in progress · [x] done · [!] blocked · [-] skipped (reason)

Line format: status, task, date, commit (filled in by the next commit), note.

## Spend
- development: $0.0001 of $1.50
- pilot: $0.00 of $2.75
- total: $0.0001 (target about $4, hard maximum $5)

## P0. Setup and safety net
- [x] P0.1 Read the inputs · 2026-10-11 · no files · read buildplan, v2-plan Parts 1, 2, 4, overview.md, eda README and architecture §1-16, edaadapter and the wine log
- [x] P0.2 Git · 2026-10-11 · 73f0f98 · git init, ignores, .gitattributes (LF), tag v1-dslab1
- [x] P0.3 Memory files · 2026-10-11 · f701d8c · CLAUDE.md, progress.md, docs/decisions.md, docs/deviations.md
- [x] P0.4 Freeze v1 · 2026-10-11 · 6d8ed7f · v1 code copied to benchmark/legacy/v1, diff with the root shows no differences
- [x] P0.5 Check the LLM · 2026-10-11 · 0fa301c · nano and mini work with temperature 0 and seed 0; prices unchanged; ledger started ($0.00005)

## P1. Foundation
- [x] P1.1 Environment · 2026-10-11 · 6271716 · Python 3.12.14 x86_64 (win-amd64), all packages import; versions in decisions.md
- [x] P1.2 Remove v1 from the root · 2026-10-11 · 900fa72 · v1 logic removed; main.py, paths.py, schemas.py, metrics.json kept for rewriting
- [x] P1.3 Core files · 2026-10-11 · f00390e · paths, settings (+ overrides, hash), metrics.json, schemas; settings and style tests pass
- [x] P1.4 Run folder, logging, metrics · 2026-10-11 · 9509f16 · RunFolder, UTF-8 run.log + ASCII console, metrics with aligned probabilities; missing-class test passes
- [x] P1.5 LLM layer · 2026-10-11 · 0d7ec3e · client (cache, retries, budgets, logging), OpenAI + fake provider, spend ledger; real nano test passed ($0.00001); cap refusal tested
- [x] P1.6 Orchestrator and CLI skeleton · 2026-10-11 · cd982be · run_stage, Budget, RunState, stub stages, CLI run/predict/report + interactive; ibmchurn run creates context.json and run.log

## P2. Baselines first
- [x] P2.1 Harness skeleton · 2026-10-11 · 4ef5d7a · runner (resumable, time-out + tree kill, 2 workers, spend stop), scoring (label keys, dummy fallback), splits, materialize; 4 local datasets level 0 seeds 0-2 written; hand-made predictions scored correctly
- [x] P2.2 Main-environment baselines (dummy, lightgbm, skrub) · 2026-10-11 · 81f7f0e, 929e78b · all 3 score on the 4 local datasets (benchmark/results/p2-local.csv)
- [x] P2.3 Separate environments (flaml, autogluon) · 2026-10-11 · 80be799 · flaml 2.7.0 and autogluon 1.6.3 in own uv envs (no WSL needed); both score on the 4 local datasets with 60 s
- [x] P2.4 v1 adapter (optional, never run) · 2026-10-11 · b1d7b1c · v1 env from frozen lock (3.10); headless.py fails cleanly with a refused OPENAI_BASE_URL (slow test)
- [-] P2.5 Optional baseline (tabpfn) · 2026-10-11 · efebcc4 · skipped: tabpfn 9.1.0 weights are gated (license acceptance via browser login)
- [x] P2.6 Baseline smoke table · 2026-10-11 · 837c64d · 20/20 cells ok, table below (benchmark/results/p2-smoke.csv)

P2.6 smoke table (4 local datasets, seed 0, 60 s budget, level 0; score and wall seconds; RMSE in price units, log-loss and RMSE lower is better):

| system | ibmchurn (roc_auc) | kinghousedata (rmse) | winequality (log_loss) | creditcardfraud (roc_auc) |
|---|---|---|---|---|
| dummy | 0.5000 (1 s) | 345120 (1 s) | 1.1818 (1 s) | 0.5000 (1 s) |
| lightgbm | 0.8386 (2 s) | 111720 (2 s) | 1.0082 (2 s) | 0.9384 (3 s) |
| skrub | 0.8417 (10 s) | 109407 (9 s) | 1.1541 (9 s) | 0.7668 (9 s) |
| flaml | 0.8506 (58 s) | 104370 (58 s) | 0.7583 (56 s) | 0.9998 (56 s) |
| autogluon | 0.8499 (50 s) | 101097 (60 s) | 0.7824 (34 s) | 1.0000 (60 s) |

Note: FLAML and AutoGluon search under a wall-clock limit, so their scores move between runs of the same cell (AutoGluon on creditcardfraud: 0.9419 in the P2.3 check, 1.0000 here). This is why the pilot uses 2 seeds for the free systems.

## P3. v2 core without LLM (variant generic)
- [x] P3.1 Vendor edatool · 2026-10-11 · 72d3773 · edatool/ and edabench/ copied; edabench paths in root paths.py; CLI audit (--no-agent) works on the 4 local datasets (2.9-22.7 s)
- [x] P3.2 Reader and glance · 2026-10-11 · ad36fa1 · reader (raw view, hard stops), glance + pattern stats; TotalCharges shows ' ':11, kinghouse date is datetime (YYYYMMDDTHHMMSS)
- [x] P3.3 Splitter · 2026-10-11 · b92e81e · presplit advice -> group/time/stratified/random hold-out + folds; kinghousedata: group split on id
- [x] P3.4 Typing step and generic stage · 2026-10-11 · (next) · pandas-parity test passes; LightGBM fits on torture headers (tree and linear kinds)
- [ ] P3.5 Engine
- [ ] P3.6 Evaluator, FittedModel, report, predict
- [ ] P3.7 v2 adapter

## P4. edatool improvements
- [ ] P4.1 Baseline numbers before changing anything
- [ ] P4.2 Windows-safe output
- [ ] P4.3 Loader robustness
- [ ] P4.4 Semantic-mess detectors
- [ ] P4.5 Policy and cost
- [ ] P4.6 Benchmark the auditor again

## P5. Toolbox and verification (variant rules)
- [ ] P5.1 Operations
- [ ] P5.2 Row and target operations
- [ ] P5.3 Translator
- [ ] P5.4 Compiler, data checks, verify integration
- [ ] P5.5 Leak guard
- [ ] P5.6 Gate
- [ ] P5.7 Variant rules end to end

## P6. Benchmark data
- [ ] P6.1 Dataset registry
- [ ] P6.2 Corruptor
- [ ] P6.3 Leak generator
- [ ] P6.4 Materialize the dev pool
- [ ] P6.5 Group C

## P7. LLM agents (variant full)
- [ ] P7.1 Framer
- [ ] P7.2 edatool agent integration
- [ ] P7.3 Cleaner
- [ ] P7.4 Repair loop
- [ ] P7.5 Refine loop
- [ ] P7.6 Sandbox and custom operation
- [ ] P7.7 Free-code variant
- [ ] P7.8 Summary paragraph
- [ ] P7.9 Dev iteration

## P8. Robustness and performance
- [ ] P8.1 Torture suite
- [ ] P8.2 Fault injection
- [ ] P8.3 Performance
- [ ] P8.4 Reproducibility test
- [ ] P8.5 Fresh-clone test

## P9. Pilot experiments (no full benchmark)
- [ ] P9.1 Materialize
- [ ] P9.2 Pre-registration
- [ ] P9.3 Smoke grid and freeze
- [ ] P9.4 E7 framer accuracy
- [ ] P9.5 pilot-main
- [ ] P9.6 pilot-leaks
- [ ] P9.7 pilot-ablations
- [ ] P9.8 pilot-types and E6b reading robustness
- [ ] P9.9 pilot-seed
- [ ] P9.10 pilot-natural
- [ ] P9.11 pilot-budget
- [ ] P9.12 E8 audit benchmark (final)
- [ ] P9.13 Held-out cells

## P10. Analysis
- [ ] P10.1 benchmark/analysis.py
- [ ] P10.2 benchmark/figures.py
- [ ] P10.3 docs/results.md
- [ ] P10.4 benchmark/projection.py and docs/projection.md

## P11. Documentation and demo
- [ ] P11.1 Code quality pass
- [ ] P11.2 Documentation
- [ ] P11.3 Demo

## P12. Final quality check and handover
- [ ] P12.1 Checks
- [ ] P12.2 Handover

## Stretch goals (only after P12)
- [ ] S1 Feature ideas in the CAAFE style
- [ ] S2 LLM swap (RQ6)
- [ ] S3 Full grids (only if the user asks)
- [ ] S4 AutoGluon Assistant baseline

## Open problems
-

## Phase summaries
- **P0 (2026-10-11).** Created: `.gitattributes`, `CLAUDE.md`, `progress.md`, `docs/decisions.md`,
  `docs/deviations.md`, `benchmark/legacy/v1/` (an exact copy of the v1 code),
  `benchmark/results/spend.json`. Edited: `.gitignore` (runs, cache, parquet, catboost and AutoGluon
  folders). Deleted: nothing. What changed: the folder is now a git repository; v1 is saved with the
  tag `v1-dslab1` and as a frozen copy; both OpenAI models work with temperature 0 and seed 0, and
  their prices match the plan. LLM spend so far: $0.00005.
- **P1 (2026-10-11).** Created: `settings.py`, `configuration/settings.json`, `orchestrator.py`,
  `tools/runfolder.py`, `tools/metrics.py`, `llms/client.py`, `llms/providers.py`, `benchmark/spend.py`,
  package `__init__.py` files, and tests (`testsettings`, `teststyle`, `testmetrics`, `testrunfolder`,
  `testllmclient`, `testllmreal`, `testorchestrator`). Edited (rewritten): `pyproject.toml`, `uv.lock`,
  `.python-version`, `paths.py`, `schemas.py`, `main.py`, `configuration/metrics.json`, `.gitignore`.
  Deleted: the v1 code in `agents/`, `llms/`, `tools/`, `prompts/`, `main2.py`, `clean.py`, three files
  in `configuration/`, and the empty `models/`, `sandbox/` and egg-info folders. What changed: a new
  Python 3.12 x86_64 environment; settings with overrides and a hash; run folders with UTF-8 logs; fixed
  metrics computed from probabilities; one LLM client with cache, retries, budgets and a global spend
  ledger; and a pipeline skeleton with a CLI. 38 fast tests pass. LLM spend so far: $0.00006.
- **P2 (2026-10-11).** Created: `benchmark/datasets.py`, `splits.py`, `materialize.py`, `scoring.py`,
  `runner.py`, `benchmark/systems/` (`common.py`, `dummy.py`, `lightgbm.py`, `skrub.py`, `flaml.py`,
  `autogluon.py`, `registry.json`), `benchmark/envs/` (`install.py`, `flaml.txt`, `autogluon.txt`,
  `tabpfn.txt`), `benchmark/legacy/v1/headless.py`, grids `p2-*.json`, results `p2-*.csv`, and tests
  (`testharness`, `testbaselines`, `testv1adapter`). Edited: `paths.py`, `docs/decisions.md`. Deleted:
  nothing (the TabPFN environment was created and removed). What changed: a resumable benchmark harness
  (fixed splits, text files, subprocess systems with hard time-outs, scoring with a dummy fallback);
  five baselines that all work on the four local datasets; a v1 adapter that is ready but never run;
  TabPFN skipped because its weights are gated.
