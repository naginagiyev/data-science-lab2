# Progress

Legend: [ ] to do · [~] in progress · [x] done · [!] blocked · [-] skipped (reason)

Line format: status, task, date, commit (filled in by the next commit), note.

## Spend
- development: $0.0421 of $1.50
- pilot: $0.00 of $2.75
- total: $0.0421 (target about $4, hard maximum $5)

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
- [x] P3.4 Typing step and generic stage · 2026-10-11 · 0e99f20 · pandas-parity test passes; LightGBM fits on torture headers (tree and linear kinds)
- [x] P3.5 Engine · 2026-10-11 · 75d6853 · dummy/linear/LightGBM/CatBoost + Optuna under a deadline, OOF average of best two, refit; kinghousedata 60 s budget -> 58.7 s
- [x] P3.6 Evaluator, FittedModel, report, predict · 2026-10-11 · e5956db · generic run folders complete for the 4 local datasets; predict works on test features
- [x] P3.7 v2 adapter · 2026-10-11 · 49cbb8c · generic is at least as good as lightgbm and skrub on all 4 local datasets (table below)

P3.7 sanity table (benchmark mode, 4 local datasets, seed 0, 60 s budget; `generic` from p3-generic.csv, the rest from p2-smoke.csv):

| system | ibmchurn (roc_auc) | kinghousedata (rmse) | winequality (log_loss) | creditcardfraud (roc_auc) |
|---|---|---|---|---|
| dummy | 0.5000 | 345120 | 1.1818 | 0.5000 |
| lightgbm | 0.8386 | 111720 | 1.0082 | 0.9384 |
| skrub | 0.8417 | 109407 | 1.1541 | 0.7668 |
| flaml | 0.8506 | 104370 | 0.7583 | 0.9998 |
| autogluon | 0.8499 | 101097 | 0.7824 | 1.0000 |
| generic | 0.8510 | 103896 | 0.8067 | 1.0000 |

## P4. edatool improvements
- [x] P4.1 Baseline numbers before changing anything · 2026-10-11 · 3bc011f · recall 0.986, precision 0.896 on 358 injected issues (same as before); baselines: ydata 0.433, old eda.py 0.317, cleanlab 0.158 recall; docs/edatool-baseline.md
- [x] P4.2 Windows-safe output · 2026-10-11 · 3eb03e2 · SafeConsoleText filter on the edatool logger; cp1252 test: 0 logging errors (2 without the filter)
- [x] P4.3 Loader robustness · 2026-10-11 · 60d2dc9 · BOM/UTF-16/cp1252 files, ; tab | delimiters, title rows and footer notes (CSV and Excel), quoted newlines, empty and header-only files: tests pass
- [x] P4.4 Semantic-mess detectors · 2026-10-11 · 830bb36 · B07 units, B08 packed, B09 mixed dates, new tokens, fix vocabulary injected; tests pass; no new issues on the 4 local datasets
- [x] P4.5 Policy and cost · 2026-10-11 · bfa1258, b438abe · creditcardfraud 5k sample, nano: 223,178 -> 14,144 agent tokens (6.3%), both runs $0.0247; batching, session cap, time limits, anonymized skip, F05/F07 no_action
- [x] P4.6 Benchmark the auditor again · 2026-10-11 · 57cbcd2 · old kinds identical (recall per kind, precision 0.8957); new kinds 14/14; before/after in docs/edatool-baseline.md

## P5. Toolbox and verification (variant rules)
- [x] P5.1 Operations · 2026-10-11 · d981c3e · 17 column operations + registry/catalog text; contract tests (index, unseen rows, NaN, unicode, empty column, one row, pickle) pass
- [x] P5.2 Row and target operations · 2026-10-11 · d981c3e · row filters never transform; map_target_labels never merges classes; log_target exact inverse
- [x] P5.3 Translator · 2026-10-11 · aab5dd7 · Appendix C default plans look right on churn, kinghousedata and wine (see decisions.md)
- [x] P5.4 Compiler, data checks, verify integration · 2026-10-11 · 47a805b · wrong unit map -> "parse_units on Weight: 40.0% of values could not be parsed, examples: \"800 g\", ..."; verify resolves C02 after replace_values
- [x] P5.5 Leak guard · 2026-10-11 · c00c5ff · decision-table tests pass (structural, semantic + statistical, semantic only, statistical only, guard off)
- [x] P5.6 Gate · 2026-10-11 · 160b245 · helpful unit parse accepted (positive effect); dropping the strongest feature and a class-removing row filter rejected; validation folds never filtered
- [x] P5.7 Variant rules end to end · 2026-10-11 · 380f8e1 · complete run folders on the 4 local datasets; rules >= generic on ibmchurn and kinghousedata (table below); worse on winequality (explained)

P5.7 table (benchmark mode, 4 local datasets, seed 0, 120 s budget, benchmark/results/p5-rules.csv):

| system | ibmchurn (roc_auc) | kinghousedata (rmse) | winequality (log_loss) | creditcardfraud (roc_auc) |
|---|---|---|---|---|
| generic | 0.8517 | 102687 | 0.8076 | 1.0000 |
| rules | 0.8528 | 100064 | 0.8245 | 1.0000 |

winequality: `rules` drops 240 exact duplicate training rows (a protected step); they look like genuine repeated wines, and without them the class weights change (CV log-loss 0.894 vs 0.876 for `generic`). creditcardfraud: equal within noise.

## P6. Benchmark data
- [x] P6.1 Dataset registry · 2026-10-11 · f581cd9 · 31 OpenML datasets downloaded and checked (ids, targets, tasks, licences); manifest + data cards; no spare needed
- [x] P6.2 Corruptor · 2026-10-11 · 2aa8f7a · 12 types (layout in the writer, leak in P6.3); determinism, nested levels, exact restore, target untouched; one example per type in docs/datasets.md
- [x] P6.3 Leak generator · 2026-10-11 · a4c88e3 · 10 post-event leak columns; G01 finds all 5 strong leaks; naive and deployment test files
- [x] P6.4 Materialize the dev pool · 2026-10-11 · 5bef67a · 6 dev datasets x levels 0-3 x seeds 0-2 = 72 cells with truth.json and cells.parquet (cache/benchmark/files/dev, 683 MB, not in git)
- [x] P6.5 Group C · 2026-10-11 · 82019e6 · 4 of 5 messy as expected; Food_Delivery_Time (cleaned) replaced by employee_salaries; section "Natural mess" in docs/datasets.md

## P7. LLM agents (variant full)
- [x] P7.1 Framer · 2026-10-11 · a6b3189 · fake-provider tests and --ask pass; real nano check: 4/4 targets right
- [x] P7.2 edatool agent integration · 2026-10-11 · 54f6c6d · agent calls in llm.jsonl and the ledger (real nano run: 14 calls, 81k tokens, $0.0085); provider failure falls back to the deterministic audit (test)
- [x] P7.3 Cleaner · 2026-10-11 · a26b73d · fake-provider tests pass; real nano run on a level-3 churn sample: valid plan with parse_boolean and map_categories on the first try
- [x] P7.4 Repair loop · 2026-10-11 · 9904ca9 · wrong unit map repaired from the structured error (test); capped at 3 attempts; no-repair ablation drops at once
- [x] P7.5 Refine loop · 2026-10-11 · 7af39ad · a plan with a harmful step is revised from the gate evidence (test); best gated round wins
- [x] P7.6 Sandbox and custom operation · 2026-10-11 · 0b4368b · AST-check, time-out and no-API-key tests pass; custom op via the sandbox; code generation with retry
- [x] P7.7 Free-code variant · 2026-10-11 · 82f41a9 · fake-provider tests and an end-to-end freecode run pass; nano on churn and kinghouse samples: success and failure logged, gate removes code that does not help
- [x] P7.8 Summary paragraph · 2026-10-11 · 55e6a8b · summary in report.md (test and real nano check); number and length checks; provider failure keeps the template report
- [x] P7.9 Dev iteration · 2026-10-12 · f8d8f91 · 2 free and 2 nano rounds on 12 dev5k cells, then mini on 6: 100% success; mini $0.029 per full run; full >= rules >= generic on the messy files (exceptions explained in decisions.md); dev spend $0.36

## P8. Robustness and performance
- [x] P8.1 Torture suite · 2026-10-12 · 15c1960 · 26 files: 20 give a model and a report, 6 stop with a clear message (header only, one row, one class, and three files whose rows are all duplicates in demo mode); the 1M-row file is measured in P8.3
- [x] P8.2 Fault injection · 2026-10-12 · 8cfa0fd · operation, Optuna, candidate model, verify and provider faults: every run ends with a model and a report (fast and slow tests pass)
- [x] P8.3 Performance · 2026-10-12 · 97a2bc3 · rules at 300 s: creditcardfraud (284,807 rows) 286 s wall, peak 2.14 GB; 1,000,000-row file 267 s, peak 1.35 GB (after fixes to verify rows, refit time and a finish reserve); full not measured (no cached answers)
- [x] P8.4 Reproducibility test · 2026-10-12 · afa30fb · same seed + warm LLM cache: identical plans, CV and test scores (Optuna off; a time-bounded search depends on machine load)
- [x] P8.5 Fresh-clone test · 2026-10-12 · 1d6defb · clone + uv sync + pytest: 182 passed (192 s cold first run, 140 s warm)

## P9. Pilot experiments (no full benchmark)
- [x] P9.1 Materialize · 2026-10-12 · df33b79 · 390 cells (A 180, single types 165, C 15, D 30); manifest benchmark/results/files.json; packed-position bug fixed
- [x] P9.2 Pre-registration · 2026-10-12 · (next) · docs/preregistration.md (pilot datasets, support criteria, held-out cells and prediction rule, exclusion rules); pilot, smoke and full grid files; tag prereg
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
- Solved in P7.9: nano's semantic name review flagged normal features as post-event leaks (stricter edatool prompt).
- Small tables (about 1,400 rows): the gate's P1 vs P0 comparison is noisy and can keep P0 although every step helps alone (P9 pilot: watch Is-this-a-good-customer-sized datasets).
- Full creditcardfraud (227k training rows): the generic model stage took 79 s for a 60 s budget (refit on all rows); check in P8.3 at 300 s.

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
- **P3 (2026-10-11).** Created: `edatool/` and `edabench/` (vendored), `edatool/core/patterns.py`,
  `edatool/README.md`, `tools/reader.py`, `tools/glance.py`, `tools/splitter.py`, `tools/report.py`,
  `operations/generic.py`, `modelling/cv.py`, `engine.py`, `fittedmodel.py`, `evaluator.py`,
  `agents/framer.py` (rules part), `benchmark/systems/v2.py`, grid `p3-generic.json`, results
  `p3-generic.csv`, and tests (`testreader`, `testglance`, `testsplitter`, `testgeneric`, `testengine`,
  `testpipeline`). Edited: `orchestrator.py` (real stages), `main.py`, `schemas.py`, `settings.py`,
  `configuration/settings.json` (framer section, `engine.max_search_rows`), `paths.py`,
  `benchmark/systems/registry.json`. Deleted: `edabench/paths.py` (paths moved to the root). What
  changed: the `generic` variant runs end to end: raw text in, quick look, rule framing, split by
  edatool's advice, pandas-like typing, skrub's generic stage, a model engine with Optuna under a time
  budget, one use of the test set, a saved model that predicts raw input, and a report. In benchmark
  mode it beats the LightGBM and skrub baselines on the 4 local datasets.
- **P4 (2026-10-11).** Created: `docs/edatool-baseline.md`, `benchmark/envs/edabench.txt`,
  `llms/agentmodel.py`, `edabench/results/*`, tests (`testedatooloutput`, `testloader`, `testdetectors`,
  `testauditpolicy`). Edited: edatool (`core/progress.py`, `core/loader.py`, `core/patterns.py`,
  `detectors/formats.py`, `detectors/structure.py`, `detectors/leakage.py`, `agent/session.py`,
  `agent/gate.py`, `agent/prompts/*.md`, `config/*.json`, `schemas.py`, `api.py`, `__main__.py`),
  `edabench/injector.py`, `edabench/evaluate.py`. Deleted: nothing. What changed: the auditor no longer
  breaks a cp1252 console; it reads BOM, UTF-16 and cp1252 files, other delimiters, title rows and footer
  notes; it finds numbers with units, packed values and mixed date formats; it never proposes to change
  the target's label space; on anonymized data its agent costs 6% of the tokens it used before. On the
  injection benchmark the old results are unchanged and the new kinds are all found. LLM spend: $0.0247.
- **P5 (2026-10-11).** Created: `operations/` (`base`, `parsing`, `values`, `columns`, `categories`,
  `text`, `rows`, `target`, `registry`, `translator`, `compiler`), `checks/` (`datachecks`, `leakage`,
  `gate`), grid `p5-rules.json`, results `p5-rules.csv`, tests (`testoperations`, `testtranslator`,
  `testcompiler`, `testleakage`, `testgate`). Edited: `orchestrator.py` (audit, leak guard, plan,
  compile and gate stages; the final plan feeds the model), `modelling/engine.py` (log-target transform),
  `modelling/evaluator.py` (target spellings), `tools/report.py`, `settings.py` and
  `configuration/settings.json` (`gate.n_jobs`, `gate.max_rows` 10,000), `tests/testpipeline.py`.
  Deleted: nothing. What changed: the `rules` variant works without any LLM: edatool audits the training
  rows, the leak guard drops ids and leaks, the translator turns issues into typed operations, data
  checks and edatool verify test them, and the gate keeps only steps that do not hurt cross-validation.
- **P6 (2026-10-11).** Created: `benchmark/corruptor.py`, `benchmark/leaks.py`, `benchmark/datacards.py`,
  `docs/datasets.md`, `benchmark/results/datasets.json`, `mess-examples.md`, `leaks.md`, `natural-mess.md`,
  tests (`testcorruptor`, `testleaks`). Edited: `benchmark/datasets.py` (all OpenML pools),
  `benchmark/materialize.py` (mess levels, single types, Group D leak files, cells.parquet),
  `benchmark/runner.py` (mess folders, naive test files). Deleted: nothing. What changed: the benchmark
  data exists: 31 OpenML datasets are downloaded and documented; a corruptor writes 12 kinds of
  realistic mess at 4 nested levels with exact ground truth; Group D gets one post-event leak column per
  dataset; Group C is checked for real mess; the dev pool is written at all levels and seeds.
- **P7 (2026-10-12).** Created: `agents/cleaner.py`, `agents/freecoder.py`, `agents/summarizer.py`,
  `tools/sandbox.py`, `operations/custom.py`, `modelling/backends.py`, `modelling/backendworker.py`,
  prompts (`framer.md`, `cleaner.md`, `repair.md`, `refine.md`, `customop.md`, `freecode.md`,
  `summary.md`), grids `p7-dev.json` and `p7-mini.json`, results `p7-dev.csv` and `p7-mini.csv`, and
  tests (`testframer`, `testagentmodel`, `testcleaner`, `testrepair`, `testrefine`, `testsandbox`,
  `testfreecode`, `testsummary`, `testbackends`). Edited: `orchestrator.py`, `settings.py`,
  `configuration/settings.json`, `paths.py`, `agents/framer.py`, `llms/agentmodel.py`,
  `benchmark/spend.py`, `benchmark/materialize.py`, `checks/gate.py`, `checks/datachecks.py`,
  `modelling/engine.py`, `operations/registry.py`, `operations/target.py`, `operations/translator.py`,
  edatool (`agent/prompts/semantic.md`, `config/tokens.json`, `core/typeinference.py`), tests
  (`testcompiler`, `testgate`, `testtranslator`), `docs/decisions.md`. Deleted: nothing. What changed:
  the `full` variant works end to end: the LLM frames the task, reviews the audit, writes a plan from
  the operation catalog, repairs failing steps and refines the plan from the gate evidence; code
  checks every answer. Generated code (custom steps and the `freecode` variant) runs only in a
  sandbox process without the API key. Reports get a short checked LLM summary. AutoGluon and FLAML
  can train on the prepared data (`full+ag`, `rules+ag`). The dev iteration fixed packed values taken
  for ids, currency codes, k/M suffixes, slow verify and engine starvation, and untested drops. With
  `gpt-4.1-mini` a `full` run costs about $0.03. LLM spend so far: $0.36 (dev).
- **P8 (2026-10-12).** Created: `tests/torture.py`, `tests/testtorture.py`, `tests/testfaults.py`,
  `tests/testreproducible.py`, `benchmark/performance.py`, `benchmark/results/performance.json`.
  Edited: `orchestrator.py`, `modelling/engine.py`, `tools/reader.py`, `settings.py`,
  `configuration/settings.json`, `paths.py`, `docs/decisions.md`. Deleted: nothing. What changed: 26
  torture files either give a model and a report or stop with a clear message; injected faults
  (operation, Optuna, model, verify, LLM provider) never stop a run; large files (284,807 and
  1,000,000 rows) finish inside 300 s with at most 2.14 GB of memory, after three time fixes (verify
  on a row sample, a refit time estimate, a reserve for the test stage); the same seed and a warm
  cache give the same plans and scores; a fresh clone installs and passes the tests.
