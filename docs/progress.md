# Progress

Legend: [ ] to do · [~] in progress · [x] done · [!] blocked · [-] skipped (reason)

Line format: status, task, date, commit (filled in by the next commit), note.

## Spend
- development: $0.39 of $1.50
- pilot: $1.62 of $2.75
- total: $2.01 (target about $4, hard maximum $5; final, 2026-10-14)

## P0. Setup and safety net
- [x] P0.1 Read the inputs · 2026-10-11 · no files · read buildplan, v2-plan Parts 1, 2, 4, overview.md, eda README and architecture §1-16, edaadapter and the wine log
- [x] P0.2 Git · 2026-10-11 · e0ac275 · git init, ignores, .gitattributes (LF), tag v1-dslab1
- [x] P0.3 Memory files · 2026-10-11 · 4d0dbf8 · CLAUDE.md, progress.md, docs/decisions.md, docs/deviations.md
- [x] P0.4 Freeze v1 · 2026-10-11 · fee24cc · v1 code copied to benchmark/legacy/v1, diff with the root shows no differences
- [x] P0.5 Check the LLM · 2026-10-11 · 3080d50 · nano and mini work with temperature 0 and seed 0; prices unchanged; ledger started ($0.00005)

## P1. Foundation
- [x] P1.1 Environment · 2026-10-11 · 96c2264 · Python 3.12.14 x86_64 (win-amd64), all packages import; versions in decisions.md
- [x] P1.2 Remove v1 from the root · 2026-10-11 · 8fa986d · v1 logic removed; main.py, paths.py, schemas.py, metrics.json kept for rewriting
- [x] P1.3 Core files · 2026-10-11 · ec8bf4d · paths, settings (+ overrides, hash), metrics.json, schemas; settings and style tests pass
- [x] P1.4 Run folder, logging, metrics · 2026-10-11 · 4f64d7d · RunFolder, UTF-8 run.log + ASCII console, metrics with aligned probabilities; missing-class test passes
- [x] P1.5 LLM layer · 2026-10-11 · 16cba91 · client (cache, retries, budgets, logging), OpenAI + fake provider, spend ledger; real nano test passed ($0.00001); cap refusal tested
- [x] P1.6 Orchestrator and CLI skeleton · 2026-10-11 · 2dac1ec · run_stage, Budget, RunState, stub stages, CLI run/predict/report + interactive; ibmchurn run creates context.json and run.log

## P2. Baselines first
- [x] P2.1 Harness skeleton · 2026-10-11 · feba3e5 · runner (resumable, time-out + tree kill, 2 workers, spend stop), scoring (label keys, dummy fallback), splits, materialize; 4 local datasets level 0 seeds 0-2 written; hand-made predictions scored correctly
- [x] P2.2 Main-environment baselines (dummy, lightgbm, skrub) · 2026-10-11 · ada081a, 9bbd076 · all 3 score on the 4 local datasets (benchmark/results/p2-local.csv)
- [x] P2.3 Separate environments (flaml, autogluon) · 2026-10-11 · b7e58b6 · flaml 2.7.0 and autogluon 1.6.3 in own uv envs (no WSL needed); both score on the 4 local datasets with 60 s
- [x] P2.4 v1 adapter (optional, never run) · 2026-10-11 · 71321c2 · v1 env from frozen lock (3.10); headless.py fails cleanly with a refused OPENAI_BASE_URL (slow test)
- [-] P2.5 Optional baseline (tabpfn) · 2026-10-11 · db33db3 · skipped: tabpfn 9.1.0 weights are gated (license acceptance via browser login)
- [x] P2.6 Baseline smoke table · 2026-10-11 · 5862891 · 20/20 cells ok, table below (benchmark/results/p2-smoke.csv)

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
- [x] P3.1 Vendor edatool · 2026-10-11 · cb31d5a · edatool/ and edabench/ copied; edabench paths in root paths.py; CLI audit (--no-agent) works on the 4 local datasets (2.9-22.7 s)
- [x] P3.2 Reader and glance · 2026-10-11 · e7f42f8 · reader (raw view, hard stops), glance + pattern stats; TotalCharges shows ' ':11, kinghouse date is datetime (YYYYMMDDTHHMMSS)
- [x] P3.3 Splitter · 2026-10-11 · 8c7780c · presplit advice -> group/time/stratified/random hold-out + folds; kinghousedata: group split on id
- [x] P3.4 Typing step and generic stage · 2026-10-11 · 01b2f46 · pandas-parity test passes; LightGBM fits on torture headers (tree and linear kinds)
- [x] P3.5 Engine · 2026-10-11 · 2e71939 · dummy/linear/LightGBM/CatBoost + Optuna under a deadline, OOF average of best two, refit; kinghousedata 60 s budget -> 58.7 s
- [x] P3.6 Evaluator, FittedModel, report, predict · 2026-10-11 · 7009e00 · generic run folders complete for the 4 local datasets; predict works on test features
- [x] P3.7 v2 adapter · 2026-10-11 · a198482 · generic is at least as good as lightgbm and skrub on all 4 local datasets (table below)

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
- [x] P4.1 Baseline numbers before changing anything · 2026-10-11 · 6c4cfcc · recall 0.986, precision 0.896 on 358 injected issues (same as before); baselines: ydata 0.433, old eda.py 0.317, cleanlab 0.158 recall; docs/edatool-baseline.md
- [x] P4.2 Windows-safe output · 2026-10-11 · 27ebd2e · SafeConsoleText filter on the edatool logger; cp1252 test: 0 logging errors (2 without the filter)
- [x] P4.3 Loader robustness · 2026-10-11 · e82869a · BOM/UTF-16/cp1252 files, ; tab | delimiters, title rows and footer notes (CSV and Excel), quoted newlines, empty and header-only files: tests pass
- [x] P4.4 Semantic-mess detectors · 2026-10-11 · 1e15729 · B07 units, B08 packed, B09 mixed dates, new tokens, fix vocabulary injected; tests pass; no new issues on the 4 local datasets
- [x] P4.5 Policy and cost · 2026-10-11 · 5bc4388, ba74b88 · creditcardfraud 5k sample, nano: 223,178 -> 14,144 agent tokens (6.3%), both runs $0.0247; batching, session cap, time limits, anonymized skip, F05/F07 no_action
- [x] P4.6 Benchmark the auditor again · 2026-10-11 · 8c0738f · old kinds identical (recall per kind, precision 0.8957); new kinds 14/14; before/after in docs/edatool-baseline.md

## P5. Toolbox and verification (variant rules)
- [x] P5.1 Operations · 2026-10-11 · 481fb95 · 17 column operations + registry/catalog text; contract tests (index, unseen rows, NaN, unicode, empty column, one row, pickle) pass
- [x] P5.2 Row and target operations · 2026-10-11 · 481fb95 · row filters never transform; map_target_labels never merges classes; log_target exact inverse
- [x] P5.3 Translator · 2026-10-11 · 39384af · Appendix C default plans look right on churn, kinghousedata and wine (see decisions.md)
- [x] P5.4 Compiler, data checks, verify integration · 2026-10-11 · 648f750 · wrong unit map -> "parse_units on Weight: 40.0% of values could not be parsed, examples: \"800 g\", ..."; verify resolves C02 after replace_values
- [x] P5.5 Leak guard · 2026-10-11 · aa0de68 · decision-table tests pass (structural, semantic + statistical, semantic only, statistical only, guard off)
- [x] P5.6 Gate · 2026-10-11 · e83d370 · helpful unit parse accepted (positive effect); dropping the strongest feature and a class-removing row filter rejected; validation folds never filtered
- [x] P5.7 Variant rules end to end · 2026-10-11 · 1c4c871 · complete run folders on the 4 local datasets; rules >= generic on ibmchurn and kinghousedata (table below); worse on winequality (explained)

P5.7 table (benchmark mode, 4 local datasets, seed 0, 120 s budget, benchmark/results/p5-rules.csv):

| system | ibmchurn (roc_auc) | kinghousedata (rmse) | winequality (log_loss) | creditcardfraud (roc_auc) |
|---|---|---|---|---|
| generic | 0.8517 | 102687 | 0.8076 | 1.0000 |
| rules | 0.8528 | 100064 | 0.8245 | 1.0000 |

winequality: `rules` drops 240 exact duplicate training rows (a protected step); they look like genuine repeated wines, and without them the class weights change (CV log-loss 0.894 vs 0.876 for `generic`). creditcardfraud: equal within noise.

## P6. Benchmark data
- [x] P6.1 Dataset registry · 2026-10-11 · f1d0856 · 31 OpenML datasets downloaded and checked (ids, targets, tasks, licences); manifest + data cards; no spare needed
- [x] P6.2 Corruptor · 2026-10-11 · bc3a1f3 · 12 types (layout in the writer, leak in P6.3); determinism, nested levels, exact restore, target untouched; one example per type in docs/datasets.md
- [x] P6.3 Leak generator · 2026-10-11 · 501214c · 10 post-event leak columns; G01 finds all 5 strong leaks; naive and deployment test files
- [x] P6.4 Materialize the dev pool · 2026-10-11 · dd8a0d3 · 6 dev datasets x levels 0-3 x seeds 0-2 = 72 cells with truth.json and cells.parquet (cache/benchmark/files/dev, 683 MB, not in git)
- [x] P6.5 Group C · 2026-10-11 · b7c0f68 · 4 of 5 messy as expected; Food_Delivery_Time (cleaned) replaced by employee_salaries; section "Natural mess" in docs/datasets.md

## P7. LLM agents (variant full)
- [x] P7.1 Framer · 2026-10-11 · c04fe65 · fake-provider tests and --ask pass; real nano check: 4/4 targets right
- [x] P7.2 edatool agent integration · 2026-10-11 · 3257ea7 · agent calls in llm.jsonl and the ledger (real nano run: 14 calls, 81k tokens, $0.0085); provider failure falls back to the deterministic audit (test)
- [x] P7.3 Cleaner · 2026-10-11 · 6e3d59e · fake-provider tests pass; real nano run on a level-3 churn sample: valid plan with parse_boolean and map_categories on the first try
- [x] P7.4 Repair loop · 2026-10-11 · 2c7033a · wrong unit map repaired from the structured error (test); capped at 3 attempts; no-repair ablation drops at once
- [x] P7.5 Refine loop · 2026-10-11 · e966740 · a plan with a harmful step is revised from the gate evidence (test); best gated round wins
- [x] P7.6 Sandbox and custom operation · 2026-10-11 · a24f6ec · AST-check, time-out and no-API-key tests pass; custom op via the sandbox; code generation with retry
- [x] P7.7 Free-code variant · 2026-10-11 · ea07999 · fake-provider tests and an end-to-end freecode run pass; nano on churn and kinghouse samples: success and failure logged, gate removes code that does not help
- [x] P7.8 Summary paragraph · 2026-10-11 · 98b7975 · summary in report.md (test and real nano check); number and length checks; provider failure keeps the template report
- [x] P7.9 Dev iteration · 2026-10-12 · c535bfe · 2 free and 2 nano rounds on 12 dev5k cells, then mini on 6: 100% success; mini $0.029 per full run; full >= rules >= generic on the messy files (exceptions explained in decisions.md); dev spend $0.36

## P8. Robustness and performance
- [x] P8.1 Torture suite · 2026-10-12 · df6aeff · 26 files: 20 give a model and a report, 6 stop with a clear message (header only, one row, one class, and three files whose rows are all duplicates in demo mode); the 1M-row file is measured in P8.3
- [x] P8.2 Fault injection · 2026-10-12 · b56ba97 · operation, Optuna, candidate model, verify and provider faults: every run ends with a model and a report (fast and slow tests pass)
- [x] P8.3 Performance · 2026-10-12 · 8955e35 · rules at 300 s: creditcardfraud (284,807 rows) 286 s wall, peak 2.14 GB; 1,000,000-row file 267 s, peak 1.35 GB (after fixes to verify rows, refit time and a finish reserve); full not measured (no cached answers)
- [x] P8.4 Reproducibility test · 2026-10-12 · 19053ef · same seed + warm LLM cache: identical plans, CV and test scores (Optuna off; a time-bounded search depends on machine load)
- [x] P8.5 Fresh-clone test · 2026-10-12 · 8fba464 · clone + uv sync + pytest: 182 passed (192 s cold first run, 140 s warm)

## P9. Pilot experiments (no full benchmark)
- [x] P9.1 Materialize · 2026-10-12 · 997785d · 390 cells (A 180, single types 165, C 15, D 30); manifest benchmark/results/files.json; packed-position bug fixed
- [x] P9.2 Pre-registration · 2026-10-12 · 4c42842 · docs/preregistration.md (pilot datasets, support criteria, held-out cells and prediction rule, exclusion rules); pilot, smoke and full grid files; tag prereg
- [x] P9.3 Smoke grid and freeze · 2026-10-12 · 6d4e8db · smoke 40/40 ok (after the FLAML column-name fix); the 6 full grids pass --check; tag v2-freeze
- [x] P9.4 E7 framer accuracy · 2026-10-12 · eccc83f · 40 datasets: rules 32/40 targets (31/40 with task), rules + mini 36/40 (35/40); $0.025
- [x] P9.5 pilot-main · 2026-10-12 · 07eea01 · 212 cells (4 held-out later), 211 ok; one full run timed out (stage limits not enforced once the budget is used up; kept as a failure); full $0.026 per run; experiments spend $0.83
- [x] P9.6 pilot-leaks · 2026-10-12 · cd347ad · 25/25 ok; full found 5/5 leaks with 0 false drops; rules and full-noleakguard 0/5; $0.13
- [x] P9.7 pilot-ablations · 2026-10-12 · b3c5fab · 20/20 ok; all v2 cleaning variants 0.986-0.995 normalized at level 3 vs generic 0.894, autogluon 0.784; $0.14
- [x] P9.8 pilot-types and E6b reading robustness · 2026-10-14 · 05e1c0f · types 47/48 ok; repair accuracy rules (15 datasets) booleans 0.99 ... impossible 0.00, full (4 datasets) packed 1.00, units 0.69; reading: generic/rules 12/15, lightgbm/skrub/autogluon 0/15
- [x] P9.9 pilot-seed · 2026-10-14 · ee93bff · 96/96 ok (free systems, seed 1); paused and resumed without loss
- [x] P9.10 pilot-natural · 2026-10-14 · 9243d04 · 12/12 ok; full, generic and autogluon within 0.006 roc_auc on the 3 Group C files; $0.12
- [x] P9.11 pilot-budget · 2026-10-14 · 108adc1 · 4/4 ok; the order of full and autogluon is the same at 120 s and 300 s
- [x] P9.12 E8 audit benchmark (final) · 2026-10-14 · 1e12924 · recall 0.987, precision 0.855 (same as after P4); benchmark/results/edabench-final.* and edabench-baselines.*
- [x] P9.13 Held-out cells · 2026-10-14 · 1d8049a · predictions committed first (7230685); 3 of 4 actual scores inside their intervals (projection not unreliable)

## P10. Analysis
- [x] P10.1 analysis.py · 2026-10-12 · fcdf603 · tests on a synthetic results file pass (normalized scores, ranks, Friedman/Nemenyi, Wilcoxon + Holm, bootstrap, reliability, leaks)
- [x] P10.2 figures.py · 2026-10-14 · 6690556 · python -m benchmark.figures regenerates 8 figures (PDF, PNG) and 13 tables (CSV, MD)
- [x] P10.3 docs/results.md · 2026-10-14 · 365beaa · generated by python -m benchmark.resultsdoc from the pilot files, labelled as pilot results
- [x] P10.4 projection · 2026-10-14 · 46b80e7 · docs/projection.md generated: power 0.28 (full vs autogluon), 0.79 (vs skrub), 0.68 (full+ag vs autogluon); held-out 3/4; budget order stable; recommendation: not worth running the full benchmark yet

## P11. Documentation and demo
- [x] P11.1 Code quality pass · 2026-10-14 · e44255c · one dead function removed, two wired (leak table CLI, optional-stop rule in the runner); ruff F clean; tests pass
- [x] P11.2 Documentation · 2026-10-14 · 3a20d4e · README (install, quick start, CLI, variants, sandbox limits, reproduction with time and cost, folder map), docs/architecture.md, benchmark/reproduce.py
- [x] P11.3 Demo · 2026-10-14 · c523ad9 · benchmark/showcase.py; real run in docs/showcase.md: concrete level 3, rmse autogluon 12.93, full 4.11, full+ag 4.20

## P12. Final quality check and handover
- [x] P12.1 Checks · 2026-10-14 · 050c540 · full suite incl. slow 216 passed (18.6 min); secret scan clean (files and history); reproduce --grid smoke works (40 finished cells skipped); style test passes; git status clean
- [x] P12.2 Handover · 2026-10-14 · (this commit) · handover section below

## P13. Cleanup
- [x] P13 Workspace cleanup · 2026-10-16 · 3d27f06 · v1 copy, dev grids, duplicates and unused files removed; one results file (runs.csv), one tables file; plan, progress, study and design inputs moved to docs/; caches 7.6 GB -> 1.4 GB; 191 fast tests pass; regenerated docs have the same numbers (decisions.md)
- [x] P13.3 GitHub · 2026-10-19 · (this commit) · history without the Claude co-author line (ids updated in the docs and runs.csv), branch V2 pushed to GitHub, push after every commit

## P14. Restructuring (one codebase at MSc scale)
- [x] P14.0 Baseline · 2026-10-19 · 96d0b2e · local git, line counts in decisions.md, fast suite 191 passed
- [x] P14.1 Shared foundations · 2026-10-19 · e185a69 · tools/columntypes.py, reader absorbs the loader, splitter absorbs the split advice; types identical on the 4 local datasets
- [x] P14.2 The audit package · 2026-10-19 · d65fd89 · audit/ (29 codes, code only), configuration/audit.json, same issues as edatool on 13 files; verify removed; rules run on ibmchurn ok
- [x] P14.3 Leak review as a normal agent · 2026-10-19 · 11a2196 · agents/leakreview.py (one structured call) replaces the smolagents review; smolagents removed; fake-provider full run on ibmchurn ok
- [x] P14.4 Slim the cleaning checks · 2026-10-19 · b851c42 · custom operation and its prompt removed; freecode in operations/freecode.py; verify already removed in P14.2
- [x] P14.5 Delete edatool and edabench · 2026-10-19 · 5b1571f · both packages, the edabench env and E8 removed; no code reference left; 184 passed, ruff F clean
- [x] P14.6 Small additions to modelling and evaluation · 2026-10-19 · 6c4d7aa · agents/modeller.py (families and time shares, CV decides), permutation importance in the report (demo mode); 187 passed
- [x] P14.7 Variants (11 to 6) · 2026-10-19 · (this commit) · generic, rules, full, full-nogate, freecode, full+ag; constant options removed; 187 passed
- [ ] P14.8 Benchmark at MSc core
- [ ] P14.9 Documentation
- [ ] P14.10 Final checks

## Stretch goals (only after P12)
- [ ] S1 Feature ideas in the CAAFE style
- [ ] S2 LLM swap (RQ6)
- [ ] S3 Full grids (only if the user asks)
- [ ] S4 AutoGluon Assistant baseline

## Handover (2026-10-14)

**What was built.** Agentic AutoML v2: a 12-stage pipeline in which the LLM chooses and sets up tested
cleaning operations and code checks every step (data checks, edatool verify, a cross-validation gate).
Variants `generic`, `rules`, `full`, `full+ag`, `rules+ag`, `freecode` and five ablations; an own
engine and AutoGluon/FLAML backends; a sandbox for generated code; run folders with every decision,
LLM call and cost; the improved auditor edatool (vendored) and its benchmark. A messy-table benchmark
with ground truth: 15 Group A datasets, 12 mess types, 4 levels, natural mess (Group C), a leakage set
(Group D), five baselines, a resumable runner, analysis, figures, a projection and one-command
reproduction. 216 tests pass (fast suite about 2.5 minutes).

**Pilot results** (`docs/results.md`; 8 datasets, one seed for the LLM systems, 120 s per run, a small
sample):
- Messy files (levels 2 and 3): `full` is ahead of every baseline in 11 to 15 of 16 cells; Wilcoxon
  over the 16 cells, Holm-corrected: vs AutoGluon p = 0.029, vs skrub 0.019, vs LightGBM 0.017.
  The 16 cells come from only 8 datasets, so they are not independent; the bootstrap interval over
  datasets of `full` vs AutoGluon at levels 2 and 3 is [-0.13, 0.33] (it crosses 0).
- `full+ag` vs AutoGluon (plug-in) at levels 2 and 3: mean +0.126 normalized, interval [0.02, 0.30].
- `rules` (no LLM) is as good as `full` on these files (mean normalized 0.986 at level 3).
- Clean files (level 0): `full` vs `generic` -0.033 (outside the pre-registered 0.02 margin; no harm
  is not shown).
- Leaks: `full` found 5 of 5 leak columns with no false drop; AutoGluon and LightGBM look almost
  perfect with the leak and fall sharply when it is masked.
- Natural mess (Group C, 3 datasets): no gain (all systems within 0.006).
- Reliability: `full` 23 of 24 runs in `pilot-main` (one time-out); repair accuracy of the cleaning
  stage from 0.99 (booleans) to 0.54 (units) and 0.00 (impossible values); reading robustness: v2
  reads 12 of 15 layout files, the CSV-based baselines 0 of 15; framer 36 of 40 targets right with
  the LLM (32 with rules).

**Projection** (`docs/projection.md`): power of the full benchmark (15 datasets x 3 seeds) for
`full` vs AutoGluon 0.28, vs skrub 0.79, vs LightGBM 0.61, `full+ag` vs AutoGluon 0.68; held-out
check 3 of 4 inside (reliable); budget order the same at 120 s and 300 s. Recommendation: the full
benchmark is not worth running yet for RQ1 as it is; first fix the two weaknesses below, then run a
reduced benchmark (`leaks`, the plug-in comparison) or more datasets.

**LLM spend:** $2.01 of the $5 cap (development $0.39, experiments $1.62).

**What failed or was skipped, and why.**
- Two weaknesses of the frozen system, kept as failures by the pre-registration: stages still do their
  work when their time limit is already used up (2 time-outs of `full`), and a regression target with
  decimal commas (layout files) is not converted (3 reading failures).
- TabPFN skipped (weights gated); v1 and AIDE never run (the plan forbids it; no AIDE adapter); the
  full grids written and checked but not run (the plan leaves them to the user).
- Harness bugs found and fixed during the pilot are in `docs/deviations.md`.

**Limitations.** A small pilot (8 datasets, one seed for the LLM systems, 120 s on an emulated x86
laptop); the controlled mess comes from our own corruptor; the gate is noisy on small tables;
the sandbox is a safety net, not a security boundary.

**Plan for the thesis semester.** Fix the two weaknesses (enforce stage limits; decimal commas in
the target) and log them as deviations; re-run the affected pilot cells; then run `leaks` and a
reduced `main` (all 15 datasets, 3 seeds, levels 0 and 3) on a stronger machine with 300 s;
re-check the projection; add AIDE if time allows; write the thesis from the generated results.

**Files of the last phase (P12).** Created: none. Edited: `progress.md`,
`benchmark/results/spend.json`. Deleted: none.

## Open problems
- Solved in P7.9: nano's semantic name review flagged normal features as post-event leaks (stricter edatool prompt).
- Small tables (about 1,400 rows): the gate's P1 vs P0 comparison is noisy and can keep P0 although every step helps alone (P9 pilot: watch Is-this-a-good-customer-sized datasets).
- Solved in P8.3: large-file overruns (verify sample, refit estimate, finish reserve).
- Found in P9.5 (not fixed, frozen method): stages still do their work when their time limit is already 0, so a run with a slow audit can overrun its budget (1 time-out in 22 full runs). Fix before a full benchmark.
- Found in P9.8 (not fixed, frozen method): a regression target with decimal commas (layout files) is not converted, so the run stops with 0 labeled rows.

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
- **P9 (2026-10-14).** Created: `docs/preregistration.md`, `benchmark/framereval.py`, the pilot, smoke,
  reading and full grid files in `benchmark/grids/`, and the result files `files.json`, `smoke.csv`,
  `framer.csv`, `pilot-main.csv`, `pilot-leaks.csv`, `pilot-ablations.csv`, `pilot-types.csv`,
  `repairs.csv`, `reading.csv`, `pilot-seed.csv`, `pilot-natural.csv`, `pilot-budget.csv`,
  `edabench-final.*`, `edabench-baselines.*`, `holdout-predictions.csv`. Edited: `benchmark/runner.py`
  (mess and budget lists, held-out cells, grid check, per-cell scoring errors),
  `benchmark/materialize.py` (manifest), `benchmark/corruptor.py` (packed position bug),
  `benchmark/scoring.py` (layout files, decimal commas), `benchmark/systems/flaml.py` (column
  names), `docs/decisions.md`, `docs/deviations.md`. Deleted: nothing. What changed: the pilot ran
  (about 560 runs) after the pre-registration (tag `prereg`) and the freeze (tag `v2-freeze`). Main
  findings: `full` found every leak; the cleaning variants beat AutoGluon on the messy level-3 files;
  no gain on naturally messy files; two weaknesses of the frozen system (time limits not enforced once
  the budget is used up; decimal commas in a regression target). Experiments spend: $1.61 of $2.75.
- **P10 and P11 (2026-10-14).** Created: `benchmark/analysis.py`, `benchmark/figures.py`,
  `benchmark/resultsdoc.py`, `benchmark/projection.py`, `benchmark/repairs.py`,
  `benchmark/reproduce.py`, `benchmark/showcase.py`, `docs/results.md`, `docs/projection.md`,
  `docs/architecture.md`, `docs/showcase.md`, `benchmark/results/figures/`, `benchmark/results/tables/`,
  tests (`testanalysis`, `testprojection`, `testrepairs`). Edited: `README.md` (rewritten for v2),
  `orchestrator.py` (stop_after option), `benchmark/runner.py`, `benchmark/leaks.py`,
  `modelling/cv.py`. Deleted: the unused `cross_val_oof`. What changed: one command regenerates every
  figure, table and result document; the projection says the full benchmark is not worth running
  yet for RQ1 (power 0.28 for full vs autogluon), with a held-out check of 3/4 and a stable budget
  order; a new reader can install, run the demo and reproduce a grid from the README.
- **P12 (2026-10-14).** Created: nothing. Edited: `progress.md` (checks, handover, final spend),
  `benchmark/results/spend.json`. Deleted: nothing. What changed: the full test suite with the slow
  tests passes (216 tests), the secret scan is clean, the one-command reproduction works, and the
  handover above sums up the project.
