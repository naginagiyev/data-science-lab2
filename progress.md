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
- [x] P1.2 Remove v1 from the root · 2026-10-11 · (next) · v1 logic removed; main.py, paths.py, schemas.py, metrics.json kept for rewriting
- [ ] P1.3 Core files
- [ ] P1.4 Run folder, logging, metrics
- [ ] P1.5 LLM layer
- [ ] P1.6 Orchestrator and CLI skeleton

## P2. Baselines first
- [ ] P2.1 Harness skeleton
- [ ] P2.2 Main-environment baselines (dummy, lightgbm, skrub)
- [ ] P2.3 Separate environments (flaml, autogluon)
- [ ] P2.4 v1 adapter (optional, never run)
- [ ] P2.5 Optional baseline (tabpfn)
- [ ] P2.6 Baseline smoke table

## P3. v2 core without LLM (variant generic)
- [ ] P3.1 Vendor edatool
- [ ] P3.2 Reader and glance
- [ ] P3.3 Splitter
- [ ] P3.4 Typing step and generic stage
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
