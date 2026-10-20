# Re-run of the tests and the pilot with the P14 code

Started 2026-10-20. Each chunk runs in the background and takes about 45 to 60 minutes; the next
chunk starts only when the previous one has finished, so the machine is otherwise idle and the time
budgets of the runs stay fair. The runner skips cells that already have a row in
`benchmark/results/runs.csv`, so a stopped chunk can simply be started again.

LLM: `gpt-4.1-mini` in the pilot grids, `gpt-4.1-nano` in the smoke grid. The experiments cap was
raised from $2.75 to $3.60 (asked by the user), so the total stays below the $4 optional-stop level
and the $5 hard cap.

| Chunk | What | Cells | LLM cells | Planned time | Status |
|---|---|---|---|---|---|
| 0 | Test suite (fast and slow, no real LLM), then the smoke grid | 208 tests, 36 cells | 8 | ~35 min | done |
| 1 | `pilot-main`: dummy, lightgbm, skrub, flaml | 96 | 0 | ~45 min | done |
| 2 | `pilot-main`: autogluon, generic | 48 | 0 | ~55 min | done |
| 3 | `pilot-main`: rules, full | 48 | 24 | ~55 min | done |
| 4 | `pilot-main`: full+ag; `pilot-leaks` | 44 | 29 | ~50 min | done |
| 5 | `pilot-ablations`, `pilot-natural`; figures and `docs/results.md` | 20 | 11 | ~30 min | done |

Commands (from the project root):

```
uv run pytest -q -m "not llm"
uv run python -m benchmark.runner --grid smoke
uv run python -m benchmark.runner --grid pilot-main --systems dummy lightgbm skrub flaml
uv run python -m benchmark.runner --grid pilot-main --systems autogluon generic
uv run python -m benchmark.runner --grid pilot-main --systems rules full
uv run python -m benchmark.runner --grid pilot-main --systems full+ag
uv run python -m benchmark.runner --grid pilot-leaks
uv run python -m benchmark.runner --grid pilot-ablations
uv run python -m benchmark.runner --grid pilot-natural
uv run python -m benchmark.figures
uv run python -m benchmark.resultsdoc
```

## Log

- **Chunk 0** (12:38 to 13:18 UTC, 40 min): 207 passed, 1 deselected (the real-LLM test) in 23 min. Smoke grid: 36 of 36 cells ok in 17 min; 48 LLM calls (`gpt-4.1-nano`), $0.0103. The slowest `full` cell took 131 s on a 60 s budget (time-out 135 s): the stage limits are not enforced from outside (known weakness).
- **Chunk 1, part 1** (13:20 UTC, stopped by the user before leaving): 48 of 96 cells done, all ok (dummy 24, lightgbm 24). The skrub cell that was running was killed and has no row, so it runs again. Resume with the chunk 1 command; finished cells are skipped.
- **Chunks 1 (rest) to 5** (15:31 to 18:04 UTC, one run without stops, asked by the user): chunk 1 rest 29 min, chunk 2 38 min, chunk 3 41 min, chunk 4 29 min, chunk 5 16 min. All 256 pilot cells ok (pilot-main 216, pilot-leaks 20, pilot-ablations 8, pilot-natural 12); the slowest LLM cell took 147 s on a 120 s budget (time-out 210 s). 246 LLM calls (`gpt-4.1-mini`), $0.3988. Ledger: experiments $2.03, total $2.43. Figures, `tables.md` and `docs/results.md` regenerated.

## Main results (pilot, 8 datasets, one seed)

- RQ1: `full` beats every baseline at levels 2 and 3 (Holm-corrected Wilcoxon p <= 0.001; 14 to 16 wins of 16). `full` vs AutoGluon: +0.153 normalized, bootstrap interval [0.044, 0.327] (before P14: [-0.13, 0.33]). Mean normalized score at level 3: `full` 0.987, `rules` 0.988, `generic` 0.884, AutoGluon 0.842.
- RQ2: at level 0, `full` vs `generic` -0.032 with interval [-0.087, 0.006], not inside the +-0.02 margin, so "no harm" is not shown (as before P14).
- RQ3: `full` found 3 of 5 leaks with no false drop (before P14: 5 of 5). The two misses are the moderate regression leaks `measured_strength_28d` and `claim_paid_amount`: G01 gives no signal (R2 below 0.95) and the leak review answered that nothing leaks.
- RQ4: every system succeeds in 100% of its runs; `full` costs $0.009 per run (before P14: about $0.026).
- RQ5: at level 3 on the 4 ablation datasets, `full` 0.983 vs `full-nogate` 0.947 and `freecode` 0.898; `rules` (no LLM) 0.998.
- Plug-in: `full+ag` vs AutoGluon at levels 2 and 3: p = 0.005, interval [0.023, 0.305].
- E10 natural mess: all four systems within 0.007 roc_auc on the 3 Group C datasets (no gain).
