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
| 1 | `pilot-main`: dummy, lightgbm, skrub, flaml | 96 | 0 | ~45 min | to do |
| 2 | `pilot-main`: autogluon, generic | 48 | 0 | ~55 min | to do |
| 3 | `pilot-main`: rules, full | 48 | 24 | ~55 min | to do |
| 4 | `pilot-main`: full+ag; `pilot-leaks` | 44 | 29 | ~50 min | to do |
| 5 | `pilot-ablations`, `pilot-natural`; figures and `docs/results.md` | 20 | 11 | ~30 min | to do |

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
