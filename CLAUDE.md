# Agentic AutoML v2 (DS LAB 2, MSc thesis)

- The work order is `docs/buildplan.md`. The status is `docs/progress.md`. Read `docs/progress.md`
  first in every session, then the current phase of `docs/buildplan.md`.
- Judgment calls go to `docs/decisions.md`; changes to pre-registered experiments go to
  `docs/deviations.md`.
- Environment: Windows on ARM, x86_64 Python 3.12 through uv. Use `uv run ...`. The path has spaces:
  quote it. Reconfigure stdout to UTF-8 in entry points.
- Never: push, rewrite git history, print or commit `.env`, invent results or citations, call a real
  LLM in the fast tests, run a full benchmark grid in the foreground, run `ruff format` or `black`.
- Style: the global coding rules apply (one empty line between definitions, imports sorted by line
  length, double quotes, pathlib paths in `paths.py`, lowercase file names without underscores).
  `docs/buildplan.md` §0.6 lists the overrides (error handling, logging, tests, comments).
- LLM spend: target about $4, hard maximum $5 for the whole plan (`configuration/settings.json` → `spend`; the
  client enforces it). Use `gpt-4.1-nano` while developing. Never start the full benchmark grids.
- Commit after every task: `P<phase>.<task>: <what changed>`.

<!-- code-study -->
## Study folder
This project has a `study/` folder that explains the code in plain English.
Whenever you add, edit, move, or remove code, use the `code-study` skill to update the matching pages in `study/content/` and rebuild.
Never change source code while doing study work.
