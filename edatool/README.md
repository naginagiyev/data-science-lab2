# edatool

A hybrid data-audit tool for tabular data, vendored from the separate `eda` project and improved here
(P4 of `buildplan.md`). In this project it is the "understand" stage of the AutoML pipeline: the only
reader and the only auditor.

- **Code** finds and measures every issue (detectors in 9 categories; every number comes from code).
- **An LLM agent** (smolagents `ToolCallingAgent`) reviews only the issues that need judgment, through
  read-only tools, and must pass a done gate.
- **The output** is a short list of issues, each with evidence, a diagnosis and a recommended fix from a
  fixed vocabulary. The pipeline's translator (`operations/translator.py`) turns the fixes into typed
  operations.

The tool never changes the data. The pipeline applies operations and calls `verify`.

## Install

edatool has no install step of its own here: it is a package inside this project and uses the main
environment (`uv sync` in the project root).

## CLI

```
uv run python -m edatool presplit data/ibmchurn.csv --target Churn --task binary-classification
uv run python -m edatool audit data/ibmchurn.csv --target Churn --task binary-classification --no-agent --out cache/edatool-runs
uv run python -m edatool verify cache/edatool-runs/<run>/report.json --snapshots <folder with steps.json>
```

Task values: `binary-classification`, `multi-class-classification`, `regression`. With the agent, the
CLI reads `OPENAI_API_KEY` from `.env`. Inside the pipeline the agent gets a logged and budgeted model
instead (`llms/agentmodel.py`).

## Python API

```python
from edatool import load_raw, presplit, audit, verify

raw = load_raw(data_path)                       # every value as text, no automatic NA conversion
advice = presplit(raw, target="Churn", task="binary-classification")   # SplitAdvice
report = audit(raw_train, target="Churn", task="binary-classification", run_dir=output_dir)
next_report = verify(report, [("step 1", frame_after_step_1)], {"step 1": ["MISS-001"]}, run_dir=output_dir)
```

`audit` options: `use_agent` (default `True`), `model` (any smolagents `Model`), `settings` (dict or
JSON path, merged over `edatool/config/*.json`), `sample_rows`, `run_dir` (`None` writes nothing). All
functions return Pydantic objects.

## Output (`<run_dir>/<timestamp>-<dataset>/`)

| File | Content |
|---|---|
| `report.json` | meta, schema (type, role, confidence per column), issues, edges, coverage |
| `issues.md` | compact view of the must_fix and should_fix issues (about 2,000 tokens at most) |
| `trace.jsonl` | every agent tool call and result (agent mode only) |
| `timings.json` | time per stage and per detector, agent sessions and tokens |

## Configuration

Every threshold has a one-line reason in `edatool/config/thresholds.json`. Tokens (hidden missing
values, sentinels, placeholder dates, name hints) are in `tokens.json`, the issue catalog (title, id
prefix, priority, review flag, template diagnosis, default fix) in `catalog.json`, and the agent
settings in `agent.json`. Override any of them per call with `settings=`.

## Benchmark of the auditor

The auditor's own benchmark lives in `edabench/` (injected issues, labelled real datasets, baselines):

```
uv run python -m edabench.evaluate types
uv run python -m edabench.evaluate benchmark
uv run python -m edabench.experiments baselines
```

Results are written to `edabench/results/`; run folders go to `cache/edabench/`.
