# edatool baseline (P4.1, before any change)

Measured on 2026-10-11 with the vendored edatool exactly as copied from `../eda` (style-only changes,
commit `72d3773`), deterministic mode, no LLM calls. Commands:

```
uv run python -m edabench.evaluate types --drafts
uv run python -m edabench.evaluate benchmark
benchmark/envs/edabench/.venv/Scripts/python.exe -m edabench.experiments baselines   (PYTHONPATH = project root)
```

The comparison tools (ydata-profiling 4.18.4, cleanlab with Datalab) run in their own environment
`benchmark/envs/edabench` (`benchmark/envs/edabench.txt`), so their pins cannot touch the main
environment. Full result files: `edabench/results/benchmark-deterministic.md|json`,
`edabench/results/baselines.md|json`.

## Type inference accuracy (draft labels)

The 4 label files are drafts (`"reviewed": false`); nobody has reviewed them yet, so these numbers are
only indicative.

```
Label files: 4, reviewed: 0, used: 4
creditcardfraud.csv (draft): 31/31 = 100.0%
ibmchurn.csv (draft): 20/21 = 95.2%
  Contract: label categorical_ordinal, predicted categorical_nominal
kinghousedata.csv (draft): 19/21 = 90.5%
  bathrooms: label numeric_discrete, predicted numeric_continuous
  floors: label numeric_discrete, predicted numeric_continuous
winequality.csv (draft): 12/12 = 100.0%
```

## Injection benchmark, deterministic (10 OpenML datasets x 3 strengths = 30 runs)

Runs: 30 · injected issues: 358 · recall: 0.986 (strict 0.978) · precision: 0.896 (TP 421, FP 49)

| Category   |   Injected |   Recall |   Strict recall |   Row F1 |   Fix action OK |   Cause link OK |
|:-----------|-----------:|---------:|----------------:|---------:|----------------:|----------------:|
| columns    |         78 |    1.000 |           1.000 |    -     |           1.000 |           -     |
| leakage    |         30 |    0.967 |           0.867 |    -     |           1.000 |           -     |
| missing    |         84 |    1.000 |           1.000 |    0.983 |           1.000 |           -     |
| structure  |         72 |    1.000 |           1.000 |    0.806 |           1.000 |           -     |
| types      |         54 |    1.000 |           1.000 |    0.984 |           1.000 |           1.000 |
| validity   |         40 |    0.900 |           0.900 |    0.972 |           1.000 |           -     |

False positives by code: E09 42, D02 7

Mean run time: 1.06 s

This equals the earlier measurement in `../eda/architecture.html` (recall 0.986, precision 0.896 on 358
injected issues), so vendoring changed nothing.

## Baselines on the injection benchmark (medium strength, 10 datasets)

Recall per injected issue kind. cleanlab works on rows: an injected row issue counts as found when at
least half of its rows are flagged with the matching issue type; its column-level precision is not
defined.

Recall per injected issue kind. cleanlab works on rows: an injected row issue counts as found when at least half of its rows are flagged with the matching issue type (outlier, near_duplicate, label, null); its column-level precision is not defined.

| Injected kind            |   edatool |   ydata-profiling |   legacy eda.py |   cleanlab |
|:-------------------------|----------:|------------------:|----------------:|-----------:|
| missing                  |     1.000 |             1.000 |           0.000 |      0.000 |
| hidden_missing           |     1.000 |             0.000 |           0.000 |      0.000 |
| hidden_missing_type      |     1.000 |             0.000 |           1.000 |      0.000 |
| sentinel                 |     1.000 |             0.111 |           0.889 |      0.000 |
| outliers                 |     0.750 |             0.000 |           0.500 |      0.375 |
| negatives                |     1.000 |             0.000 |           0.000 |      0.000 |
| label_variants           |     1.000 |             0.000 |           0.000 |      0.000 |
| number_format            |     1.000 |             0.000 |           1.000 |      0.000 |
| duplicates               |     1.000 |             0.000 |           0.000 |      1.000 |
| conflicts                |     1.000 |             0.000 |           0.000 |      0.857 |
| id_column                |     1.000 |             1.000 |           0.000 |      0.000 |
| constant_column          |     1.000 |             1.000 |           0.000 |      0.000 |
| duplicate_column         |     1.000 |             1.000 |           1.000 |      0.000 |
| linear_relation          |     1.000 |             1.000 |           0.750 |      0.000 |
| one_to_one               |     1.000 |             0.000 |           0.000 |      0.000 |
| leakage                  |     1.000 |             1.000 |           0.000 |      0.000 |
| all injected             |     0.983 |             0.433 |           0.317 |      0.158 |
| precision (column-level) |     0.905 |             0.683 |           0.446 |      -     |
