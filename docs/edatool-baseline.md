# edatool baseline (P4.1, before any change)

Measured on 2026-10-11 with the vendored edatool exactly as copied from `../eda` (style-only changes,
commit `cb31d5a`), deterministic mode, no LLM calls. Commands:

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

# After the P4 changes (P4.6)

Measured on 2026-10-11 after P4.2 to P4.5 (Windows-safe output, loader, B07/B08/B09 detectors, new
tokens, policy and cost) and with four new injected kinds in `edabench/injector.py`: `units` (one unit),
`mixed_units` (kg and g in one column), `packed` (two numeric columns as "a / b") and `mixed_dates` (ISO
and dd/mm/yyyy). The new kinds run after all old injections, with their own random generator and only
on numeric columns that no other injection used, so the old kinds see exactly the same data as in
P4.1. Accepted codes: B07 for units and mixed_units, B08 for packed, B09 for mixed_dates.

## Recall per kind, before and after (deterministic, 30 runs)

| Injected kind | before (P4.1) | after (P4.6) |
|---|---|---|
| missing | 30/30 = 1.000 | 30/30 = 1.000 |
| hidden_missing | 27/27 = 1.000 | 27/27 = 1.000 |
| hidden_missing_type | 27/27 = 1.000 | 27/27 = 1.000 |
| sentinel | 27/27 = 1.000 | 27/27 = 1.000 |
| label_variants | 9/9 = 1.000 | 9/9 = 1.000 |
| constant_column | 30/30 = 1.000 | 30/30 = 1.000 |
| one_to_one | 6/6 = 1.000 | 6/6 = 1.000 |
| leakage | 29/30 = 0.967 | 29/30 = 0.967 |
| duplicates | 30/30 = 1.000 | 30/30 = 1.000 |
| conflicts | 21/21 = 1.000 | 21/21 = 1.000 |
| id_column | 30/30 = 1.000 | 30/30 = 1.000 |
| outliers | 20/24 = 0.833 | 20/24 = 0.833 |
| negatives | 16/16 = 1.000 | 16/16 = 1.000 |
| duplicate_column | 21/21 = 1.000 | 21/21 = 1.000 |
| number_format | 18/18 = 1.000 | 18/18 = 1.000 |
| linear_relation | 12/12 = 1.000 | 12/12 = 1.000 |
| units | - (new kind) | 6/6 = 1.000 |
| mixed_units | - (new kind) | 3/3 = 1.000 |
| mixed_dates | - (new kind) | 3/3 = 1.000 |
| packed | - (new kind) | 2/2 = 1.000 |

- Old kinds: recall per kind is identical; precision on old kinds is 0.8957
  (TP 421, FP 49), the same as P4.1 (0.8957). Within 1 point: yes.
- New kinds: recall 14/14 = 1.000 (target 0.9). Only 14 new-kind injections
  were possible (many datasets have few unused positive numeric columns), so this is a small sample.
- The new columns add 25 secondary findings that the benchmark counts as false positives
  (mostly E04 high cardinality and D05 rare levels on packed columns, B02 on mixed-date columns).
  With them, the overall precision is 0.855 (TP 435, FP 74).
- The first after-run found only 1 of 3 `mixed_units` columns: values like `1.2e+06kg` (scientific
  notation) did not match the number-with-unit pattern. The pattern now accepts an exponent; after the
  fix all 3 are found. Type accuracy on the draft labels is unchanged.

## Injection benchmark, deterministic, after (full table)

Runs: 30 · injected issues: 372 · recall: 0.987 (strict 0.978) · precision: 0.855 (TP 435, FP 74)

| Category   |   Injected |   Recall |   Strict recall |   Row F1 |   Fix action OK |   Cause link OK |
|:-----------|-----------:|---------:|----------------:|---------:|----------------:|----------------:|
| columns    |         78 |    1.000 |           1.000 |    -     |           1.000 |           -     |
| leakage    |         30 |    0.967 |           0.867 |    -     |           1.000 |           -     |
| missing    |         84 |    1.000 |           1.000 |    0.983 |           1.000 |           -     |
| structure  |         72 |    1.000 |           1.000 |    0.806 |           1.000 |           -     |
| types      |         68 |    1.000 |           1.000 |    0.984 |           1.000 |           1.000 |
| validity   |         40 |    0.900 |           0.900 |    0.972 |           1.000 |           -     |

False positives by code: E09 42, E04 11, D05 11, D02 7, B02 3

Mean run time: 1.70 s

## Baselines on the injection benchmark, after (medium strength, 10 datasets)

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
| units                    |     1.000 |             0.000 |           0.000 |      0.000 |
| mixed_units              |     1.000 |             0.000 |           0.000 |      0.000 |
| packed                   |     1.000 |             0.000 |           0.000 |      0.000 |
| mixed_dates              |     1.000 |             0.000 |           0.000 |      0.000 |
| all injected             |     0.984 |             0.416 |           0.304 |      0.152 |
| precision (column-level) |     0.857 |             0.648 |           0.423 |      -     |
