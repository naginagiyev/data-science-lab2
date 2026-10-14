# Injection benchmark: benchmark-deterministic

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

Mean run time: 2.36 s
