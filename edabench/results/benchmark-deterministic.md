# Injection benchmark: benchmark-deterministic

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
