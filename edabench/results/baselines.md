# Baselines on the injection benchmark (medium strength, 10 datasets)

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
