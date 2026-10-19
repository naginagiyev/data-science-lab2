You choose which model families an AutoML engine searches for one table, and how to split the
search time between them. Code then tunes every chosen family with cross-validation and keeps the
best one (or the average of the best two), so a family you leave out is never tried.
Families:
- linear: logistic or ridge regression on one-hot encoded, scaled features. Fast. Good for small
  tables, many one-hot or text columns, and signals that are close to linear.
- lightgbm: gradient-boosted trees. Fast on large tables. A strong default.
- catboost: gradient-boosted trees with their own handling of categories. Slower. Often the best
  with many categorical columns, but it may not finish many trials on large tables.
Rules:
- Choose at least one of lightgbm and catboost; add linear when it can help.
- Give each chosen family a share of the search time (code normalizes the shares).
- Give a one-sentence reason based on the facts.
Facts:
{profile}
