You check one table for target leakage before a model is trained. The target is `{target}` ({task}).
A column leaks when it is the outcome itself or is recorded after or because of the outcome (for
example cancel_date, refund_issued, days_since_churn, final_result, discharge_status). Values that
exist before the prediction is made are fine, even when they are related to the target: totals and
charges so far, tenure, counts, balances, scores, prices, measurements. When unsure, the column is
fine.
Code found the strong signals below: each of these columns predicts the target almost alone. A strong
signal is not a leak by itself; decide by the meaning of the column.
Return only the columns that leak, each with a short reason, or an empty list. Never invent column
names and never list the target.
Columns:
{columns}
Strong signals:
{signals}
