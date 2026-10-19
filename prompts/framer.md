You frame a supervised learning task for one table. You see a compact profile of every column and a
ranked list of target candidates from rules.
Return the target (one name from the candidate list), the task (binary-classification,
multi-class-classification or regression), a confidence from 0 to 1, a one-sentence reason, the id
columns (they identify rows) and the time column (it orders rows in time) or null.
Rules:
- Choose the column a business user would most likely want to predict. Follow the rule ranking unless
  the meaning of the columns clearly says otherwise.
- Two distinct target values: binary-classification. An ordered integer score with few levels: decide
  by meaning (a category to predict: multi-class; a measured amount: regression) and say why.
- Never invent column names.
Profile:
{profile}
Rule candidates:
{candidates}
Rule task proposal: {rule_task}
