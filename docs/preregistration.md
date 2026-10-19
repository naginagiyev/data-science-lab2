# Pre-registration (pilot, 2026-10-12)

Written before any pilot grid ran, after the development phases P0 to P8. Every later change goes
to `docs/deviations.md` with its date, reason and affected cells. The pilot estimates what the full
benchmark would show; it does not give final results.

## 1. Research questions and hypotheses

| RQ | Question | Hypothesis (not a result) |
|---|---|---|
| RQ1 messy data | On messy tables, does v2 beat classical AutoML and code-writing agents? Does the gap grow with the mess level? | Yes, and the gap grows with the level. |
| RQ2 no harm | On clean tables, is v2 as good as the same engine without the agent? | No significant difference. |
| RQ3 leakage | Does v2 find leaky columns that AutoML uses? | High recall, few false alarms; AutoML collapses at deployment. |
| RQ4 reliability and cost | Success rate, variance, time, cost? | About 100% success; lower cost and variance than code agents. |
| RQ5 design | Which parts matter (typed operations, gate, repair, audit review, value samples)? | Typed operations beat free code; the gate prevents harm. |

## 2. Datasets and pools

Pools and OpenML ids: `benchmark/datasets.py` and `docs/datasets.md` (data cards with licences).

- **Dev pool** (development only, never reported as results): ibmchurn, kinghousedata, winequality,
  creditcardfraud, houses, Is-this-a-good-customer.
- **Group A** (clean base, 15 TabArena-v0.1 datasets): credit-g, Fitness_Club, Marketing_Campaign,
  Bank_Customer_Churn, E-CommereShippingData, online_shoppers_intention, maternal_health_risk,
  students_dropout_and_academic_success, SDSS17 (fixed sample of 30,000 rows),
  concrete_compressive_strength, healthcare_insurance_expenses, airfoil_self_noise,
  Another-Dataset-on-used-Fiat-500, miami_housing, diamonds (fixed sample of 30,000 rows).
- **Group B**: Group A with the corruptor at levels 0 to 3 (mixed mess: types 1 to 9 and 12 of
  Appendix E1).
- **Group C** (natural mess): HR_Analytics_Job_Change_of_Data_Scientists,
  in_vehicle_coupon_recommendation, heloc, bank-marketing, employee_salaries.
- **Group D** (one realistic post-event leak each, Appendix E2): credit-g, Fitness_Club,
  Marketing_Campaign, Bank_Customer_Churn, E-CommereShippingData, maternal_health_risk,
  concrete_compressive_strength, healthcare_insurance_expenses, Another-Dataset-on-used-Fiat-500,
  diamonds.

**Pilot datasets** (chosen before any result, from the Appendix D marks: all three task types, small
and large tables):

| Use | Datasets |
|---|---|
| `pilot-main`, `pilot-seed` (Group A/B) | credit-g, Bank_Customer_Churn, E-CommereShippingData (binary); maternal_health_risk, students_dropout_and_academic_success (multi-class); concrete_compressive_strength, Another-Dataset-on-used-Fiat-500, diamonds (regression) |
| `pilot-leaks` (Group D) | credit-g, Bank_Customer_Churn, maternal_health_risk, concrete_compressive_strength, healthcare_insurance_expenses |
| `pilot-ablations` (level 3) | Bank_Customer_Churn, students_dropout_and_academic_success, concrete_compressive_strength, diamonds |
| `pilot-types` (trained cells, 4 key types: units, mixed units, packed, spellings) | Bank_Customer_Churn, students_dropout_and_academic_success, Another-Dataset-on-used-Fiat-500, diamonds |
| `pilot-natural` (Group C) | HR_Analytics_Job_Change_of_Data_Scientists, in_vehicle_coupon_recommendation, heloc |
| `pilot-budget` (level 3, 300 s) | Bank_Customer_Churn, diamonds |

Files: one fixed 80/20 outer split per dataset and seed (stratified for classification);
`benchmark/results/files.json` lists every file with its sha256 hash.

## 3. Systems, versions, budgets, seeds

- Baselines: `dummy`, `lightgbm` (4.7.0, default parameters), `skrub` (0.11.0, TableVectorizer +
  HistGradientBoosting), `flaml` (2.7.0), `autogluon` (1.6.3, medium_quality, sequential folds).
- v2 variants (this repository at the tag `v2-freeze`): `generic`, `rules`, `full`, `full+ag`, and the
  ablations `freecode`, `full-nogate`, `full-norepair`, `full-noreview`, `full-nosamples`,
  `full-noleakguard`. Own engine: scikit-learn 1.9.1, LightGBM 4.7.0, CatBoost 1.2.10, Optuna 5.0.0.
  LLM: `gpt-4.1-mini` (temperature 0, seed 0) for every pilot grid; the smoke grid uses
  `gpt-4.1-nano`. Python 3.12 x86_64 under emulation on Windows on ARM (Snapdragon, 8 cores, 16 GB).
- Budget: 120 s wall clock per run for every system (300 s in `pilot-budget`), 4 threads, at most 2
  runs in parallel. Seeds: 0 (`pilot-seed`: 1, free systems only).
- Spend: experiments cap $2.75, total cap $5. If the mean `full` run costs more than $0.06 after the
  first 10 `full` runs, `pilot-ablations` and `pilot-types` are reduced to 2 datasets (§4.6).

## 4. Metrics and analysis plan

- Primary metric per task (fixed): roc_auc (binary), log_loss (multi-class), rmse (regression), on the
  test set, scored by the harness only.
- **Normalized score** per (dataset, level, seed): (score - dummy) / (best - dummy), best = the best
  score of any system in that cell; log_loss and rmse are sign-flipped first. Mean over seeds, then
  over datasets.
- Average rank per level; critical-difference diagram (Friedman + Nemenyi) per level.
- Wilcoxon signed-rank tests of `full` against each baseline, per level and pooled over levels 2 and
  3, Holm-corrected; win/tie/loss counts; bootstrap 95% intervals (10,000 resamples over datasets) of
  the mean normalized-score difference.
- Decision quality from `decisions.jsonl` and `truth.json`: framer accuracy; leak precision, recall
  and F1; false leak drops on clean data; repair accuracy per mess type; gate acceptance of LLM steps;
  repair-loop success; fallbacks per stage.
- Reliability and cost per system: success rate, median seconds, mean tokens and dollars, standard
  deviation over seeds.

## 5. Projection method (§4.10)

`benchmark/projection.py`, run after the pilot: effect estimates with bootstrap intervals for `full`
vs `autogluon`, `skrub`, `lightgbm`, `generic` and `full+ag` vs `autogluon` per level; seed variance
from the seed-1 runs of the free systems added to the one-seed `full` estimates; power for the full
benchmark by simulation (15 datasets, 3 seeds, dataset effects drawn from the pilot mean and spread,
seed noise added, the pre-registered Holm-corrected Wilcoxon test, 2,000 repetitions); the held-out
check (section 7); the budget check (does the order of `full` and `autogluon` change between 120 s
and 300 s?); the mechanism check (gains linked to repair accuracy per mess type). Output:
`docs/projection.md` with one recommendation per RQ.

## 6. What counts as support (full benchmark; the pilot only projects these)

- **H1 (RQ1)** supported if `full` beats `autogluon` in a Holm-corrected Wilcoxon test at levels 2
  and 3 pooled (p < 0.05), and the mean normalized-score difference of `full` vs `autogluon` at level
  3 is larger than at level 0 (bootstrap 95% interval of the difference of differences above 0).
- **H2 (RQ2)** supported if, at level 0, the Wilcoxon test of `full` vs `generic` is not significant
  and the bootstrap 95% interval of the mean normalized-score difference lies inside [-0.02, +0.02].
- **H3 (RQ3)** supported if, on Group D, leak recall >= 0.8 and precision >= 0.8 for `full`, at most
  1 false leak drop per 10 clean (Group A level 0) datasets, and the deployment score of `autogluon`
  falls below its naive score by more than the drop of `full`.
- **H4 (RQ4)** supported if `full` succeeds in >= 99% of its runs and costs <= $0.05 per run on
  average with `gpt-4.1-mini`.
- **H5 (RQ5)** supported if, at level 3, `full` beats `freecode` and `full-nogate` (mean normalized
  difference > 0 with the bootstrap interval above 0), and `full-nogate` loses to `full` on at least
  one dataset where a harmful step was proposed.
- **Contribution 4** (plug-in) supported if `full+ag` beats `autogluon` at levels 2 and 3 pooled
  (Holm-corrected Wilcoxon, p < 0.05).

## 7. Held-out cells (§4.10.5)

Four cells of `pilot-main` run last: **E-CommereShippingData** and **concrete_compressive_strength**
at **level 2**, seed 0, for `full` and `autogluon`.

Prediction rule (fixed now; the numbers are computed from the other pilot-main cells and committed
to this file before the held-out cells run):

- point prediction for system s on dataset d at level 2: s(d, 0) + 2/3 * (s(d, 3) - s(d, 0)), the
  linear interpolation between the same system's level-0 and level-3 test scores on d;
- 95% interval: the point prediction times (1 + q), where q runs from the 2.5% to the 97.5% quantile
  of the relative interpolation errors (actual - interpolated) / |interpolated| of the same system at
  level 2 on the other 6 pilot datasets.

The projection is called unreliable if fewer than 3 of the 4 actual scores fall inside their
intervals.

## 8. Exclusion and failure rules

- A failed or timed-out run counts as a run with the dummy score (normalized score 0) for that
  system; it is not removed (reliability is part of the question).
- A cell where the dummy score equals the best score is left out of the normalized scores (the
  normalization is undefined) and reported.
- A run that fails because of a harness bug (not the system) is re-run after the fix, with an entry
  in `docs/deviations.md`.
- No dataset is removed after the pilot starts. After the freeze, method changes because of pilot
  results are not allowed; bug fixes are logged in `docs/deviations.md` and every affected cell is
  re-run.

### 7.1 Held-out predictions (committed on 2026-10-14, before the held-out cells ran)

Computed with the rule above by `python -m benchmark.projection --predict` from the other
`pilot-main` cells (`benchmark/results/holdout-predictions.csv`). The interval of `full` uses 5 other
datasets, because its students_dropout level-3 run failed and has no score.

| dataset | system | metric | prediction | 95% interval |
|---|---|---|---|---|
| E-CommereShippingData | full | roc_auc | 0.7377 | [0.6954, 0.7441] |
| E-CommereShippingData | autogluon | roc_auc | 0.7197 | [0.6758, 0.7566] |
| concrete_compressive_strength | full | rmse | 3.8314 | [3.6116, 3.8646] |
| concrete_compressive_strength | autogluon | rmse | 9.7706 | [9.1749, 10.2716] |
