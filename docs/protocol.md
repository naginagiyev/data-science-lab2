# Experiment protocol (after P14, 2026-10-19)

This protocol replaces the pre-registration of 2026-10-12 (`docs/preregistration.md`, kept in the
git history). It is written after a first pilot of the earlier code (rows in
`benchmark/results/runs-before-p14.csv`), so it is not a pre-registration in the strict sense: the
hypotheses and the analysis below were fixed before that pilot and are only adapted to the six
variants of the restructured code. Every later change goes to `docs/deviations.md` with its date,
reason and affected cells.

## 1. Research questions and hypotheses

| RQ | Question | Hypothesis (not a result) |
|---|---|---|
| RQ1 messy data | On messy tables, does v2 beat classical AutoML? Does the gap grow with the mess level? | Yes, and the gap grows with the level. |
| RQ2 no harm | On clean tables, is v2 as good as the same engine without cleaning? | No significant difference. |
| RQ3 leakage | Does v2 find leaky columns that AutoML uses? | High recall, few false alarms; AutoML collapses at deployment. |
| RQ4 reliability and cost | Success rate, variance, time, cost? | About 100% success at a few cents per run. |
| RQ5 design | Do typed operations and the gate matter? | Typed operations beat free code; the gate prevents harm. |

## 2. Datasets

Pools and OpenML ids: `benchmark/datasets.py` and `docs/datasets.md` (data cards with licences).

- **Dev pool** (development only, never reported as results): ibmchurn, kinghousedata, winequality,
  creditcardfraud, houses, Is-this-a-good-customer.
- **Group A** (clean base, 15 TabArena-v0.1 datasets) and **Group B** (Group A with the corruptor at
  levels 0 to 3, 10 mixed mess types).
- **Group C** (natural mess): HR_Analytics_Job_Change_of_Data_Scientists,
  in_vehicle_coupon_recommendation, heloc, bank-marketing, employee_salaries.
- **Group D** (one post-event leak column each): 10 Group A datasets (`benchmark/leaks.py`).

| Grid | Datasets | Levels | Systems |
|---|---|---|---|
| `pilot-main` | credit-g, Bank_Customer_Churn, E-CommereShippingData, maternal_health_risk, students_dropout_and_academic_success, concrete_compressive_strength, Another-Dataset-on-used-Fiat-500, diamonds | 0, 2, 3 | dummy, lightgbm, skrub, flaml, autogluon, generic, rules, full, full+ag |
| `pilot-leaks` | credit-g, Bank_Customer_Churn, maternal_health_risk, concrete_compressive_strength, healthcare_insurance_expenses | 0 (Group D) | full, rules, autogluon, lightgbm |
| `pilot-ablations` | Bank_Customer_Churn, students_dropout_and_academic_success, concrete_compressive_strength, diamonds | 3 | freecode, full-nogate (with generic, rules and full of `pilot-main`) |
| `pilot-natural` | HR_Analytics_Job_Change_of_Data_Scientists, in_vehicle_coupon_recommendation, heloc | 0 (Group C) | full, generic, autogluon, lightgbm |

Files: one fixed 80/20 outer split per dataset and seed (stratified for classification);
`benchmark/results/files.json` lists every file with its sha256 hash.

## 3. Systems, budgets, seeds

- Baselines: `dummy`, `lightgbm` (default parameters), `skrub` (tabular pipeline), `flaml` (2.7.0),
  `autogluon` (1.6.3, medium_quality, sequential folds).
- v2 variants: `generic`, `rules`, `full`, `full-nogate`, `freecode`, `full+ag`. LLM: `gpt-4.1-mini`
  (temperature 0, seed 0) in every pilot grid; the smoke grid uses `gpt-4.1-nano`.
- Budget: 120 s wall clock per pilot run for every system, 4 threads, at most 2 runs in parallel.
  Seed 0. The full grids use 300 s and up to three seeds.
- Spend: the runner skips LLM cells once the experiments cap or the optional-stop level ($4 total)
  is reached; the total cap is $5.

## 4. Metrics and analysis

- Primary metric per task (fixed): roc_auc (binary), log_loss (multi-class), rmse (regression), on the
  test set, scored by the harness only.
- **Normalized score** per (dataset, level, seed): (score - dummy) / (best - dummy), best = the best
  score of any system in that cell; log_loss and rmse are sign-flipped first. Mean over seeds, then
  over datasets.
- Average rank per level; critical-difference diagram (Friedman + Nemenyi) per level.
- Wilcoxon signed-rank tests of `full` against each baseline, per level and pooled over levels 2 and
  3, Holm-corrected; win/tie/loss counts; bootstrap 95% intervals (10,000 resamples over datasets) of
  the mean normalized-score difference.
- Leak detection from `leaks.json` and `truth.json`: recall, precision and false drops; naive versus
  deployment score.
- Reliability and cost per system: success rate, median seconds, mean tokens and dollars.

## 5. What counts as support

On the pilot these are indications only; the full grids are needed for a confirmatory test.

- **H1 (RQ1)** if `full` beats `autogluon` in a Holm-corrected Wilcoxon test at levels 2 and 3 pooled
  (p < 0.05), and the mean normalized-score difference of `full` vs `autogluon` is larger at level 3
  than at level 0.
- **H2 (RQ2)** if, at level 0, the Wilcoxon test of `full` vs `generic` is not significant and the
  bootstrap 95% interval of the mean normalized-score difference lies inside [-0.02, +0.02].
- **H3 (RQ3)** if, on Group D, leak recall >= 0.8 and precision >= 0.8 for `full`, and the deployment
  score of `autogluon` falls below its naive score by more than the drop of `full`.
- **H4 (RQ4)** if `full` succeeds in >= 99% of its runs and costs <= $0.05 per run on average.
- **H5 (RQ5)** if, at level 3, `full` beats `freecode` and `full-nogate` (mean normalized difference
  > 0 with the bootstrap interval above 0).
- **Plug-in** if `full+ag` beats `autogluon` at levels 2 and 3 pooled (Holm-corrected Wilcoxon,
  p < 0.05).

## 6. Exclusion and failure rules

- A failed or timed-out run counts as a run with the dummy score (normalized score 0) for that
  system; it is not removed (reliability is part of the question).
- A cell where the dummy score equals the best score is left out of the normalized scores (the
  normalization is undefined) and reported.
- A run that fails because of a harness bug (not the system) is re-run after the fix, with an entry
  in `docs/deviations.md`.
- No dataset is removed after the re-run starts.
