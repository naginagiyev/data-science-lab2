# Naturally messy and leaky datasets (research for next step 1)

Status: research and proposal, 2026-10-24. Nothing is downloaded yet; the downloads wait for the
user's approval. Numbers marked "on download" will be measured, not copied.

## 1. What we need and why the current data is not enough

The supervisor asked for datasets whose mess, corruption or leakage is **natural** (it was in the
data when it was published, nobody injected it), and for a **measure of how messy or leaky** each
dataset is that the scientific community accepts.

The current benchmark has three gaps here:

1. Group B mess comes from our own corruptor, and the Group D leaks come from `benchmark/leaks.py`.
   Both are synthetic. Recent work criticises injected errors because "they often rely on randomness
   or explicit patterns, in contrast to latent patterns that manifest in the real world" [14].
2. Group C (natural mess) has no ground truth. Its mess is described by our own audit
   (`docs/datasets.md`, "Natural mess"). If the system's own detector measures the mess that the
   same system then cleans, the measure is **circular**: mess that the audit cannot see is never
   counted, so the system looks better than it is. The measure must be independent of `audit/`.
3. No dataset has a leak that occurs naturally and is documented.

Selection criteria used below:

- the error or leak is in the published data, and a source outside this project documents it
  (the data provider, a peer-reviewed paper, or a gold-standard clean version);
- a tabular task the system supports (binary, multi-class or regression) with a clear target;
- a licence that allows research use, and a download without a login if possible;
- a size that works with the 120 s budget (larger tables get the fixed 30,000-row sample, as now).

## 2. How messiness and leakage are measured in the literature

There is no single accepted "messiness score". Instead, the literature uses a small set of
well-defined measures, and the right one depends on what ground truth exists. I propose to report
the measures below for every dataset. Each one is computed by a separate script, without our
`audit/` package.

### M1. Cell error rate against a gold-standard clean version (the strongest measure)

For a dirty table D and its clean version C with the same rows and columns:

    error rate = |{cells (i, j) : D[i, j] != C[i, j]}| / (rows x columns)

This is the standard measure in error-detection and data-repair research: Abedjan et al. [1] built
their study on real datasets "for which they could obtain both the raw data and the ground truth";
Raha [2], Baran [3], the repair benchmark of Ni et al. [4] and the 2026 address benchmark [14]
define an erroneous cell as a dirty value that does not match the ground-truth value. We also
report it **per column and per error type** (missing token, format only, wrong value), because a
single number hides what kind of mess it is.

- Reliability: **high**. It is an exact count, not an estimate, and anyone can reproduce it from
  the two files.
- Limits: (a) the clean version was made by people and can have errors left in it; (b) a
  formatting difference such as `12.0 oz` vs `12` counts as an error, so we normalise trivial
  differences (for example `12` vs `12.0`) and report "format only" errors separately; (c) only a
  few datasets with real errors have a clean version (Section 3).

### M2. Data-quality profile from the provider's documentation (ISO/IEC 25012 dimensions)

For datasets without a clean version, we measure the dimensions of the ISO/IEC 25012 data-quality
model [5] as simple ratios, the form that Pipino, Lee and Wang [6] and ISO/IEC 25024 [7] use:

- completeness = 1 - (missing cells / all cells), with **hidden** missing tokens (`?`, `NULL`,
  `unknown`, `-1`) counted as missing when the provider's documentation says they mean missing;
- validity = share of values that follow the documented domain (for example, a booking with zero
  guests, or a negative daily rate, breaks the documented meaning);
- uniqueness = 1 - (duplicate rows / rows), plus duplicate **entities** (the same patient in
  several rows);
- consistency = share of values in one documented format (for example, one date format, one unit).

The important rule is that **every check comes from external documentation, and we write it down
before we look at any system output**. This keeps M2 independent of our audit.

- Reliability: **medium**. Each ratio is deterministic and reproducible, and the dimensions are
  standard [5, 8]. But it counts only the issues that are documented, so it is a **lower bound** of
  the real mess. ISO/IEC 25024 gives no thresholds or grades [7], so we report a profile (one value
  per dimension), not a weighted "messiness index". Heinrich et al. [9] list the requirements a
  data-quality metric must meet (for example normalisation and interpretability), and simple ratios
  meet them.

### M3. Error rate estimated from a random sample (optional, for the strongest claim)

Without a clean version, the standard way to estimate the real error rate is to label a random
sample by hand: n randomly chosen cells or rows, two raters, agreement measured with Cohen's kappa,
and a Wilson 95% interval for the error share. With n = 385 the interval is at most +/- 5 points.
This is how label-error studies validated their estimates [10].

- Reliability: **high validity, with known precision** (the interval). The cost is manual work
  (about 2 hours per dataset for 385 cells).

### M4. Effect of the mess on a model (impact, not amount)

CleanML [11] measures what the mess does to a model: train on dirty versus clean data, test on the
same clean test set, repeat over 20 random splits, and use paired t-tests with Benjamini-Yekutieli
control of false discoveries. Where a clean version exists, we can also report how much of the
dirty-to-clean gap a system closes:

    recovered share = (score(system on dirty) - score(reference on dirty))
                      / (score(reference on clean) - score(reference on dirty))

- Reliability: **medium to high**. It depends on the model and the metric, so we fix one reference
  model (default LightGBM, like the current baseline) and report intervals.

### Leakage measures

Leakage is defined as information in the training data that would not be legitimately available
when the model is used [12]. Kapoor and Narayanan [13] give the accepted taxonomy of 8 types; the
ones that matter here are **L2 (illegitimate features)**, **L3.2 (non-independence between training
and test rows)** and **L1.4 (duplicates across the split)**.

- **L-truth: documented ground truth.** A feature, or a row structure, counts as a leak only if the
  data provider or a peer-reviewed paper says so, or if its documented meaning makes it known only
  after the outcome. This gives exact labels, so detection is scored with recall and precision, as
  for Group D today. Reliability: **high**, as long as the prediction moment is fixed (Section 3
  says it for each dataset).
- **L-strength: predictive power of the feature alone.** The cross-validated AUC (classification)
  or R2 (regression) of a model that uses only that feature, or only whether it is missing.
  Reliability: **high**, and easy to interpret.
- **L-inflation: how much the leak inflates the score.** Inflation = naive score - leak-free score,
  with one fixed reference model and bootstrap intervals over repeated splits. For L2 the leak-free
  score drops the leaky feature; for L3.2 it uses a grouped split (by patient) instead of a random
  one. This is the counterfactual design used to show leakage in published work [13].
  Reliability: **medium to high** (it depends on the model, so the model is fixed and reported).

### What I recommend not using

- Counts from our own audit as the "true" mess: they are circular (Section 1). They can still be
  reported as "what the system saw", next to the independent measures.
- One combined messiness index: there are no accepted weights, and the standard gives no grades.
- Label-noise estimates (confident learning [10]) as a main measure: they are estimates that depend
  on the model, and the system has no label-cleaning operations. They can be an extra column.

## 3. Datasets found

Tier 1 has a gold-standard clean version (M1 and M4 apply). Tier 2 has documented natural mess and
natural leaks (M2, L-truth, L-strength, L-inflation). Tier 3 has documented natural mess only (M2).
M3 applies to Tiers 2 and 3 if the user chooses to do it.

| Dataset | Tier | Rows x cols | Task (target) | Natural mess | Natural leak | Licence |
|---|---|---|---|---|---|---|
| Beers | 1 | 2,410 x 11 | regression (`abv`), or multi-class (`style`) | 16% erroneous cells [4]: units in `ounces`, `%` in `abv`, `N/A` tokens, city/state dependency violations | none | not stated (research benchmark); see the note below |
| Movies | 1 | on download (Raha `movies_1`) | regression (`Rating`) | real errors against the clean version: `109 min` runtimes, packed genres, missing values | none | not stated (Magellan repository); see the note below |
| Diabetes 130-US hospitals | 2 | 101,766 x 47 | binary (readmitted < 30 days) | `?` as missing, `weight` 97% missing, payer code and medical specialty about 40-50% missing [15], age as bins `[70-80)`, ICD-9 codes | **L3.2**: one patient has several encounters; the authors kept only the first encounter because rows "could not be considered as statistically independent" [15] | CC BY 4.0 |
| Hotel booking demand | 2 | 119,390 x 32 | binary (`is_canceled`) | `NULL` tokens in `agent` and `company`, `Undefined` categories, impossible rows (zero guests, negative rate): counts on download | **L2**: `reservation_status` is the "reservation last status" (Canceled, Check-Out, No-Show), so it restates the target; `reservation_status_date` is the "date at which the last status was set" [16, 17] | CC BY 4.0 [16] |
| Bank marketing (full, with `duration`) | 2 | 45,211 x 17 | binary (`y`) | `unknown` tokens, `pdays = -1` sentinel | **L2**: the provider writes that `duration` "highly affects the output target (e.g., if duration=0 then y='no')", it is "not known before a call is performed", and it "should be discarded if the intention is to have a realistic predictive model" [18] | CC BY 4.0 |
| Titanic (OpenML 40945, already cached) | 2 | 1,309 x 14 | binary (`survived`) | `cabin` mostly missing, free-text `home.dest`, ticket codes | **L2**: `boat` (lifeboat) and `body` (body number) are recorded after the outcome. Checked on the cached copy: `boat` filled -> 477 of 486 survived; `body` filled -> 121 of 121 died | Public (OpenML) |
| Inside Airbnb listings (one city, one snapshot) | 3 | about 10k-40k x 75 (city dependent) | regression (`price`) | `$1,234.00` prices, `95%` rates, `t`/`f` booleans, `1.5 shared baths`, amenities as JSON lists, dates as text | possible: derived revenue columns (price x nights) in newer snapshots; checked on download | CC BY 4.0 [19] |

Notes per dataset:

- **Beers.** The data was scraped from the web and "manually cleaned by the dataset owner" [4], so
  `dirty.csv` and `clean.csv` hold the same rows (Raha repository [2]). A first look confirms the
  mess: `12.0 oz`, `12.0 OZ.`, `12.0 ounce` in the dirty file are all `12` in the clean one, and
  `0.09%` is `0.09`. The prediction target comes from the clean file, so only the features are
  dirty. The mess types (units, percent signs, missing tokens) match corruptor types 1, 5 and 12.
  This makes Beers a direct check on whether the corruptor is realistic.
- **Movies.** Same repository and format as Beers; the dirty file has units (`109 min`), multi-value
  genre lists and missing values. The size and error rate are measured on download.
- **Licence note (Tier 1).** Neither repository states a data licence. Both datasets are used in
  peer-reviewed benchmarks [2, 3, 4]. For a thesis (research use, no redistribution of the files)
  this is the usual practice; we would cite the sources and not commit the files.
- **Diabetes 130.** Prediction moment: discharge. Rows that end in death or hospice were removed by
  the authors "to avoid biasing our analysis" [15]. These rows are **not** a leak (the discharge
  outcome is known at discharge) but out-of-scope rows, and we report them as such. The leak is
  the repeated patient: a random split puts the same patient in training and test. The large file
  needs the 30,000-row sample; we sample **by patient**, so the group structure stays.
- **Hotel booking.** Prediction moment: booking time. Ground-truth leaks are `reservation_status`
  and `reservation_status_date` (definitional, strong). `booking_changes`, `assigned_room_type` and
  `days_in_waiting_list` change after the booking, so they are reported separately as
  "depends on the prediction moment" and are not counted in recall or precision.
- **Bank marketing.** The project already uses the OpenML version **without** `duration`
  (Group C). With the UCI `bank-full.csv` we get the same data with its documented leak, so the
  leak effect is measured on otherwise identical data.
- **Titanic.** Already in `cache/openml` (spare-C), so it needs no new download.
- **Inside Airbnb.** It is the closest real match to the corruptor's mess types (units, currency,
  percent, booleans, packed values, dates). We pin one snapshot by date and store its sha256 hash,
  because the website replaces old snapshots after about a year.
- **Existing Group C.** I propose to measure HR_Analytics, in_vehicle_coupon, heloc, bank-marketing
  and employee_salaries with M2 as well, so all natural datasets have the same independent measure.

## 4. Checked and not used

| Candidate | Why not |
|---|---|
| Hospital, Tax (Raha, BART) | errors are synthetic (injected with BART) [4] |
| REIN's 14 datasets | errors were injected into real data [20] |
| Flights, Rayyan (Raha) | real errors with a clean version, but no meaningful prediction target |
| CleanML's datasets | real errors, but "we do not have ground-truth versions" except two hand-cleaned sets [11]; Dropbox download, licences not stated. Kept as a reserve |
| Clean Me If You Can (addresses, 2026) | real errors with ground truth [14], but no prediction task |
| Lending Club loans | well-known post-loan leaks, but only on Kaggle (login), licence unclear, about 1.6 GB |
| Natural label-noise sets | no tabular set with real label errors and a verified clean version at a useful size; the system cannot fix labels anyway |

## 5. Reliability summary

| Measure | What it answers | Needs | Reliability | Main weakness |
|---|---|---|---|---|
| M1 cell error rate | how many cells are wrong | clean version | high (exact count) | the clean version can have errors left; format-only errors need care |
| M2 quality profile | how complete, valid, unique and consistent the data is | provider documentation | medium (deterministic, lower bound) | counts only documented issues |
| M3 sample audit | the real error rate, with an interval | about 2 h of labelling per dataset | high validity, +/- 5 points | manual work; two raters needed for kappa |
| M4 impact | how much the mess hurts a model | clean version | medium to high | depends on the model (fixed reference model) |
| L-truth | which features or structures leak | documentation | high | the prediction moment must be fixed |
| L-strength | how strong a single leak is | none | high | does not show leaks that work only together with other features |
| L-inflation | how much the leak inflates the score | none | medium to high | depends on the model (fixed reference model) |

## 6. Plan after approval

1. Download each approved dataset into its own folder under `cache/natural/` (git-ignored). Record
   the source URL, the version or snapshot date, the sha256 hash and the licence in the data cards.
2. Write the measures (M1, M2, M4, L-strength, L-inflation) in a new script under `benchmark/`,
   with no import from `audit/`. The M2 checks and the leak labels go into one JSON file, written
   from the documentation before any system runs on the data.
3. Report one table per dataset in `docs/datasets.md` and the numbers in
   `benchmark/results/natural.json`. No LLM calls (no spend).

## References

1. Z. Abedjan et al. Detecting data errors: where are we and what needs to be done? PVLDB 9(12),
   2016. https://cs.uwaterloo.ca/~ilyas/papers/AbedjanVLDB2016.pdf
2. M. Mahdavi et al. Raha: a configuration-free error detection system. SIGMOD 2019.
   https://doi.org/10.1145/3299869.3324956 (datasets: https://github.com/BigDaMa/raha)
3. M. Mahdavi, Z. Abedjan. Baran: effective error correction via a unified context representation
   and transfer learning. PVLDB 13(11), 2020.
4. W. Ni et al. Automatic data repair: are we ready to deploy? PVLDB 17, 2024.
   https://arxiv.org/abs/2310.00711
5. ISO/IEC 25012:2008. Software engineering - SQuaRE - Data quality model.
6. L. Pipino, Y. Lee, R. Wang. Data quality assessment. Communications of the ACM 45(4), 2002.
7. ISO/IEC 25024:2015. Systems and software engineering - SQuaRE - Measurement of data quality.
8. C. Batini, C. Cappiello, C. Francalanci, A. Maurino. Methodologies for data quality assessment
   and improvement. ACM Computing Surveys 41(3), 2009.
9. B. Heinrich, D. Hristova, M. Klier, A. Schiller, M. Szubartowicz. Requirements for data quality
   metrics. ACM Journal of Data and Information Quality 9(2), 2018.
   https://epub.uni-regensburg.de/36889/
10. C. Northcutt, L. Jiang, I. Chuang. Confident learning: estimating uncertainty in dataset labels.
    JAIR 70, 2021.
11. P. Li, X. Rao, J. Blase, Y. Zhang, X. Chu, C. Zhang. CleanML: a study for evaluating the impact
    of data cleaning on ML classification tasks. ICDE 2021. https://arxiv.org/abs/1904.09483
12. S. Kaufman, S. Rosset, C. Perlich, O. Stitelman. Leakage in data mining: formulation, detection,
    and avoidance. ACM TKDD 6(4), 2012. https://doi.org/10.1145/2382577.2382579
13. S. Kapoor, A. Narayanan. Leakage and the reproducibility crisis in machine-learning-based
    science. Patterns 4(9), 2023. https://doi.org/10.1016/j.patter.2023.100804
14. Clean me if you can: a large collection of real-world addresses for data cleaning benchmarking.
    arXiv 2606.31983, 2026. https://arxiv.org/abs/2606.31983
15. B. Strack et al. Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000
    clinical database patient records. BioMed Research International, 2014.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/ (data: UCI 296,
    https://doi.org/10.24432/C5230J)
16. N. Antonio, A. de Almeida, L. Nunes. Hotel booking demand datasets. Data in Brief 22, 2019.
    https://doi.org/10.1016/j.dib.2018.11.126
17. TidyTuesday 2020-02-11 data dictionary (hotels.csv).
    https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11
18. S. Moro, P. Cortez, P. Rita. A data-driven approach to predict the success of bank
    telemarketing. Decision Support Systems 62, 2014 (data: UCI 222,
    https://doi.org/10.24432/C5K306; the note on `duration` is in the dataset's attribute file).
19. Inside Airbnb, get the data. https://insideairbnb.com/get-the-data/
20. M. Abdelaal, C. Hammacher, H. Schoening. REIN: a comprehensive benchmark framework for data
    cleaning methods in ML pipelines. EDBT 2023. https://arxiv.org/abs/2302.04702
