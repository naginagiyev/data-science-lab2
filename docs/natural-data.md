# Natural data (naturally messy and leaky datasets)

The datasets whose mess or leakage was in the data when it was published (nobody injected it). This
page is the practical reference: what we use, where it comes from, what we checked. The literature
review behind the choice and the measures (why these measures, how reliable they are, what was
rejected) is in the research folder: `DS LAB II/RESEARCH/Topics/natural-data.md`.

Download: `uv run python -m benchmark.natural` writes the files to `cache/natural/<name>/`
(git-ignored) and the manifest with sha256 hashes, sizes, rows and columns to
`benchmark/results/natural.json`. No login, no payment, no API key.

## Datasets

Three kinds, three datasets each (`kind` in `benchmark/natural.py`):

- **mess**: documented natural mess, no documented leak;
- **leak**: documented natural leak, and no mess beyond missing values;
- **both**: documented leak and documented mess beyond missing values.

| Name | Kind | Task (target) | Rows x cols | Source | Licence |
|---|---|---|---|---|---|
| beers | mess | regression (`abv`, from the clean file) | 2,410 x 11 (dirty and clean) | Raha repository (BigDaMa) | ODbL 1.0 + DbCL 1.0 |
| coupon | mess | binary (`Y`) | 12,684 x 26 | UCI 603 | CC BY 4.0 |
| hr-analytics | mess | binary (`LookingForJobChange`) | 19,158 x 13 | OpenML 46935 | CC0 |
| bank-additional | leak | binary (`y`) | 41,188 x 21 | UCI 222 (`bank-additional-full.csv`) | CC BY 4.0 |
| titanic | leak | binary (`survived`) | 1,309 x 14 | OpenML 40945 | Public (OpenML) |
| student-performance | leak | regression (`G3`) | 649 x 33 (`student-por.csv`) | UCI 320 | CC BY 4.0 |
| diabetes130 | both | binary (`readmitted` < 30 days) | 101,766 x 50 | UCI 296 | CC BY 4.0 |
| hotels | both | binary (`IsCanceled`) | H1 40,060 x 31, H2 79,330 x 31 | Data in Brief supplement (Europe PMC) | CC BY 4.0 |
| airbnb-austin | both | regression (`price`) | 11,111 x 90 | Inside Airbnb, Austin, snapshot 2026-09-17 | CC BY 4.0 |

Licences in short: CC BY 4.0 allows any use (also changes and commercial use) if the source is
credited; CC0 and public domain have no conditions; ODbL/DbCL allow any use with credit, and a
changed version of the database itself may only be published under the same licence. We do not
publish any of the files (they stay in `cache/`, outside git).

## Leak labels

A leak label needs evidence. Three levels; only **documented** and **proven** count as ground truth
for leak recall and precision, **suspected** is reported separately.

| Dataset | Column or structure | Level | Evidence | Measured (2026-10-26) |
|---|---|---|---|---|
| bank-additional | `duration` | documented | `bank-additional-names.txt`: "this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. ... should be discarded if the intention is to have a realistic predictive model." | AUC 0.808 alone (depth-3 tree, 5-fold CV); 4 rows with `duration` 0, all `no` |
| titanic | `boat`, `body` | documented | lifeboat and body number are recorded after the sinking (titanic3 variable list) | `boat` filled: 477 of 486 survived, AUC 0.975; `body` filled: 121 of 121 died, AUC 0.575 (few rows) |
| student-performance | `G1`, `G2` | documented | UCI page: "the target attribute G3 has a strong correlation with attributes G2 and G1"; `student.txt`: G1 and G2 are the first and second period grades, G3 the final grade | r = 0.826 and 0.919 with G3; R2 alone 0.676 and 0.810 |
| diabetes130 | repeated patients (L3.2) | documented | Strack et al. 2014 kept one encounter per patient because the rows "could not be considered as statistically independent" | 71,518 patients; 47,021 rows (46%) belong to patients with more than one encounter (up to 40) |
| hotels | `ReservationStatus` | documented | Data in Brief: "Reservation last status" (Canceled, Check-Out, No-Show) | identical to `IsCanceled` in H1 and H2 (AUC 1.0) |
| hotels | `ReservationStatusDate` | documented | "Date at which the last status was set" | AUC 0.578 (H1), 0.629 (H2) alone |
| airbnb-austin | `estimated_revenue_l365d` | proven | revenue / `estimated_occupancy_l365d` equals the price in 100% of the 7,464 rows that have all three; empty exactly where the price is empty | R2 alone only 0.164: it leaks together with occupancy |
| hotels | `Country` | suspected | the authors write that hotels often learn the nationality only at check-in | cancel rate PRT 0.42 vs others 0.16 (H1), 0.65 vs 0.27 (H2); AUC 0.67 and 0.71 |
| airbnb-austin | `price_quote_price_per_night` | suspected | a price quote for one stay (not in the data dictionary) | R2 alone 0.949 with log price |

Not leaks, reported as such: diabetes130 rows that end in death (1,652, all `NO`) or hospice (771)
are out of scope (Strack et al. removed them); `hotels` `BookingChanges`, `AssignedRoomType` and
`DaysInWaitingList` depend on the prediction moment (the authors took the values of the day before
arrival). `coupon` has about 20 rows per survey participant (652 participants, Wang et al. 2017) but
no participant id, so this dependence cannot be split on.

## Measured mess (2026-10-26)

| Dataset | Natural mess found in the downloaded files |
|---|---|
| beers | 4,362 of 26,510 cells (16.45%) differ from the clean file (published: 16%): `ounces` 2,410 (`12.0 oz`, `12.0 OZ.`, `12.0 ounce` vs `12`), `abv` 693 (`0.09%`), `ibu` 1,005 (`N/A` vs empty), `city` and `state` 127 each (`San Francisco CA` with an empty state) |
| coupon | `car` 99.1% empty; constant column `toCoupon_GEQ5min`; 74 duplicate rows; frequency tokens `never`, `less1`, `1~3`, `4~8`, `gt8`; income ranges (`$25000 - $37499`, `$100000 or More`); `50plus`, `below21`; `1d`, `2h` |
| hr-analytics | the text token `nan` (company_size 5,938, gender 4,508); `>20`, `<1`, `>4`, `never`; company size `10/49` next to `<10`, `50-99`, `100-500` (the only range written with `/`) |
| bank-additional | `unknown` (documented as missing): default 8,597, education 1,731, housing and loan 990, job 330, marital 80; `pdays` 999 (documented: not contacted) in 96.3%; 12 duplicate rows |
| titanic | empty cells: body 90.8%, cabin 77.5%, boat 62.9%, home.dest 43.1%, age 20.1%; free text in `name`, `home.dest` |
| student-performance | 15 students with final grade 0 |
| diabetes130 | `?` as missing: weight 96.9%, medical_specialty 49.1%, payer_code 39.6%, race 2.2%; gender `Unknown/Invalid` (3); age as bins `[70-80)`; ICD-9 codes |
| hotels | `NULL` (documented as "not applicable", not missing) in Agent and Company; Country `NULL` (464, 24); `Undefined` in Meal (1,169), DistributionChannel, MarketSegment; values padded with spaces (Agent, Company, Meal); ADR -6.38 (1 row) and 5,400 (H2); zero guests (13, 167); Children `NA` (4); duplicate rows (6,092, 25,902) |
| airbnb-austin | price as text (`$79.74`), 10% empty; 13 columns completely empty in this snapshot (host response, host since, bathrooms, ...); `t`/`f` booleans; `1.5 shared baths`; amenities as JSON lists |

## Measures (P16.3, 2026-10-27)

`uv run python -m benchmark.messmeasures` writes `benchmark/results/natural-measures.json`. The
script imports nothing from `audit/`, so the system never measures its own work. Reference model:
the `lightgbm` baseline (default LightGBM on `pd.read_csv`-style types).

| Dataset | Completeness | Hidden missing cells | Uniqueness | Other documented issues |
|---|---|---|---|---|
| beers | 0.955 | 1,005 (`N/A`) | 1.000 | M1: 16.45% erroneous cells (below) |
| coupon | 0.960 | 0 | 0.994 | 1 column over half empty |
| hr-analytics | 0.917 | 20,668 (`nan`) | 0.997 | |
| bank-additional | 0.985 | 12,718 (`unknown`) | 0.9997 | |
| titanic | 0.790 | 0 | 1.000 | 3 columns over half empty |
| student-performance | 1.000 | 0 | 1.000 | |
| diabetes130 | 0.962 | 192,849 (`?`) | 1.000 | 3 invalid genders; 47,021 rows of repeated patients (71,518 patients) |
| hotels (H2) | 1.000 | 4 (`NA`) | 0.674 | 83,796 `NULL` = not applicable; 167 bookings with no guests |
| airbnb-austin | 0.806 | 0 | 1.000 | 13 empty columns, 15 over half empty |

M1 (beers): 4,362 of 26,510 cells differ from the clean file: 2,961 format or unit, 1,005 missing
token, 142 wrong value, 127 packed value, 127 missing value. M4 (beers, 20 splits): training on the
clean instead of the dirty features lowers the reference RMSE by 0.0019 ABV points (95% interval
[0.0017, 0.0021], p < 0.001): a tiny but consistent effect.

| Leak | Strength alone (5-fold CV) | Inflation of the reference score (5 splits, 95% interval) |
|---|---|---|
| bank `duration` | AUC 0.803 | +0.147 AUC [0.135, 0.159] |
| titanic `boat`, `body` | AUC 0.975, 0.575 | +0.143 AUC [0.121, 0.164] |
| student `G1`, `G2` | R2 0.676, 0.810 | RMSE 1.458 lower with the leak [1.177, 1.739] |
| hotels `ReservationStatus`, `ReservationStatusDate` | AUC 1.000, 0.628 | +0.055 AUC [0.054, 0.057] |
| diabetes130 repeated patients | - | log loss 0.005 lower with a random split than a patient split [-0.013, 0.023] (not significant) |
| airbnb `estimated_revenue_l365d` | R2 0.061 (0.137 with occupancy) | RMSE 1.02 higher with the leak [-0.80, 2.85] (not significant) |

Two documented or proven leaks barely move the reference model: the repeated diabetes patients
and the Airbnb revenue column (the price is a ratio, revenue / occupancy, which trees do not
learn). Leak strength alone also misses partial leaks (titanic `body` covers few rows). These are
limits of the measures, reported as such.

The measures in short:

- **M1 cell error rate** (beers): erroneous cells / all cells against the clean file, per column
  and per error type.
- **M2 quality profile** (all): completeness, validity, uniqueness and consistency ratios (ISO/IEC
  25012 dimensions), with every check taken from the provider's documentation and written down
  before any system runs.
- **M4 impact** (beers): dirty-trained vs clean-trained reference model on the same clean test set.
- **Leak strength** (all labelled leaks): score of the leak column alone (and its missingness).
- **Leak inflation** (all labelled leaks): naive minus leak-free score of a fixed reference model
  (for diabetes130: random vs patient-grouped split).

## Found on the way: a leak in an existing Group C dataset

`employee_salaries` (Group C, not in any pilot grid) has `2016_gross_pay_received` and
`2016_overtime_pay` next to the target `current_annual_salary` (gross pay vs salary: r = 0.889 on
9,128 rows). No source documents this, so it is a suspected leak. No reported result uses this
dataset.

## Benchmark results on the natural data (E12, 2026-10-27)

Full tables: `docs/agentic.md` (grid `agentic-natural`; 300 s, one seed, every system on Linux in WSL).
Mean normalized score: AutoGluon 0.980, LightGBM 0.746, AIDE 0.466, MLZero 0.000, `full` -3.727.

- `full` is best or near best on the mess datasets (beers 1.000, coupon 1.000, hr-analytics 0.998).
- **diabetes130 (`full` log loss 4.66 vs dummy 0.96):** a bug in the system. Audit check F03
  ("inconsistent class labels") took `<30` and `>30` for two spellings of one class, and the
  translator added the protected target step `map_target_labels <30 -> >30`, so the model never
  saw the class `<30`. Open problem; a fix is a method change and needs the user's agreement.
- **Leaks:** `full` dropped the titanic leaks (naive = deployment score) but kept bank `duration`
  (deployment ROC AUC 0.449 vs naive 0.951), hotels `ReservationStatus` (0.672 vs 1.000) and
  student `G1`/`G2` (RMSE 4.60 vs 0.95): the leak-review weakness known since P15.
- **airbnb-austin:** AutoGluon's RMSE of 11.2 (others about 50) most likely comes from the suspected
  leak `price_quote_price_per_night`, which stays in the deployment file.
- Without diabetes130, the mean normalized score of `full` is 0.739 and of AutoGluon 0.977.
