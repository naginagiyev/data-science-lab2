# Deviations

Every change to a pre-registered experiment (from P9.2 on): date, what changed, why, which cells were
re-run.

| Date | Change | Reason | Affected cells |
|---|---|---|---|
| 2026-10-12 | Held-out predictions (§4.10.5): the pre-registration fixes a prediction rule (interpolation between the same system's level-0 and level-3 scores, interval from the interpolation errors on the other 6 pilot datasets) instead of numbers. The numbers are computed from the other `pilot-main` cells and committed before the held-out cells run. | At P9.2 no pilot score exists, so numbers written then could only be guesses; a fixed rule applied before the held-out runs keeps the check honest. | none (the 4 held-out cells run last, with `--holdout`) |
| 2026-10-12 | The full grid `subsets` (E4) lists only `v1`; `aide` is left out. | No AIDE adapter was built (§1.4 allows it only with spare time, and it is never run in this plan). | none (full grid, not run) |
