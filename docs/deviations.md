# Deviations

Every change to a pre-registered experiment (from P9.2 on): date, what changed, why, which cells were
re-run.

| Date | Change | Reason | Affected cells |
|---|---|---|---|
| 2026-10-12 | Held-out predictions (§4.10.5): the pre-registration fixes a prediction rule (interpolation between the same system's level-0 and level-3 scores, interval from the interpolation errors on the other 6 pilot datasets) instead of numbers. The numbers are computed from the other `pilot-main` cells and committed before the held-out cells run. | At P9.2 no pilot score exists, so numbers written then could only be guesses; a fixed rule applied before the held-out runs keeps the check honest. | none (the 4 held-out cells run last, with `--holdout`) |
| 2026-10-12 | The full grid `subsets` (E4) lists only `v1`; `aide` is left out. | No AIDE adapter was built (§1.4 allows it only with spare time, and it is never run in this plan). | none (full grid, not run) |
| 2026-10-14 | Harness fix: the scorer reads the training labels of a layout file (title rows, `;`, cp1252) with the edatool loader when plain `pd.read_csv` fails. | The first `reading` run crashed in the scorer before any cell was scored (a harness bug, not a system failure). | the 75 `reading` cells (none had a result; all run after the fix) |
| 2026-10-14 | Harness fixes: the dummy prediction reads decimal commas in the training target of a layout file; a scoring error marks only its cell as failed instead of stopping the grid. | The second `reading` run stopped after 9 cells when the scorer could not parse "39,604880392" (harness bugs, not system failures). | `reading` cells after the first 9 (the 9 written rows are valid and kept) |
| 2026-10-14 | Evaluation fix (`benchmark/repairs.py`): a boolean cell counts as repaired when the prepared values map one to one onto the two original labels. | Columns with word labels ("yes"/"no") restored as 1/0 were scored as not repaired. | the 15 `rules` boolean repair checks, re-run with the fix |
