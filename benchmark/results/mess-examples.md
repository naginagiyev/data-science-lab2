Level 3, one type at a time, seed 0; three changed cells per type (before → after).

- **1. units** (credit-g, expected `parse_number`):
  `credit_amount`: `3190` → `$3,190`; `credit_amount`: `4380` → `$4,380`; `credit_amount`: `2325` → `2,325 USD`
- **2. mixed_units** (credit-g, expected `parse_units`):
  `duration_months`: `18` → `540 days`; `duration_months`: `18` → `18 months`; `duration_months`: `24` → `24 months`
- **3. dates** (Marketing_Campaign, expected `parse_datetime`):
  `Dt_Customer`: `2014-02-26` → `26/02/2014`; `Dt_Customer`: `2013-01-02` → `02/01/2013`; `Dt_Customer`: `2014-05-05` → `May 05 2014`
- **4. spellings** (credit-g, expected `map_categories`):
  `checking_status`: `>= 200 DM / salary assignments for >= 1 year` → `>= 200 DM / salary assignments fro >= 1 year`; `checking_status`: `>= 200 DM / salary assignments for >= 1 year` → `>= 200 DM / salary assignments for >= 1 year `; `checking_status`: `0 <= ... < 200 DM` → `0 <= ... < 200 DM `
- **5. fake_missing** (credit-g, expected `replace_values`):
  `residence_since`: `2` → `-999`; `residence_since`: `4` → `-999`; `residence_since`: `2` → `-999`
- **6. impossible** (credit-g, expected `set_missing_where`):
  `duration_months`: `24` → `2400`; `duration_months`: `12` → `-12`; `duration_months`: `9` → `900`
- **7. booleans** (credit-g, expected `parse_boolean`):
  `foreign_worker`: `yes` → `t`; `foreign_worker`: `yes` → `TRUE`; `foreign_worker`: `yes` → `t`
- **8. packed** (credit-g, expected `extract_pattern`):
  `residence_since`: `2` → `2 / 24`; `residence_since`: `4` → `4 / 35`; `residence_since`: `3` → `3 / 32`
- **9. ids** (credit-g, expected `drop_columns`):
  added columns ['customer_id', 'Unnamed: 0'], e.g. `C0000587`
- **12. plain_missing** (credit-g, expected `None`):
  `other_installment_plans`: `bank` → ``; `other_installment_plans`: `none` → ``; `other_installment_plans`: `none` → ``
- **10. layout** (credit-g, level 3: title rows, `;`, decimal commas, cp1252, footer; reader test E6b): first lines:

```
Export of the customer table
Created by the reporting system

checking_status;duration_months;credit_history;credit_purpose;credit_amount;savings_status;employment_since;in
<0 DM;18;existing credits paid duly till now;radio/television;3190;< 100 DM;1 <= ... < 4 years;2;female: divor
```
