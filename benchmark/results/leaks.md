One leak column per Group D dataset (`benchmark/leaks.py`). In `test_features.csv` the column is empty
(deployment); `test_features_naive.csv` keeps it.

| Dataset | Leak column | Style | Strength |
|---|---|---|---|
| credit-g | `collections_case_opened` | flag | strong |
| Fitness_Club | `checked_in_at_desk` | flag | strong |
| Bank_Customer_Churn | `account_closed_flag` | flag | strong |
| maternal_health_risk | `final_risk_grade` | grade | strong |
| Another-Dataset-on-used-Fiat-500 | `appraised_value_after_sale` | amount | strong |
| Marketing_Campaign | `campaign_response_date` | date | moderate |
| E-CommereShippingData | `delay_compensation_paid` | flag | moderate |
| concrete_compressive_strength | `measured_strength_28d` | amount | moderate |
| healthcare_insurance_expenses | `claim_paid_amount` | amount | moderate |
| diamonds | `insurance_valuation` | amount | moderate |
