import zlib
import numpy as np
import pandas as pd

# one post-event leak column per Group D dataset: its name, its style and its...
# strength (strong: 98% agreement or target x U(0.97, 1.03); moderate: 80%...
# agreement or a correlation of about 0.8)
LEAKS = {
    "credit-g": {"column": "collections_case_opened", "style": "flag", "strength": "strong"},
    "Fitness_Club": {"column": "checked_in_at_desk", "style": "flag", "strength": "strong"},
    "Bank_Customer_Churn": {"column": "account_closed_flag", "style": "flag", "strength": "strong"},
    "maternal_health_risk": {"column": "final_risk_grade", "style": "grade", "strength": "strong"},
    "Another-Dataset-on-used-Fiat-500": {"column": "appraised_value_after_sale", "style": "amount", "strength": "strong"},
    "Marketing_Campaign": {"column": "campaign_response_date", "style": "date", "strength": "moderate"},
    "E-CommereShippingData": {"column": "delay_compensation_paid", "style": "flag", "strength": "moderate"},
    "concrete_compressive_strength": {"column": "measured_strength_28d", "style": "amount", "strength": "moderate"},
    "healthcare_insurance_expenses": {"column": "claim_paid_amount", "style": "amount", "strength": "moderate"},
    "diamonds": {"column": "insurance_valuation", "style": "amount", "strength": "moderate"},
}
AGREEMENT = {"strong": 0.98, "moderate": 0.80}

# a seeded random generator for the leak of one dataset and seed
def leak_generator(name, seed):
    return np.random.default_rng([seed, zlib.crc32(f"leak|{name}".encode("utf-8"))])

# the positive class of a binary target: a yes-like label, else the rarer one
def positive_label(labels):
    counts = labels.value_counts()
    named = [label for label in counts.index if str(label).strip().lower() in {"yes", "true", "1", "good", "bad", "late"}]
    return named[0] if named else counts.index[-1]

# a classification leak: a flag or a date that exists for `agreement` of the...
# positive rows and never for negative rows (it is written only after the...
# event), or a grade code of the class with a share of rows set to another grade
def class_leak(labels, style, agreement, rng):
    labels = labels.astype(str)
    flip = rng.random(len(labels)) > agreement
    if style == "grade":
        classes = sorted(labels.unique())
        codes = {label: f"grade_{index + 1}" for index, label in enumerate(classes)}
        values = labels.map(codes).to_numpy(dtype=object)
        for position in np.flatnonzero(flip):
            others = [codes[label] for label in classes if codes[label] != values[position]]
            values[position] = others[int(rng.integers(len(others)))]
        return values
    positive = labels.to_numpy() == positive_label(labels)
    hit = positive & ~flip
    if style == "date":
        days = rng.integers(0, 365, len(labels))
        dates = (pd.Timestamp("2014-01-01") + pd.to_timedelta(days, unit="D")).strftime("%Y-%m-%d")
        return np.where(hit, dates, "")
    return np.where(hit, "yes", "no")

# a regression leak: target x U(0.97, 1.03) (strong) or target plus noise with...
# a correlation of about 0.8 (moderate)
def amount_leak(target, strength, rng):
    values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    if strength == "strong":
        leak = values * rng.uniform(0.97, 1.03, len(values))
    else:
        leak = values + rng.normal(0, np.nanstd(values) * 0.75, len(values))
    return np.array([f"{value:.2f}" for value in leak], dtype=object)

# adds the leak column of a Group D dataset to its clean raw table (before the...
# split, so train and test get the same process); returns the table and the spec
def add_leak(raw, name, target, task, seed):
    spec = dict(LEAKS[name])
    rng = leak_generator(name, seed)
    if task == "regression":
        values = amount_leak(raw[target], spec["strength"], rng)
    else:
        values = class_leak(raw[target], spec["style"], AGREEMENT[spec["strength"]], rng)
    leaked = raw.copy()
    leaked[spec["column"]] = values
    spec.update(dataset=name, target=target, agreement=AGREEMENT[spec["strength"]] if task != "regression" else None)
    return leaked, spec

# the leak table of docs/datasets.md; python -m benchmark.leaks prints it
def leak_table():
    lines = ["One leak column per Group D dataset (`benchmark/leaks.py`). In `test_features.csv` the column is empty", "(deployment); `test_features_naive.csv` keeps it.", "", "| Dataset | Leak column | Style | Strength |", "|---|---|---|---|"]
    lines += [f"| {name} | `{spec['column']}` | {spec['style']} | {spec['strength']} |" for name, spec in LEAKS.items()]
    return "\n".join(lines)

if __name__ == "__main__":
    print(leak_table())
