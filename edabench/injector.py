import numpy as np
import pandas as pd

# share of rows changed at each injection strength
strengths = {"low": 0.01, "medium": 0.05, "high": 0.15}

# columns that are fully numeric and have more than 30 values
def numeric_columns(raw, target):
    columns = []
    for column in raw.columns:
        if column == target:
            continue
        numbers = pd.to_numeric(raw[column], errors="coerce")
        if numbers.notna().mean() == 1 and numbers.nunique() > 30:
            columns.append(column)
    return columns

# text columns with 3 to 30 levels and no empty cells
def category_columns(raw, target):
    columns = []
    for column in raw.columns:
        if column == target:
            continue
        numbers = pd.to_numeric(raw[column], errors="coerce")
        levels = raw[column].nunique()
        if numbers.notna().mean() < 0.5 and 2 < levels <= 30 and (raw[column] != "").all():
            columns.append(column)
    return columns

# picks random rows for an injection (at least a minimum count)
def pick_rows(rng, n_rows, share, minimum=3):
    count = max(minimum, int(round(share * n_rows)))
    return np.sort(rng.choice(n_rows, size=min(count, n_rows), replace=False))

# the ground truth record of one injected issue
def truth(code, columns, rows=None, fix=None, kind=None):
    return {
        "code": code,
        "columns": list(columns),
        "rows": [int(row) for row in rows] if rows is not None else None,
        "fix": fix,
        "kind": kind or code,
    }

# empties some cells of a numeric column (expected: C01)
def inject_missing(raw, target, rng, share, used):
    column = next_column(numeric_columns(raw, target), used, rng)
    if column is None:
        return []
    rows = pick_rows(rng, len(raw), share)
    raw.loc[rows, column] = ""
    return [truth("C01", [column], rows, "impute", "missing")]

# writes '?' into some cells of a numeric column (expected: C02 and B01)
def inject_hidden_missing(raw, target, rng, share, used):
    column = next_column(numeric_columns(raw, target), used, rng)
    if column is None:
        return []
    rows = pick_rows(rng, len(raw), share)
    raw.loc[rows, column] = "?"
    return [truth("C02", [column], rows, "replace_values", "hidden_missing"), truth("B01", [column], None, "convert_type", "hidden_missing_type")]

# writes -999 into a non-negative column (expected: C03)
def inject_sentinel(raw, target, rng, share, used):
    candidates = [column for column in numeric_columns(raw, target) if pd.to_numeric(raw[column]).min() >= 0]
    column = next_column(candidates, used, rng)
    if column is None:
        return []
    rows = pick_rows(rng, len(raw), max(share, 0.01))
    raw.loc[rows, column] = "-999"
    return [truth("C03", [column], rows, "replace_values", "sentinel")]

# writes a few very large values (expected: D02)
def inject_outliers(raw, target, rng, share, used):
    column = next_column(numeric_columns(raw, target), used, rng)
    if column is None:
        return []
    numbers = pd.to_numeric(raw[column])
    rows = pick_rows(rng, len(raw), min(share, 0.005), minimum=2)
    spread = float(numbers.quantile(0.75) - numbers.quantile(0.25)) or float(numbers.std()) or 1.0
    values = numbers.max() + spread * rng.uniform(40, 80, len(rows))
    raw.loc[rows, column] = [f"{value:.4f}" for value in values]
    return [truth("D02", [column], rows, "replace_values", "outliers")]

# makes a few values of a positive column negative (expected: D01)
def inject_negatives(raw, target, rng, share, used):
    candidates = [column for column in numeric_columns(raw, target) if pd.to_numeric(raw[column]).min() > 0]
    column = next_column(candidates, used, rng)
    if column is None:
        return []
    rows = pick_rows(rng, len(raw), min(share, 0.005), minimum=2)
    numbers = pd.to_numeric(raw.loc[rows, column])
    raw.loc[rows, column] = [str(-value) for value in numbers]
    return [truth("D01", [column], rows, "replace_values", "negatives")]

# writes a label in other cases and with extra spaces (expected: B05)
def inject_label_variants(raw, target, rng, share, used):
    column = next_column(category_columns(raw, target), used, rng)
    if column is None:
        return []
    tokens = {"", " ", "?", "-", "na", "n/a", "null", "none", "nan", "missing", "unknown"}
    counts = raw[column].value_counts()
    label = next((value for value in counts.index if value.strip().lower() not in tokens), counts.index[0])
    positions = np.where(raw[column] == label)[0]
    rows = np.sort(rng.choice(positions, size=max(2, int(len(positions) * min(share * 2, 0.5))), replace=False))
    variants = [label.upper() + " ", " " + label.lower(), label.title() + "  "]
    variants = [variant for variant in variants if variant != label] or [label + " "]
    raw.loc[rows, column] = [variants[index % len(variants)] for index in range(len(rows))]
    return [truth("B05", [column], rows, "normalize_labels", "label_variants")]

# writes some numbers as '$1,234.00' (expected: B01)
def inject_number_format(raw, target, rng, share, used):
    candidates = [column for column in numeric_columns(raw, target) if pd.to_numeric(raw[column]).abs().max() >= 1000]
    column = next_column(candidates or numeric_columns(raw, target), used, rng)
    if column is None:
        return []
    rows = pick_rows(rng, len(raw), share)
    numbers = pd.to_numeric(raw.loc[rows, column])
    raw.loc[rows, column] = [f"${value:,.2f}" for value in numbers]
    return [truth("B01", [column], rows, "convert_type", "number_format")]

# adds copies of some rows (expected: A04)
def inject_duplicates(raw, target, rng, share):
    rows = pick_rows(rng, len(raw), share / 2)
    copies = raw.iloc[rows].copy()
    raw = pd.concat([raw, copies], ignore_index=True)
    new_rows = np.arange(len(raw) - len(copies), len(raw))
    return raw, [truth("A04", [], new_rows, "drop_rows", "duplicates")]

# adds copies of some rows with another target (expected: A05)
def inject_conflicts(raw, target, rng, share):
    labels = raw[target].unique()
    if len(labels) < 2 or pd.to_numeric(raw[target], errors="coerce").nunique() > 30:
        return raw, []
    rows = pick_rows(rng, len(raw), min(share / 4, 0.0025))
    copies = raw.iloc[rows].copy()
    copies[target] = [labels[(list(labels).index(value) + 1) % len(labels)] for value in copies[target]]
    raw = pd.concat([raw, copies], ignore_index=True)
    new_rows = np.arange(len(raw) - len(copies), len(raw))
    return raw, [truth("A05", [target], new_rows, "drop_rows", "conflicts")]

# adds whole columns with known issues: constant, duplicate column, linear...
# relation and a one-to-one code
def add_columns(raw, target, rng, used):
    found = []
    numbers = [column for column in numeric_columns(raw, target) if column not in used]
    raw["source_system"] = "crm"
    found.append(truth("E01", ["source_system"], None, "drop_column", "constant_column"))
    if numbers:
        base = numbers[0]
        raw[f"{base}_copy"] = raw[base]
        found.append(truth("A03", [base, f"{base}_copy"], None, "drop_column", "duplicate_column"))
    if len(numbers) >= 3:
        first, second = numbers[1], numbers[2]
        total = pd.to_numeric(raw[first]) + pd.to_numeric(raw[second])
        raw["combined_total"] = [repr(float(value)) for value in total]
        found.append(truth("E07", ["combined_total", first, second], None, "drop_column", "linear_relation"))
    categories = [column for column in category_columns(raw, target) if column not in used]
    if categories:
        base = categories[0]
        codes = {label: f"CODE{index:02d}" for index, label in enumerate(sorted(raw[base].unique()))}
        raw[f"{base}_code"] = raw[base].map(codes)
        found.append(truth("E06", [base, f"{base}_code"], None, "drop_column", "one_to_one"))
    return found

# adds a column that leaks the target (expected: G01)
def add_leak(raw, target, rng):
    n_rows = len(raw)
    found = []
    target_numbers = pd.to_numeric(raw[target], errors="coerce")
    if target_numbers.notna().all() and target_numbers.nunique() > 10:
        leak = target_numbers * 1.5 + rng.normal(0, target_numbers.std() * 0.01, n_rows)
        raw["final_assessment"] = [f"{value:.4f}" for value in leak]
    else:
        raw["final_assessment"] = raw[target].map(lambda value: f"outcome {value}")
    found.append(truth("G01", ["final_assessment"], None, "drop_column", "leakage"))
    return found

# picks a random column that no other injection used yet
def next_column(candidates, used, rng):
    free = [column for column in candidates if column not in used]
    if not free:
        return None
    column = free[int(rng.integers(len(free)))]
    used.add(column)
    return column

# the injectors that change one column each
column_injectors = [
    inject_missing,
    inject_hidden_missing,
    inject_sentinel,
    inject_outliers,
    inject_negatives,
    inject_label_variants,
    inject_number_format,
]

# injects all issue kinds into a copy of the data and returns it with the...
# ground truth
def inject(raw, target, strength, seed=0):
    rng = np.random.default_rng(seed)
    share = strengths[strength]
    raw = raw.copy().reset_index(drop=True)
    used = set()
    ground_truth = []
    for injector in column_injectors:
        ground_truth.extend(injector(raw, target, rng, share, used))
    ground_truth.extend(add_columns(raw, target, rng, used))
    ground_truth.extend(add_leak(raw, target, rng))
    raw, found = inject_duplicates(raw, target, rng, share)
    ground_truth.extend(found)
    raw, found = inject_conflicts(raw, target, rng, share)
    ground_truth.extend(found)
    raw["record_id"] = [f"R{index:07d}" for index in rng.permutation(len(raw))]
    ground_truth.append(truth("E03", ["record_id"], None, "set_role", "id_column"))
    raw.attrs["edatool"] = {"raw_text": True}
    return raw, ground_truth
