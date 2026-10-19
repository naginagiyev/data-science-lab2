import numpy as np
import pandas as pd
from audit.context import ALL_TYPES, mask_index
from tools.columntypes import CATEGORY_TYPES, NUMERIC_TYPES, TOKENS

# one hash per row of one column (numeric columns use their parsed values)
def column_hash(context, column):
    values = context.typed[column] if context.types[column] in NUMERIC_TYPES else context.raw[column]
    return pd.util.hash_pandas_object(values, index=False).to_numpy()

# one hash per row over several columns, to find duplicate rows
def row_hash(context, columns):
    combined = np.zeros(context.n_rows, dtype=np.uint64)
    for column in columns:
        combined = combined * np.uint64(1000003) ^ column_hash(context, column)
    return combined

# A02: an old index saved as a column (a name like 'Unnamed: 0' or the values...
# 0..n-1)
def index_column(context):
    for column in context.raw.columns:
        name = str(column).strip().lower()
        by_name = name in TOKENS["index_column_names"] or name.startswith("unnamed:")
        by_values = False
        if context.n_rows > 1 and str(context.raw[column].iloc[0]).strip() in {"0", "1"}:
            numbers = pd.to_numeric(context.text(column), errors="coerce").to_numpy()
            by_values = bool(np.array_equal(numbers, np.arange(numbers[0], numbers[0] + context.n_rows)))
        if by_name or by_values:
            context.add("A02", [column], "name" if by_name else "values", {"name_like_index": by_name, "equals_row_number": by_values})

# A04: exact duplicate rows, ignoring id columns
def duplicate_rows(context):
    used = [column for column in context.raw.columns if context.types[column] != "id"]
    duplicated = pd.Series(row_hash(context, used), index=context.raw.index).duplicated() if used else pd.Series(dtype=bool)
    if duplicated.any():
        rows = mask_index(duplicated)
        context.add("A04", [], "exact", {"duplicate_rows": len(rows)}, rows=rows)

# A05: rows with the same features but another target (label noise)
def conflicting_rows(context):
    features = [column for column in context.raw.columns if column != context.target and context.types[column] != "id"]
    if not features:
        return
    frame = pd.DataFrame({"key": row_hash(context, features), "target": column_hash(context, context.target)}, index=context.raw.index)
    candidates = frame[frame["key"].isin(frame["key"][frame["key"].duplicated()].unique())]
    conflicting = candidates.groupby("key")["target"].nunique().loc[lambda counts: counts > 1].index
    if len(conflicting):
        rows = candidates.index[candidates["key"].isin(conflicting).to_numpy()]
        context.add("A05", [context.target], "same_features_other_target", {"conflicting_groups": len(conflicting), "rows": len(rows)}, rows=rows)

# E01 and E02: constant columns, and columns where one value covers at least...
# 99% of the rows
def constant_columns(context):
    for column in context.columns_of({"constant"}):
        context.add("E01", [column], "one_value", {"value": str(context.typed[column].dropna().iloc[0])})
    top_share = context.value("E02", "min_top_share")
    for column in context.columns_of(ALL_TYPES - {"id", "constant", "empty", "text"}):
        details = context.details[column]
        if details.get("unique", 0) > 1 + (1 - top_share) * details.get("present", 0):
            continue
        shares = context.typed[column].dropna().value_counts(normalize=True)
        if not shares.empty and shares.iloc[0] >= top_share:
            context.add("E02", [column], "top_share", {"top_value": str(shares.index[0]), "top_share": round(float(shares.iloc[0]), 4)})

# E03: id columns (the leak guard drops them)
def id_columns(context):
    for column in context.columns_of({"id"}):
        details = context.details[column]
        share = details["unique"] / details["present"] if details["present"] else 0.0
        context.add("E03", [column], "unique_share", {"unique": details["unique"], "unique_share": round(share, 4), "confidence": details["confidence"]})

# E04 and E05: categories with too many levels, and free text columns
def wide_categories(context):
    for column in context.columns_of(CATEGORY_TYPES):
        levels = context.details[column].get("unique", 0)
        share_rule = levels > context.value("E04", "max_level_share") * context.n_rows and levels >= context.value("E04", "min_levels")
        if levels > context.value("E04", "max_levels") or share_rule:
            context.add("E04", [column], "levels", {"levels": levels, "rows": context.n_rows}, params={"min_count": context.value("D05", "min_count")})
    for column in context.columns_of({"text"}):
        values = context.typed[column].dropna().astype(str)
        context.add("E05", [column], "long_unique_strings", {"mean_words": round(float(values.str.split().str.len().mean()), 2), "unique": int(values.nunique())})

# E06: category pairs that map one to one (the second one is redundant)
def one_to_one(context):
    groups = {}
    for column in context.columns_of(CATEGORY_TYPES | {"boolean", "numeric_code"}):
        codes, _ = pd.factorize(context.typed[column].astype(object), use_na_sentinel=False)
        if len(np.unique(codes)) >= 2:
            groups.setdefault(codes.astype(np.int64).tobytes(), []).append((column, len(np.unique(codes))))
    for group in groups.values():
        first, levels = group[0]
        for second, _ in group[1:]:
            context.add("E06", [first, second], "one_to_one", {"levels": levels}, params={"column": second})

# E08: numeric pairs with a rank correlation of at least 0.95 (the second one...
# is redundant)
def correlated(context):
    columns = context.columns_of(NUMERIC_TYPES)
    if len(columns) < 2 or len(columns) > context.value("E08", "max_columns"):
        return
    ranks = context.typed.loc[context.sample_rows(), columns].astype(float).rank()
    with np.errstate(all="ignore"):
        correlation = np.corrcoef(ranks.fillna(ranks.mean()).to_numpy(), rowvar=False)
    for first, second in np.argwhere(np.triu(np.abs(np.nan_to_num(correlation)) >= context.value("E08", "min_abs_spearman"), k=1)):
        context.add("E08", [columns[first], columns[second]], "spearman", {"spearman": round(float(correlation[first, second]), 4)}, params={"column": columns[second]})

CHECKS = [index_column, duplicate_rows, conflicting_rows, constant_columns, id_columns, wide_categories, one_to_one, correlated]
