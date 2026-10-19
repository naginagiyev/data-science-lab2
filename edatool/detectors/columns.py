import numpy as np
import pandas as pd
from edatool.core.registry import register
from edatool.detectors.common import all_types, category_types, numeric_types, skewness

# E01: columns with only one value
@register("E01", applies_to={"constant"})
def constant_column(context, columns):
    for column in columns:
        value = context.typed[column].dropna().iloc[0]
        context.ledger.add("E01", [column], "one_value", {"value": str(value)})

# E02: columns where one value covers at least 99% of the rows
@register("E02", applies_to=all_types - {"id", "constant", "empty", "text"})
def quasi_constant(context, columns):
    for column in columns:
        details = context.details[column]
        top_share = context.value("E02", "min_top_share")
        if details.get("unique", 0) > 1 + (1 - top_share) * details.get("present", 0):
            continue
        shares = context.typed[column].dropna().value_counts(normalize=True)
        if not shares.empty and shares.iloc[0] >= context.value("E02", "min_top_share"):
            evidence = {"top_value": str(shares.index[0]), "top_share": round(float(shares.iloc[0]), 4)}
            context.ledger.add("E02", [column], "top_share", evidence)

# E03: id columns; a low-confidence id goes to agent review
@register("E03", applies_to={"id"})
def id_like_column(context, columns):
    for column in columns:
        details = context.details[column]
        unique_share = details["unique"] / details["present"] if details["present"] else 0.0
        evidence = {"unique": details["unique"], "unique_share": round(unique_share, 4)}
        review = context.confidence[column] == "low"
        context.ledger.add("E03", [column], "unique_share", evidence, needs_review=review)

# E04: categorical columns with too many levels for one-hot encoding
@register("E04", applies_to=category_types)
def high_cardinality(context, columns):
    for column in columns:
        levels = context.details[column].get("unique", 0)
        too_many = levels > context.value("E04", "max_levels")
        share_rule = levels > context.value("E04", "max_level_share") * context.n_rows
        if too_many or (share_rule and levels >= context.value("E04", "min_levels")):
            evidence = {"levels": levels, "rows": context.n_rows}
            context.ledger.add("E04", [column], "levels", evidence, fix_params={"min_count": context.value("D05", "min_count")})

# E05: free text columns
@register("E05", applies_to={"text"})
def free_text(context, columns):
    for column in columns:
        values = context.typed[column].dropna().astype(str)
        words = values.str.split().str.len()
        evidence = {"mean_words": round(float(words.mean()), 2), "unique": int(values.nunique())}
        context.ledger.add("E05", [column], "long_unique_strings", evidence)

# E06: pairs of columns that map one to one; columns with identical codes are...
# grouped by a hash
@register("E06", applies_to=category_types | {"boolean", "numeric_code"})
def one_to_one_mapping(context, columns):
    groups = {}
    for column in columns:
        values, _ = pd.factorize(context.typed[column].astype(object), use_na_sentinel=False)
        levels = len(np.unique(values))
        if levels >= 2:
            groups.setdefault(values.astype(np.int64).tobytes(), []).append((column, levels))
    for group in groups.values():
        first, levels = group[0]
        for second, _ in group[1:]:
            context.ledger.add("E06", [first, second], "one_to_one", {"levels": levels}, fix_params={"column": second})

# E07: a column that is an exact linear combination of others: SVD on a...
# sample, then confirmed on all rows
@register("E07", applies_to=numeric_types, cost="S")
def exact_linear_relation(context, columns):
    if len(columns) > context.value("E07", "max_columns"):
        return {"*": f"more than {context.value('E07', 'max_columns')} numeric columns"}
    frame = context.typed[columns].astype(float)
    full = frame.dropna()
    sample = full.loc[full.index.intersection(context.sample_rows())]
    std = sample.std()
    used = [column for column in columns if std[column] > 0]
    if len(used) < 3 or len(sample) <= len(used):
        return
    standardized = ((sample[used] - sample[used].mean()) / std[used]).to_numpy()
    _, singular, vectors = np.linalg.svd(standardized, full_matrices=False)
    tolerance = context.value("E07", "svd_tolerance")
    seen = set()
    for position in np.where(singular / singular[0] < tolerance)[0]:
        weights = vectors[position]
        involved = [used[index] for index in np.where(np.abs(weights) > 1e-6 * np.abs(weights).max())[0]]
        dependent = used[int(np.argmax(np.abs(weights)))]
        others = [column for column in involved if column != dependent]
        key = tuple(sorted(involved))
        if not others or key in seen:
            continue
        seen.add(key)
        design = np.column_stack([full[others].to_numpy(), np.ones(len(full))])
        target = full[dependent].to_numpy()
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = np.abs(design @ coefficients - target).max()
        if residual <= context.value("E07", "confirm_tolerance") * max(1.0, np.abs(target).max()):
            terms = {column: round(float(value), 6) for column, value in zip(others, coefficients[:-1])}
            evidence = {"dependent": dependent, "coefficients": terms, "intercept": round(float(coefficients[-1]), 6), "max_residual": float(f"{residual:.3g}")}
            context.ledger.add("E07", [dependent] + others, "svd", evidence, fix_params={"column": dependent})

# E08: column pairs with a rank correlation of at least 0.95 (one numpy...
# correlation matrix)
@register("E08", applies_to=numeric_types, cost="S")
def very_high_correlation(context, columns):
    if len(columns) < 2:
        return
    if len(columns) > context.value("E08", "max_columns"):
        return {"*": f"more than {context.value('E08', 'max_columns')} numeric columns"}
    ranks = context.typed.loc[context.sample_rows(), columns].astype(float).rank()
    ranks = ranks.fillna(ranks.mean()).to_numpy()
    with np.errstate(all="ignore"):
        correlation = np.corrcoef(ranks, rowvar=False)
    minimum = context.value("E08", "min_abs_spearman")
    pairs = np.argwhere(np.triu(np.abs(np.nan_to_num(correlation)) >= minimum, k=1))
    for first, second in pairs:
        evidence = {"spearman": round(float(correlation[first, second]), 4)}
        context.ledger.add("E08", [columns[first], columns[second]], "spearman", evidence, fix_params={"column": columns[second]})

# E09: continuous columns with heavy skew; suggests a log transform when it...
# is possible
@register("E09", applies_to={"numeric_continuous"})
def heavy_skew(context, columns):
    for column in columns:
        values = context.typed[column].dropna().astype(float).to_numpy()
        value = skewness(values)
        if abs(value) <= context.value("E09", "min_abs_skew"):
            continue
        evidence = {"skew": round(value, 3), "min": float(values.min())}
        fix_action = None
        fix_params = {}
        if values.min() > 0:
            evidence["skew_after_log"] = round(skewness(np.log(values)), 3)
        elif values.min() == 0:
            fix_params = {"offset": 1}
            evidence["skew_after_log1p"] = round(skewness(np.log1p(values)), 3)
        else:
            fix_action = "no_action"
            fix_params = {"reason": "negative values; log is not possible"}
        context.ledger.add("E09", [column], "skew", evidence, fix_action=fix_action, fix_params=fix_params)
