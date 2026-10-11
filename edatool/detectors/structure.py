import re
import numpy as np
import pandas as pd
from edatool.core.registry import register
from edatool.detectors.common import mask_index, text_values

# A01: reports one-column files, lines with a wrong field count and repeated...
# header rows
@register("A01", mode="both")
def parsing_problems(context, columns):
    info = context.raw.attrs.get("edatool", {})
    if context.raw.shape[1] == 1:
        evidence = {"columns": 1, "delimiter": info.get("delimiter")}
        context.ledger.add("A01", [], "one_column", evidence)
    if info.get("skipped_lines"):
        lines = info["skipped_lines"]
        context.ledger.add("A01", [], "bad_field_count", {"skipped_lines": len(lines), "first_lines": lines[:5]})
    if info.get("short_rows"):
        context.ledger.add("A01", [], "short_rows", {"short_rows": info["short_rows"]})
    header = [str(name) for name in context.raw.columns]
    first = text_values(context, context.raw.columns[0]) if header else pd.Series(dtype=object)
    candidates = context.raw[first == header[0]] if header else context.raw
    if len(candidates):
        repeated = (candidates.astype(str) == header).all(axis=1)
        if repeated.any():
            rows = mask_index(repeated)
            context.ledger.add("A01", [], "repeated_header", {"repeated_header_rows": len(rows)}, rows=rows)

# A02: finds an old index saved as a column (a name like 'Unnamed: 0' or the...
# values 0..n-1)
@register("A02", mode="both")
def index_column(context, columns):
    index_names = set(context.settings["tokens"]["index_column_names"])
    for column in context.raw.columns:
        name = str(column).strip().lower()
        by_name = name in index_names or name.startswith("unnamed:")
        by_values = False
        if context.n_rows > 1 and str(context.raw[column].iloc[0]).strip() in {"0", "1"}:
            values = text_values(context, column)
            numbers = pd.to_numeric(values, errors="coerce").to_numpy()
            start = numbers[0]
            by_values = bool(np.array_equal(numbers, np.arange(start, start + context.n_rows)))
        if by_name or by_values:
            evidence = {"name_like_index": by_name, "equals_row_number": by_values}
            context.ledger.add("A02", [column], "name" if by_name else "values", evidence)

# A03: finds columns with a duplicated name or exactly the same content
@register("A03", mode="both")
def duplicate_columns(context, columns):
    names = [str(column) for column in context.raw.columns]
    for column in names:
        match = re.fullmatch(r"(.*)\.(\d+)", column)
        if match and match.group(1) in names:
            group = [match.group(1), column]
            context.ledger.add("A03", group, "same_name", {"names": group}, fix_params={"column": column})
    groups = {}
    for column in context.raw.columns:
        key = context.column_hash(column).tobytes()
        groups.setdefault(key, []).append(column)
    for group in groups.values():
        if len(group) > 1 and context.n_rows > 0:
            fix_params = {"columns": group[1:]}
            context.ledger.add("A03", group, "same_content", {"identical_columns": group}, fix_params=fix_params)

# A04: finds exact duplicate rows, ignoring id columns
@register("A04", mode="both")
def duplicate_rows(context, columns):
    ids = [column for column in context.raw.columns if context.types[column] == "id"]
    used = [column for column in context.raw.columns if column not in ids]
    if not used:
        return
    duplicated = pd.Series(context.row_hash(used), index=context.raw.index).duplicated()
    if duplicated.any():
        rows = mask_index(duplicated)
        evidence = {"duplicate_rows": len(rows), "ignored_id_columns": ids}
        context.ledger.add("A04", [], "exact", evidence, rows=rows)

# A05: finds rows with the same features but a different target
@register("A05", needs_target=True, mode="both")
def conflicting_duplicates(context, columns):
    features = [
        column
        for column in context.raw.columns
        if column != context.target and context.types[column] != "id"
    ]
    if not features:
        return
    frame = pd.DataFrame(
        {"key": context.row_hash(features), "target": context.column_hash(context.target)},
        index=context.raw.index,
    )
    duplicated_keys = frame["key"][frame["key"].duplicated()].unique()
    if len(duplicated_keys) == 0:
        return
    candidates = frame[frame["key"].isin(duplicated_keys)]
    target_counts = candidates.groupby("key")["target"].nunique()
    conflicting = target_counts[target_counts > 1].index
    if len(conflicting):
        rows = candidates.index[candidates["key"].isin(conflicting).to_numpy()]
        evidence = {"conflicting_groups": len(conflicting), "rows": len(rows)}
        context.ledger.add("A05", [context.target], "same_features_other_target", evidence, rows=rows)

# A06: finds rows where at least half of the cells are missing
@register("A06", mode="both")
def almost_empty_rows(context, columns):
    if context.raw.shape[1] < 2:
        return
    missing_share = context.typed.isna().mean(axis=1)
    empty = missing_share >= context.value("A06", "min_missing_share")
    if empty.any():
        rows = mask_index(empty)
        evidence = {"rows": len(rows), "min_missing_share": context.value("A06", "min_missing_share")}
        context.ledger.add("A06", [], "missing_share", evidence, rows=rows)

# A07: finds column names with spaces at the ends, empty names or special...
# characters
@register("A07", mode="both")
def problem_column_names(context, columns):
    for column in context.raw.columns:
        name = str(column)
        problems = []
        if name.strip() == "":
            problems.append("empty name")
        elif name != name.strip():
            problems.append("leading or trailing spaces")
        if re.search(r"[^\w\s.\-]", name):
            problems.append("special characters")
        if problems:
            context.ledger.add("A07", [column], "+".join(problems), {"name": repr(name), "problems": problems})
