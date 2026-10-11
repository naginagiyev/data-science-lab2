import numpy as np

# issue codes that get a row contrast, codes whose missing values may have a...
# meaning, and the root cause rules on one column
contrast_codes = {"C01", "C02", "C03", "D01", "D02", "D03", "D04", "F07"}
meaning_codes = {"C01", "C02", "C03"}
same_column_rules = {
    "C02": {"B01", "B03", "E04"},
    "C03": {"E09", "D02", "E02"},
}
redundancy_codes = {"E06", "E07", "E08", "A03"}

# counts how often a value appears in a whole column (cached)
def full_count(context, column, value, cache):
    if (column, value) not in cache:
        cache[(column, value)] = int((context.raw[column].astype(str) == value).sum())
    return cache[(column, value)]

# finds columns whose most common value in the issue rows is much rarer in...
# the other rows
def row_contrast(context, issue, rows, cache):
    minimum = context.value("enrichment", "contrast_min_difference")
    results = []
    outside_rows = context.n_rows - len(rows)
    for column in context.raw.columns:
        if column in issue.columns or context.types[column] in {"id", "text", "empty"}:
            continue
        inside = context.raw.loc[rows, column].astype(str)
        top = inside.value_counts()
        if top.empty:
            continue
        value = top.index[0]
        share_in = top.iloc[0] / len(inside)
        if share_in < minimum:
            continue
        outside_count = full_count(context, column, value, cache) - top.iloc[0]
        share_out = float(outside_count / outside_rows) if outside_rows else 0.0
        results.append(
            {
                "column": str(column),
                "value": value,
                "share_in_rows": round(float(share_in), 4),
                "share_elsewhere": round(share_out, 4),
                "difference": share_in - share_out,
            }
        )
    strong = sorted([item for item in results if item["difference"] >= minimum], key=lambda item: -item["difference"])
    top_k = context.value("enrichment", "contrast_top_k")
    return [{key: value for key, value in item.items() if key != "difference"} for item in strong[:top_k]]

# adds row contrasts to issues with row sets; a perfect contrast on missing...
# values sends the issue to agent review
def add_contrasts(context):
    cache = {}
    for issue in context.ledger.issue_list():
        rows = context.ledger.rows.get(issue.id)
        if issue.code not in contrast_codes or rows is None or len(rows) == 0:
            continue
        if len(rows) > context.n_rows / 2:
            continue
        contrast = row_contrast(context, issue, rows, cache)
        if contrast:
            issue.evidence["contrast"] = contrast
            if issue.code in meaning_codes and contrast[0]["share_in_rows"] >= 0.99:
                issue.needs_review = True
                issue.status = "open"

# links symptoms to their root cause on the same column (for example hidden...
# missing values cause numbers stored as text)
def link_same_column(ledger):
    issues = ledger.issue_list()
    for root in issues:
        children = same_column_rules.get(root.code, set())
        for issue in issues:
            if issue.code in children and issue.columns == root.columns:
                ledger.link(issue.id, root.id, "caused_by", f"same column, {root.code} explains {issue.code}")

# makes a co-missing block the parent of the missing value issues of its...
# columns
def link_co_missing(ledger):
    issues = ledger.issue_list()
    for block in [issue for issue in issues if issue.code == "C05"]:
        for issue in issues:
            if issue.code in {"C01", "C02"} and issue.columns[0] in block.columns:
                ledger.link(issue.id, block.id, "caused_by", "column is part of a co-missing block")

# links issues that cover almost the same rows (jaccard >= 0.9)
def link_same_rows(context):
    ledger = context.ledger
    minimum = context.value("enrichment", "same_rows_min_jaccard")
    with_rows = [issue for issue in ledger.issue_list() if len(ledger.rows.get(issue.id, [])) > 0]
    for index, first in enumerate(with_rows):
        for second in with_rows[index + 1 :]:
            if first.caused_by == second.id or second.caused_by == first.id:
                continue
            first_rows = ledger.rows[first.id]
            second_rows = ledger.rows[second.id]
            both = len(np.intersect1d(first_rows, second_rows, assume_unique=True))
            union = len(first_rows) + len(second_rows) - both
            if union and both / union >= minimum:
                ledger.link(second.id, first.id, "same_rows", f"jaccard {both / union:.2f}")

# groups the redundancy issues that share a column
def link_redundancy(ledger):
    issues = [issue for issue in ledger.issue_list() if issue.code in redundancy_codes]
    for index, first in enumerate(issues):
        for second in issues[index + 1 :]:
            if set(first.columns) & set(second.columns):
                ledger.link(second.id, first.id, "group", "redundant columns share a column")

# returns groups of linked issues (connected components); the agent reviews...
# one group per session
def clusters(issues, edges):
    parent = {issue.id: issue.id for issue in issues}
    def find(item):
        while parent[item] != item:
            item = parent[item]
        return item
    for edge in edges:
        parent[find(edge.source)] = find(edge.target)
    groups = {}
    for issue in issues:
        groups.setdefault(find(issue.id), []).append(issue.id)
    return list(groups.values())

# runs all enrichment steps after the detectors
def enrich(context):
    add_contrasts(context)
    link_same_column(context.ledger)
    link_co_missing(context.ledger)
    link_same_rows(context)
    link_redundancy(context.ledger)
