import hashlib
import numpy as np
from edatool.schemas import Edge, Fix, Issue, RowSet

# builds a stable fingerprint from the code, sorted columns and rule, so the...
# same issue matches across runs
def make_fingerprint(code, columns, rule):
    key = f"{code}|{'|'.join(sorted(columns))}|{rule}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

# the issue store of a run: gives readable ids, merges issues with the same...
# fingerprint and keeps links and row sets
class Ledger:
    def __init__(self, catalog, n_rows):
        self.catalog = catalog
        self.n_rows = n_rows
        self.issues = {}
        self.fingerprints = {}
        self.rows = {}
        self.edges = []
        self.counters = {}
        self.detector = "unknown"

    # adds an issue with the catalog defaults (title, priority, diagnosis,...
    # fix), or merges it into the issue with the same fingerprint
    def add(
        self,
        code,
        columns,
        rule,
        evidence=None,
        rows=None,
        needs_review=None,
        priority=None,
        title=None,
        diagnosis=None,
        fix_action=None,
        fix_params=None,
    ):
        columns = [str(column) for column in columns]
        fingerprint = make_fingerprint(code, columns, rule)
        if fingerprint in self.fingerprints:
            issue = self.issues[self.fingerprints[fingerprint]]
            issue.evidence.update(evidence or {})
            if rows is not None:
                self.set_rows(issue, np.union1d(self.rows.get(issue.id, []), np.asarray(rows)))
            return issue

        entry = self.catalog[code]
        prefix = entry["prefix"]
        self.counters[prefix] = self.counters.get(prefix, 0) + 1
        review = entry["needs_review"] if needs_review is None else needs_review
        params = dict(entry["fix"]["params"])
        if len(columns) == 1:
            params["column"] = columns[0]
        elif columns:
            params["columns"] = columns
        params.update(fix_params or {})
        issue = Issue(
            id=f"{prefix}-{self.counters[prefix]:03d}",
            fingerprint=fingerprint,
            code=code,
            title=title or entry["title"],
            columns=columns,
            evidence=evidence or {},
            detector=f"{self.detector}, rule: {rule}",
            priority=priority or entry["priority"],
            needs_review=review,
            status="open" if review else "confirmed",
            diagnosis=diagnosis or entry["diagnosis"],
            fix=Fix(action=fix_action or entry["fix"]["action"], params=params),
        )
        self.issues[issue.id] = issue
        self.fingerprints[fingerprint] = issue.id
        if rows is not None:
            self.set_rows(issue, rows)
        return issue

    # stores the full row set of an issue and a short RowSet summary with 5...
    # example rows
    def set_rows(self, issue, rows):
        rows = np.unique(np.asarray(rows))
        self.rows[issue.id] = rows
        share = round(len(rows) / self.n_rows, 4) if self.n_rows else 0.0
        examples = [int(row) for row in rows[:5]]
        issue.rows = RowSet(count=len(rows), share=share, examples=examples)

    # adds a link between two issues only once; caused_by also sets the...
    # parent and the children
    def link(self, child_id, parent_id, kind="caused_by", reason=None):
        if child_id == parent_id:
            return
        for edge in self.edges:
            if {edge.source, edge.target} == {child_id, parent_id} and edge.kind == kind:
                return
        child = self.issues[child_id]
        parent = self.issues[parent_id]
        if kind == "caused_by":
            if child.caused_by is not None:
                return
            child.caused_by = parent_id
            parent.children.append(child_id)
        self.edges.append(Edge(source=child_id, target=parent_id, kind=kind, reason=reason))

    # returns the issues in the order they were added
    def issue_list(self):
        return list(self.issues.values())
