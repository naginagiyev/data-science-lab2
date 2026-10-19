import json
import threading
import numpy as np
import pandas as pd
from smolagents import Tool
from edatool.agent.rules import evaluate_rule
from edatool.detectors.common import numeric_types
from scipy.stats import chi2_contingency, spearmanr
from edatool.core.progress import arguments_text, logger, short

# gives tool calls unique ids (c1, c2, ...) and collects the trace; all...
# sessions share it
class CallCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0
        self.trace = []

    def next_id(self):
        with self.lock:
            self.count += 1
            return f"c{self.count}"

    def record(self, entry):
        with self.lock:
            self.trace.append(entry)

# the state of one agent session: its issues, decisions, call cache, tokens...
# and progress
class SessionState:
    def __init__(self, key, kind, context, issues, review_ids, counter, settings):
        self.key = key
        self.kind = kind
        self.context = context
        self.issues = issues
        self.review_ids = review_ids
        self.counter = counter
        self.settings = settings
        self.lock = threading.Lock()
        self.cache = {}
        self.calls = {}
        self.decisions = {}
        self.links = []
        self.added = []
        self.types = {}
        self.roles = {}
        self.names = {}
        self.changes = 0
        self.reminder = None
        self.gate_passed = False
        self.failure = None
        self.gate_failures = 0
        self.tokens = {"input": 0, "output": 0}
        self.steps = 0

    # records progress after a successful action; this also clears the guard...
    # reminder
    def changed(self):
        with self.lock:
            self.changes += 1
            self.reminder = None

    # lists what the agent still has to do; the done gate uses this list
    def open_items(self):
        items = [f"{issue_id} (needs confirm, dismiss or accept)" for issue_id in self.review_ids if issue_id not in self.decisions]
        if self.kind == "dataset":
            schema = {column.name: column for column in self.context.schema}
            items += [f"set_type for {name} (low confidence type)" for name, column in schema.items() if column.confidence == "low" and name not in self.types]
            items += [f"review_name for {name}" for name in schema if name not in self.names]
        return items

    # the maximum number of example rows a tool may return
    def max_rows(self):
        return min(10, self.settings["agent"]["max_example_rows"])

# turns a tool result into short JSON text
def compact(value, limit=3000):
    text = json.dumps(value, default=str, ensure_ascii=False)
    if len(text) > limit:
        text = text[:limit] + " ...(cut)"
    return text

# finds the columns a tool call is about, so actions can check their evidence
def call_columns(state, arguments):
    columns = set()
    for key in ["column", "time_column"]:
        if arguments.get(key):
            columns.add(arguments[key])
    for column in arguments.get("columns") or []:
        columns.add(column)
    for key in ["condition", "expression"]:
        if arguments.get(key):
            columns |= {name for name in state.context.typed.columns if str(name) in arguments[key]}
    issue_id = arguments.get("issue_id") or arguments.get("issue_or_condition")
    if issue_id in state.issues:
        columns |= set(state.issues[issue_id].columns)
    return columns

# base class of the read-only tools: gives call ids, caches repeated calls,...
# writes the trace and adds guard reminders
class EdaTool(Tool):
    output_type = "string"

    def __init__(self, state):
        self.state = state
        super().__init__()

    def call(self, arguments, compute):
        state = self.state
        key = (self.name, json.dumps(arguments, sort_keys=True, default=str))
        with state.lock:
            cached = state.cache.get(key)
        if cached is not None:
            result = {"note": "You already ran this call. Same result as before.", **cached}
            logger.debug(f"  [{state.key}] -> {self.name}({arguments_text(arguments)}) (repeated call, cached answer)")
        else:
            call_id = state.counter.next_id()
            logger.debug(f"  [{state.key}] -> {call_id} {self.name}({arguments_text(arguments)})")
            try:
                body = compute()
            except Exception as exception:
                body = {"error": f"{type(exception).__name__}: {exception}"}
            logger.debug(f"  [{state.key}]      <- {short(body)}")
            result = {"call_id": call_id, "result": body}
            columns = call_columns(state, arguments)
            issue_id = arguments.get("issue_id") or arguments.get("issue_or_condition")
            with state.lock:
                state.cache[key] = result
                state.calls[call_id] = {"tool": self.name, "columns": columns, "issue_id": issue_id}
            state.counter.record({"call_id": call_id, "session": state.key, "tool": self.name, "arguments": arguments, "result": body})
        if state.reminder:
            result = {**result, "reminder": state.reminder}
        return compact(result)

# returns the typed view of the session data
def typed(state):
    return state.context.typed

# returns the target column of the typed view
def target_values(state):
    return state.context.typed[state.context.target]

# builds a short summary of one column: type, missing, unique, quantiles or...
# top values
def summary_of(state, column):
    values = typed(state)[column]
    info = {
        "type": state.context.types.get(column),
        "rows": len(values),
        "missing": int(values.isna().sum()),
        "unique": int(values.nunique()),
    }
    present = values.dropna()
    if state.context.types.get(column) in numeric_types and len(present):
        numbers = present.astype(float)
        info["quantiles"] = {str(q): round(float(numbers.quantile(q)), 4) for q in [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]}
        info["zeros"] = int((numbers == 0).sum())
        info["negatives"] = int((numbers < 0).sum())
    elif len(present):
        top = present.astype(str).value_counts().head(5)
        info["top_values"] = {} if state.settings["agent"]["hide_examples"] else top.to_dict()
    return info

# tool: returns the full record of one issue
class GetIssue(EdaTool):
    name = "get_issue"
    description = "Return the full record of one issue (evidence, rows, default fix, links)."
    inputs = {"issue_id": {"type": "string", "description": "Issue id, for example MISS-001."}}

    def forward(self, issue_id: str) -> str:
        def compute():
            issue = self.state.context.ledger.issues.get(issue_id)
            if issue is None:
                return {"error": f"Unknown issue id {issue_id}."}
            return issue.model_dump(exclude={"fingerprint"})
        return self.call({"issue_id": issue_id}, compute)

# tool: a short summary of one column
class ColumnSummary(EdaTool):
    name = "column_summary"
    description = "Short summary of one column: type, missing values, unique values, quantiles or top values."
    inputs = {"column": {"type": "string", "description": "Column name."}}

    def forward(self, column: str) -> str:
        def compute():
            if column not in typed(self.state).columns:
                return {"error": f"Unknown column {column!r}."}
            return summary_of(self.state, column)
        return self.call({"column": column}, compute)

# tool: the top raw values of one column
class ValueCounts(EdaTool):
    name = "value_counts"
    description = "Top values of one column with their counts (raw text values, so hidden tokens are visible)."
    inputs = {
        "column": {"type": "string", "description": "Column name."},
        "top": {"type": "integer", "description": "How many values to return (max 30).", "nullable": True},
    }

    def forward(self, column: str, top: int | None = None) -> str:
        def compute():
            if column not in self.state.context.raw.columns:
                return {"error": f"Unknown column {column!r}."}
            counts = self.state.context.raw[column].astype(str).value_counts()
            shown = counts.head(min(top or 10, 30))
            return {"unique": int(len(counts)), "counts": {repr(key): int(value) for key, value in shown.items()}}
        return self.call({"column": column, "top": top}, compute)

# tool: counts the rows that match a condition and shows a few examples
class RowsWhere(EdaTool):
    name = "rows_where"
    description = (
        "Count rows that match a condition and show up to 10 example rows with the listed columns. "
        "Condition grammar: columns in backticks or plain names, numbers, 'strings', + - * /, "
        "== != < <= > >=, and/or/not, missing(column). Numbers compare parsed values; a comparison with a 'string' "
        "uses the raw text, so hidden tokens such as ' ' or '?' can be matched. missing(column) is true for empty and "
        "hidden-missing values. Example: `tenure` == 0 and missing(TotalCharges)."
    )
    inputs = {
        "condition": {"type": "string", "description": "Condition in the rule grammar."},
        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to show."},
        "n": {"type": "integer", "description": "Number of example rows (max 10).", "nullable": True},
    }

    def forward(self, condition: str, columns: list, n: int | None = None) -> str:
        def compute():
            mask, _ = evaluate_rule(typed(self.state), raw=self.state.context.raw, expression=condition)
            known = [column for column in columns if column in typed(self.state).columns]
            result = {"matching_rows": int(mask.sum()), "share": round(float(mask.mean()), 4)}
            if not self.state.settings["agent"]["hide_examples"]:
                rows = typed(self.state).loc[mask.to_numpy(), known].head(min(n or 5, self.state.max_rows()))
                result["examples"] = {int(index): {key: str(value) for key, value in row.items()} for index, row in rows.iterrows()}
            return result
        return self.call({"condition": condition, "columns": columns, "n": n}, compute)

# tool: columns whose values are special in the rows of an issue or a...
# condition
class ContrastRows(EdaTool):
    name = "contrast_rows"
    description = "For an issue id or a condition, list the columns whose most common value differs most between these rows and the other rows."
    inputs = {"issue_or_condition": {"type": "string", "description": "An issue id (for example MISS-001) or a condition."}}

    def forward(self, issue_or_condition: str) -> str:
        def compute():
            context = self.state.context
            if issue_or_condition in context.ledger.issues:
                rows = context.ledger.rows.get(issue_or_condition)
                if rows is None or len(rows) == 0:
                    return {"error": "This issue has no row set."}
                mask = context.raw.index.isin(rows)
            else:
                mask = evaluate_rule(typed(self.state), raw=self.state.context.raw, expression=issue_or_condition)[0].to_numpy()
            if mask.sum() == 0 or mask.all():
                return {"error": "The condition selects no rows or all rows."}
            results = []
            for column in context.raw.columns:
                if context.types[column] in {"id", "text", "empty"}:
                    continue
                values = context.raw[column].astype(str)
                inside = values[mask].value_counts(normalize=True)
                value = inside.index[0]
                outside = float((values[~mask] == value).mean())
                results.append({"column": str(column), "value": value, "share_in_rows": round(float(inside.iloc[0]), 4), "share_elsewhere": round(outside, 4)})
            results.sort(key=lambda item: item["share_elsewhere"] - item["share_in_rows"])
            return {"rows": int(mask.sum()), "top_contrasts": results[:5]}
        return self.call({"issue_or_condition": issue_or_condition}, compute)

# tool: target statistics where a condition is true vs false
class CompareGroups(EdaTool):
    name = "compare_groups"
    description = "Target statistics for rows where the condition is true vs false (class shares or mean/median)."
    inputs = {"condition": {"type": "string", "description": "Condition in the rule grammar."}}

    def forward(self, condition: str) -> str:
        def compute():
            mask, _ = evaluate_rule(typed(self.state), raw=self.state.context.raw, expression=condition)
            target = target_values(self.state)
            result = {}
            for name, part in [("condition_true", target[mask.to_numpy()]), ("condition_false", target[~mask.to_numpy()])]:
                part = part.dropna()
                if self.state.context.is_classification:
                    result[name] = {"rows": len(part), "class_shares": part.astype(str).value_counts(normalize=True).round(4).to_dict()}
                else:
                    numbers = part.astype(float)
                    result[name] = {"rows": len(part), "mean": round(float(numbers.mean()), 4) if len(part) else None, "median": round(float(numbers.median()), 4) if len(part) else None}
            return result
        return self.call({"condition": condition}, compute)

# correlation ratio (eta) between categories and numbers
def correlation_ratio(categories, values):
    frame = pd.DataFrame({"group": categories, "value": values}).dropna()
    if frame.empty or frame["value"].var() == 0:
        return 0.0
    grand = frame["value"].mean()
    groups = frame.groupby("group")["value"]
    between = (groups.count() * (groups.mean() - grand) ** 2).sum()
    total = ((frame["value"] - grand) ** 2).sum()
    return float(np.sqrt(between / total))

# Cramer's V between two categorical columns
def cramers_v(first, second):
    table = pd.crosstab(first, second)
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table)[0]
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))

# tool: the association between a column and the target, with the right...
# measure for the types
class TargetAssociation(EdaTool):
    name = "target_association"
    description = "Association between a column and the target, with the right measure for the type pair (Spearman, correlation ratio eta, Cramer's V)."
    inputs = {
        "column": {"type": "string", "description": "Column name."},
        "treat_as": {"type": "string", "description": "'numeric' or 'categorical'."},
    }

    def forward(self, column: str, treat_as: str) -> str:
        def compute():
            frame = typed(self.state)
            if column not in frame.columns:
                return {"error": f"Unknown column {column!r}."}
            target = target_values(self.state)
            values = frame[column]
            if treat_as == "numeric":
                numbers = pd.to_numeric(values, errors="coerce")
                if self.state.context.is_classification:
                    return {"measure": "eta (correlation ratio)", "value": round(correlation_ratio(target.astype(str).where(target.notna()), numbers), 4)}
                both = pd.DataFrame({"x": numbers, "y": target.astype(float)}).dropna()
                return {"measure": "spearman", "value": round(float(spearmanr(both["x"], both["y"])[0]), 4)}
            labels = values.astype(str).where(values.notna())
            if self.state.context.is_classification:
                both = pd.DataFrame({"x": labels, "y": target.astype(str).where(target.notna())}).dropna()
                return {"measure": "cramers_v", "value": round(cramers_v(both["x"], both["y"]), 4)}
            return {"measure": "eta (correlation ratio)", "value": round(correlation_ratio(labels, target.astype(float)), 4)}
        return self.call({"column": column, "treat_as": treat_as}, compute)

# tool: counts the rows that break a rule
class CheckRule(EdaTool):
    name = "check_rule"
    description = (
        "Count rows that VIOLATE a rule (the rule should be true for every row). Same grammar as rows_where. "
        "Example: `sqft_living` == `sqft_above` + `sqft_basement`."
    )
    inputs = {"expression": {"type": "string", "description": "Rule that should hold for every row."}}

    def forward(self, expression: str) -> str:
        def compute():
            holds, columns = evaluate_rule(typed(self.state), raw=self.state.context.raw, expression=expression)
            known = typed(self.state)[list(columns)].notna().all(axis=1) if columns else holds | True
            violations = known & ~holds
            result = {"checked_rows": int(known.sum()), "violations": int(violations.sum())}
            if not self.state.settings["agent"]["hide_examples"]:
                result["example_rows"] = [int(row) for row in violations[violations].index[:5]]
            return result
        return self.call({"expression": expression}, compute)

# tool: checks a linear relation (numbers) or a one-to-one mapping...
# (categories)
class Relation(EdaTool):
    name = "relation"
    description = "Check if the first column is an exact or near linear function of the others (numeric), or a one-to-one mapping (categorical)."
    inputs = {"columns": {"type": "array", "items": {"type": "string"}, "description": "Two or more columns; the first is explained by the others."}}

    def forward(self, columns: list) -> str:
        def compute():
            frame = typed(self.state)
            unknown = [column for column in columns if column not in frame.columns]
            if unknown or len(columns) < 2:
                return {"error": f"Need two or more known columns. Unknown: {unknown}"}
            types = [self.state.context.types[column] for column in columns]
            if all(kind in numeric_types for kind in types):
                data = frame[columns].astype(float).dropna()
                design = np.column_stack([data[columns[1:]].to_numpy(), np.ones(len(data))])
                coefficients = np.linalg.lstsq(design, data[columns[0]].to_numpy(), rcond=None)[0]
                residual = data[columns[0]].to_numpy() - design @ coefficients
                r2 = 1 - residual.var() / data[columns[0]].var() if data[columns[0]].var() else 1.0
                return {"kind": "linear", "r2": round(float(r2), 6), "coefficients": dict(zip(columns[1:] + ["intercept"], np.round(coefficients, 6).tolist())), "max_abs_residual": float(np.abs(residual).max())}
            data = frame[columns[:2]].astype(str)
            pairs = data.drop_duplicates()
            return {"kind": "mapping", "levels_first": int(data[columns[0]].nunique()), "levels_second": int(data[columns[1]].nunique()), "pairs": len(pairs), "one_to_one": len(pairs) == data[columns[0]].nunique() == data[columns[1]].nunique()}
        return self.call({"columns": columns}, compute)

# tool: the target per time bin of a date column
class TimeTrend(EdaTool):
    name = "time_trend"
    description = "Target over 10 time bins of a datetime column (class shares or mean per bin)."
    inputs = {"time_column": {"type": "string", "description": "Datetime column."}}

    def forward(self, time_column: str) -> str:
        def compute():
            frame = typed(self.state)
            if self.state.context.types.get(time_column) != "datetime":
                return {"error": f"{time_column!r} is not a datetime column."}
            target = target_values(self.state)
            data = pd.DataFrame({"time": frame[time_column], "target": target}).dropna()
            data["bin"] = pd.qcut(data["time"].rank(method="first"), 10, labels=False)
            bins = {}
            for number, part in data.groupby("bin"):
                label = f"{part['time'].min().date()}..{part['time'].max().date()}"
                if self.state.context.is_classification:
                    bins[label] = part["target"].astype(str).value_counts(normalize=True).round(3).to_dict()
                else:
                    bins[label] = round(float(part["target"].astype(float).mean()), 4)
            return {"bins": bins}
        return self.call({"time_column": time_column}, compute)

# all read-only tools; every session gets them
read_tools = [GetIssue, ColumnSummary, ValueCounts, RowsWhere, ContrastRows, CompareGroups, TargetAssociation, CheckRule, Relation, TimeTrend]
