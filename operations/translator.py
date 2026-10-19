import json
import logging
from schemas import Plan, Step
from tools.columntypes import TOKENS

logger = logging.getLogger("automl.plan")
LEAK_CODES = {"E03", "G01"}
TARGET_CODES = {"F03", "A05"}
ORDER = [
    "replace_values", "set_missing_where", "parse_number", "parse_units", "parse_datetime", "parse_boolean", "extract_pattern",
    "map_categories", "as_category", "date_difference", "add_missing_indicator", "group_rare", "text_features", "drop_columns",
]
DATE_FORMATS = {"ISO8601": "ISO8601", "mixed": "mixed"}
MAGNITUDES = {"k": 1e3, "m": 1e6, "mn": 1e6, "b": 1e9, "bn": 1e9}
NUMERIC_OPS = {"parse_number", "parse_units", "parse_datetime", "parse_boolean", "extract_pattern"}

# one translated step; protected steps are never gated
def step(op, params, issue, protected=False, reason=None):
    return Step(op=op, params=params, issues=[issue.id], reason=reason or issue.diagnosis or issue.title, protected=protected, source="translator")

# the column of an issue: the fix parameter, else its only column
def column_of(issue):
    return issue.params.get("column") or (issue.columns[0] if len(issue.columns) == 1 else None)

# unit factors from the fix parameters: a dict {unit: factor}, a list of...
# {"unit", "factor"} pairs, one unit without a factor (factor 1), or the...
# magnitude suffixes k, M, B (with k present: M is million, not metres)
def unit_factors(issue):
    units = issue.params.get("units") or []
    if isinstance(units, dict):
        return [{"unit": str(unit), "factor": float(factor)} for unit, factor in units.items() if factor is not None]
    pairs = [item for item in units if isinstance(item, dict) and "unit" in item and "factor" in item]
    if pairs:
        return [{"unit": str(item["unit"]), "factor": float(item["factor"])} for item in pairs]
    if len(units) == 1:
        return [{"unit": str(units[0]), "factor": 1.0}]
    names = {str(unit).lower() for unit in units}
    if names <= set(MAGNITUDES) and "k" in names:
        return [{"unit": str(unit), "factor": MAGNITUDES[str(unit).lower()]} for unit in units]
    return []

# the set_missing_where parameters of D02 from the flagged values: values...
# above the median give `above`, values below give `below`
def outlier_limits(issue):
    values = issue.evidence.get("values") or []
    median = issue.evidence.get("median")
    if not values or median is None:
        return None
    high = [value for value in values if value > median]
    low = [value for value in values if value < median]
    params = {}
    if high:
        params["above"] = min(high) - 1e-9 * max(1.0, abs(min(high)))
    if low:
        params["below"] = max(low) + 1e-9 * max(1.0, abs(max(low)))
    return params

# turns one issue into steps (Appendix C); returns ([column steps], [row...
# steps], [target steps], reason when skipped)
def translate_issue(issue, tokens):
    code = issue.code
    params = issue.params
    column = column_of(issue)
    if code == "A02":
        return [step("drop_columns", {"columns": [column]}, issue, True)], [], [], None
    if code == "C02":
        items = [{"value": token, "new": None} for token in issue.evidence.get("tokens", {})]
        return [step("replace_values", {"column": column, "items": items}, issue, True)], [], [], None
    if code == "A04":
        return [], [step("drop_duplicate_rows", {}, issue, True)], [], None
    if code == "F03":
        items = [{"value": str(value), "canonical": str(canonical)} for value, canonical in (params.get("mapping") or {}).items()]
        return ([], [], [step("map_target_labels", {"items": items}, issue, True)], None) if items else ([], [], [], "no mapping")
    if code == "B01":
        cleaning = params.get("cleaning") or []
        options = {"column": column, "remove": [char for char in cleaning if char not in {",", "%"}], "thousands": "," if "," in cleaning else None, "percent": "%" in cleaning}
        return [step("parse_number", options, issue)], [], [], None
    if code == "B07":
        factors = unit_factors(issue)
        if not factors:
            return [], [], [], "unit factors unknown (two or more units); left to the cleaner"
        return [step("parse_units", {"column": column, "units": factors}, issue)], [], [], None
    if code in {"B02", "B09"}:
        formats = params.get("formats") or [params.get("format") or "mixed"]
        formats = [DATE_FORMATS.get(item, item) for item in formats if item] + (["mixed"] if "mixed" not in formats else [])
        options = {"column": column, "formats": formats, "dayfirst": bool(params.get("dayfirst", False)), "parts": ["year", "month", "day", "weekday"]}
        return [step("parse_datetime", options, issue)], [], [], None
    if code == "B08":
        if not params.get("pattern") or not params.get("into"):
            return [], [], [], "no pattern"
        options = {"column": column, "pattern": params["pattern"], "into": params["into"], "types": ["number"] * len(params["into"])}
        return [step("extract_pattern", options, issue)], [], [], None
    if code == "B04":
        return [step("as_category", {"column": column}, issue)], [], [], None
    if code == "B05":
        mapping = params.get("mapping") or {}
        if not mapping:
            return [], [], [], "no mapping"
        items = [{"value": str(value), "canonical": str(canonical)} for value, canonical in mapping.items()]
        return [step("map_categories", {"column": column, "items": items, "case_fold": False, "strip": False}, issue)], [], [], None
    if code == "B06":
        values = issue.evidence.get("tokens") or []
        true_values = [value for value in values if value.strip().lower() in tokens["boolean_true"]]
        false_values = [value for value in values if value.strip().lower() in tokens["boolean_false"]]
        if not true_values or not false_values:
            return [], [], [], "tokens are not yes/no pairs"
        return [step("parse_boolean", {"column": column, "true_values": true_values, "false_values": false_values}, issue)], [], [], None
    if code == "C03":
        value = issue.evidence.get("value")
        if value is None:
            return [], [], [], "no sentinel value"
        replace = step("set_missing_where", {"column": column, "values": [str(value)]}, issue)
        return [replace, step("add_missing_indicator", {"columns": [column]}, issue)], [], [], None
    if code in {"C04", "C05"}:
        return [step("add_missing_indicator", {"columns": list(issue.columns)}, issue)], [], [], None
    if code in {"C06", "E01", "E02", "E06", "E08"}:
        return [step("drop_columns", {"columns": [column]}, issue)], [], [], None
    if code == "D01":
        rule = issue.rule
        if rule == "few_negatives":
            return [step("set_missing_where", {"column": column, "below": 0}, issue)], [], [], None
        if rule == "percent_over_100":
            return [step("set_missing_where", {"column": column, "above": 100}, issue)], [], [], None
        return [], [], [], f"no rule for {rule}"
    if code == "D02":
        limits = outlier_limits(issue)
        if not limits:
            return [], [], [], "no flagged values"
        return [step("set_missing_where", {"column": column, **limits}, issue)], [], [], None
    if code == "D03":
        dates = issue.evidence.get("dates") or {}
        if not dates:
            return [], [], [], "only future dates; nothing to replace"
        items = [{"value": str(value), "new": None} for value in dates]
        return [step("replace_values", {"column": column, "items": items}, issue)], [], [], None
    if code == "A05":
        return [], [step("drop_conflicting_rows", {}, issue)], [], None
    if code in {"D05", "E04"}:
        return [step("group_rare", {"column": column, "min_count": int(params.get("min_count", 10)), "label": "Other"}, issue)], [], [], None
    if code == "E05":
        return [step("text_features", {"column": column}, issue)], [], [], None
    return [], [], [], f"no translation for {code}"

# the default plan from the audit report (Appendix C): must_fix and should_fix...
# issues; leaks and ids are left to the leak guard; target-changing fixes are never...
# translated; group_rare is left out on columns parsed to numbers; returns the...
# plan and the skipped issues with reasons
def default_plan(report, target):
    tokens = {name: set(TOKENS[name]) for name in ["boolean_true", "boolean_false"]}
    steps, row_steps, target_steps, skipped = [], [], [], []
    for issue in report.issues:
        if issue.code in LEAK_CODES or issue.priority == "consider":
            continue
        if target in issue.columns and issue.code not in TARGET_CODES:
            skipped.append({"issue": issue.id, "reason": "touches the target"})
            continue
        try:
            columns, rows, targets, reason = translate_issue(issue, tokens)
        except Exception as error:
            columns, rows, targets, reason = [], [], [], f"{type(error).__name__}: {error}"
        if reason:
            skipped.append({"issue": issue.id, "code": issue.code, "reason": reason})
            logger.info(f"translator skipped {issue.id} ({issue.code}): {reason}")
            continue
        steps += columns
        row_steps += rows
        target_steps += targets
    parsed = {item.params.get("column") for item in steps if item.op in NUMERIC_OPS}
    for item in [item for item in steps if item.op == "group_rare" and item.params.get("column") in parsed]:
        skipped.append({"issue": item.issues[0], "reason": "group_rare left out: the column is parsed to numbers"})
        steps.remove(item)
    steps = sorted(steps, key=lambda item: (not item.protected, ORDER.index(item.op) if item.op in ORDER else len(ORDER)))
    unique_rows = {(item.op, json.dumps(item.params, sort_keys=True)): item for item in row_steps}
    return Plan(version=0, steps=steps, row_steps=list(unique_rows.values()), target_steps=target_steps, notes="translator"), skipped
