import json
from typing import get_args
from edatool.agent.tools import EdaTool, compact
from edatool.core.progress import arguments_text, logger
from edatool.schemas import FixAction, Priority, Role, SemanticType

# the allowed fix actions
fix_actions = set(get_args(FixAction))

# base class of the actions that write decisions; invalid input returns an...
# error message instead of a crash
class ActionTool(EdaTool):
    # runs an action, writes it to the trace and the log, and records...
    # progress when it worked
    def act(self, arguments, apply):
        try:
            message = apply()
        except ValueError as error:
            message = {"error": str(error)}
        prefix = f"  [{self.state.key}]"
        if "error" in message:
            logger.debug(f"{prefix} x {self.name} rejected: {message['error']}")
        else:
            self.state.changed()
            details = arguments_text(arguments)
            if self.name == "review_name" and arguments.get("verdict") == "clean":
                logger.debug(f"{prefix} + {self.name}: {details}")
            else:
                logger.debug(f"{prefix} + {self.name}: {details}")
        self.state.counter.record({"session": self.state.key, "tool": self.name, "arguments": arguments, "result": message})
        if self.state.reminder and "error" in message:
            message = {**message, "reminder": self.state.reminder}
        return compact(message)

    # checks that an issue belongs to this session (and needs review, for a...
    # dismissal)
    def require_issue(self, issue_id, review_only=False):
        if issue_id not in self.state.issues:
            raise ValueError(f"Issue {issue_id} is not part of this session. Session issues: {sorted(self.state.issues)}")
        issue = self.state.issues[issue_id]
        if review_only and not issue.needs_review:
            raise ValueError(f"Issue {issue_id} has needs_review = false and can not be dismissed.")
        return issue

    # checks that the cited call ids exist and are about the same columns
    def require_calls(self, calls, columns=None, issue_id=None):
        calls = calls or []
        if not calls:
            raise ValueError("Cite at least one call id (for example c3) from your tool results as evidence.")
        unknown = [call for call in calls if call not in self.state.calls]
        if unknown:
            raise ValueError(f"Unknown call ids {unknown}. Use ids returned by the read-only tools in this session.")
        if columns:
            related = [
                call for call in calls
                if self.state.calls[call]["columns"] & set(columns) or (issue_id and self.state.calls[call]["issue_id"] == issue_id)
            ]
            if not related:
                raise ValueError(f"None of the cited calls involve the columns {sorted(columns)}.")
        return calls

    # checks that a fix uses an allowed action and known columns
    def check_fix(self, action, params):
        if action not in fix_actions:
            raise ValueError(f"Unknown fix action {action!r}. Allowed: {sorted(fix_actions)}")
        params = params or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                raise ValueError("fix_params must be a JSON object.")
        names = [params.get("column")] if params.get("column") else []
        names += params.get("columns") or []
        unknown = [name for name in names if name not in self.state.context.raw.columns]
        if unknown:
            raise ValueError(f"Fix params name unknown columns {unknown}.")
        return {"action": action, "params": params}

# action: confirm a real issue with a diagnosis and a fix
class Confirm(ActionTool):
    name = "confirm"
    description = "Confirm an issue as real, with a diagnosis and a fix from the fix vocabulary. Cite the call ids that support it."
    inputs = {
        "issue_id": {"type": "string", "description": "Issue id."},
        "diagnosis": {"type": "string", "description": "One or two sentences: the cause, in plain words."},
        "fix_action": {"type": "string", "description": "One action from the fix vocabulary."},
        "fix_params": {"type": "object", "description": "Parameters of the fix, for example {\"column\": \"x\", \"mapping\": {\" \": 0}}."},
        "calls": {"type": "array", "items": {"type": "string"}, "description": "Call ids that support the decision.", "nullable": True},
        "alternatives": {"type": "array", "items": {"type": "object"}, "description": "Other possible fixes as {action, params}.", "nullable": True},
    }

    def forward(self, issue_id: str, diagnosis: str, fix_action: str, fix_params: dict, calls: list | None = None, alternatives: list | None = None) -> str:
        arguments = {"issue_id": issue_id, "diagnosis": diagnosis, "fix_action": fix_action, "fix_params": fix_params, "calls": calls, "alternatives": alternatives}
        def apply():
            issue = self.require_issue(issue_id)
            if not diagnosis or not diagnosis.strip():
                raise ValueError("A diagnosis is required.")
            fix = self.check_fix(fix_action, fix_params)
            other = [self.check_fix(item.get("action"), item.get("params")) for item in alternatives or []]
            if calls:
                self.require_calls(calls)
            self.state.decisions[issue_id] = {"status": "confirmed", "diagnosis": diagnosis, "fix": fix, "alternatives": other, "calls": calls or []}
            return {"ok": f"{issue.id} confirmed."}
        return self.act(arguments, apply)

# action: dismiss a false positive (review items only, with evidence)
class Dismiss(ActionTool):
    name = "dismiss"
    description = "Dismiss a false positive. Only for issues with needs_review = true. Cite at least one call id about the same columns."
    inputs = {
        "issue_id": {"type": "string", "description": "Issue id."},
        "reason": {"type": "string", "description": "Why this is not a real issue."},
        "calls": {"type": "array", "items": {"type": "string"}, "description": "Call ids that support the dismissal."},
    }

    def forward(self, issue_id: str, reason: str, calls: list) -> str:
        def apply():
            issue = self.require_issue(issue_id, review_only=True)
            self.require_calls(calls, issue.columns, issue_id)
            self.state.decisions[issue_id] = {"status": "dismissed", "diagnosis": reason, "calls": calls}
            return {"ok": f"{issue.id} dismissed."}
        return self.act({"issue_id": issue_id, "reason": reason, "calls": calls}, apply)

# action: accept a real issue that needs no fix, with a reason
class Accept(ActionTool):
    name = "accept"
    description = "Accept a real issue that should not be fixed on purpose (for example skew with tree models). Give the reason."
    inputs = {
        "issue_id": {"type": "string", "description": "Issue id."},
        "reason": {"type": "string", "description": "Why no fix is needed."},
    }

    def forward(self, issue_id: str, reason: str) -> str:
        def apply():
            issue = self.require_issue(issue_id)
            if not reason.strip():
                raise ValueError("A reason is required.")
            self.state.decisions[issue_id] = {"status": "accepted", "diagnosis": reason, "calls": []}
            return {"ok": f"{issue.id} accepted."}
        return self.act({"issue_id": issue_id, "reason": reason}, apply)

# action: record that one issue is caused by another
class Link(ActionTool):
    name = "link"
    description = "Record that one issue is caused by another (child caused by parent). Cite at least one call id."
    inputs = {
        "child_id": {"type": "string", "description": "Issue that is a symptom."},
        "parent_id": {"type": "string", "description": "Issue that is the cause."},
        "reason": {"type": "string", "description": "Why they are linked."},
        "calls": {"type": "array", "items": {"type": "string"}, "description": "Call ids that support the link."},
    }

    def forward(self, child_id: str, parent_id: str, reason: str, calls: list) -> str:
        def apply():
            child = self.require_issue(child_id)
            parent = self.require_issue(parent_id)
            if child_id == parent_id:
                raise ValueError("An issue can not cause itself.")
            self.require_calls(calls, set(child.columns) | set(parent.columns))
            self.state.links.append({"child": child_id, "parent": parent_id, "reason": reason, "calls": calls})
            return {"ok": f"{child_id} linked to {parent_id}."}
        return self.act({"child_id": child_id, "parent_id": parent_id, "reason": reason, "calls": calls}, apply)

# action: add an issue that the detectors missed, with evidence
class AddIssue(ActionTool):
    name = "add_issue"
    description = "Add an issue that the detectors did not find. Use a catalog code (for example D04, G03). Cite at least one call id."
    inputs = {
        "code": {"type": "string", "description": "Issue code from the catalog."},
        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns involved."},
        "title": {"type": "string", "description": "Short title."},
        "priority": {"type": "string", "description": "must_fix, should_fix or consider."},
        "diagnosis": {"type": "string", "description": "The cause, in plain words."},
        "fix_action": {"type": "string", "description": "One action from the fix vocabulary."},
        "fix_params": {"type": "object", "description": "Parameters of the fix."},
        "calls": {"type": "array", "items": {"type": "string"}, "description": "Call ids that support the issue."},
    }

    def forward(self, code: str, columns: list, title: str, priority: str, diagnosis: str, fix_action: str, fix_params: dict, calls: list) -> str:
        arguments = {"code": code, "columns": columns, "title": title, "priority": priority, "diagnosis": diagnosis, "fix_action": fix_action, "fix_params": fix_params, "calls": calls}
        def apply():
            if code not in self.state.context.ledger.catalog:
                raise ValueError(f"Unknown code {code!r}.")
            if priority not in get_args(Priority):
                raise ValueError(f"Priority must be one of {get_args(Priority)}.")
            unknown = [column for column in columns if column not in self.state.context.raw.columns]
            if unknown:
                raise ValueError(f"Unknown columns {unknown}.")
            self.require_calls(calls, columns)
            fix = self.check_fix(fix_action, fix_params)
            self.state.added.append({**arguments, "fix": fix})
            return {"ok": f"New {code} issue recorded for {columns}."}
        return self.act(arguments, apply)

# action: set the semantic type of a column
class SetType(ActionTool):
    name = "set_type"
    description = "Set the semantic type of a column (for low-confidence types)."
    inputs = {
        "column": {"type": "string", "description": "Column name."},
        "semantic_type": {"type": "string", "description": f"One of {', '.join(get_args(SemanticType))}."},
        "calls": {"type": "array", "items": {"type": "string"}, "description": "Call ids that support the type.", "nullable": True},
    }

    def forward(self, column: str, semantic_type: str, calls: list | None = None) -> str:
        def apply():
            if column not in self.state.context.raw.columns:
                raise ValueError(f"Unknown column {column!r}.")
            if semantic_type not in get_args(SemanticType):
                raise ValueError(f"Unknown type {semantic_type!r}.")
            self.state.types[column] = semantic_type
            return {"ok": f"{column} set to {semantic_type}."}
        return self.act({"column": column, "semantic_type": semantic_type, "calls": calls}, apply)

# action: set the role of a column
class SetRole(ActionTool):
    name = "set_role"
    description = "Set the role of a column: feature, target, id, group_key, time_key or drop."
    inputs = {
        "column": {"type": "string", "description": "Column name."},
        "role": {"type": "string", "description": "feature, target, id, group_key, time_key or drop."},
        "calls": {"type": "array", "items": {"type": "string"}, "description": "Call ids that support the role.", "nullable": True},
    }

    def forward(self, column: str, role: str, calls: list | None = None) -> str:
        def apply():
            if column not in self.state.context.raw.columns:
                raise ValueError(f"Unknown column {column!r}.")
            if role not in get_args(Role):
                raise ValueError(f"Unknown role {role!r}.")
            self.state.roles[column] = role
            return {"ok": f"{column} role set to {role}."}
        return self.act({"column": column, "role": role, "calls": calls}, apply)

# action: mark a column name as clean or as suspected leakage (G03)
class ReviewName(ActionTool):
    name = "review_name"
    description = "Mark whether a column name suggests information known only after the target event (leakage by meaning)."
    inputs = {
        "column": {"type": "string", "description": "Column name."},
        "verdict": {"type": "string", "description": "'clean' or 'suspected'."},
        "reason": {"type": "string", "description": "Short reason; required for 'suspected'.", "nullable": True},
    }

    def forward(self, column: str, verdict: str, reason: str | None = None) -> str:
        def apply():
            if column not in self.state.context.raw.columns:
                raise ValueError(f"Unknown column {column!r}.")
            if verdict not in {"clean", "suspected"}:
                raise ValueError("verdict must be 'clean' or 'suspected'.")
            if verdict == "suspected" and not reason:
                raise ValueError("Give a reason for a suspected column.")
            self.state.names[column] = {"verdict": verdict, "reason": reason}
            return {"ok": f"{column}: {verdict}."}
        return self.act({"column": column, "verdict": verdict, "reason": reason}, apply)

# the actions of the issue sessions and of the dataset session
cluster_actions = [Confirm, Dismiss, Accept, Link, AddIssue]
dataset_actions = [Confirm, Dismiss, Accept, SetType, SetRole, ReviewName, AddIssue]
