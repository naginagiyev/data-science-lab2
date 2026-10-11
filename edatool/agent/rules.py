import re
import numpy as np
import pandas as pd

# tokens of the rule language: numbers, 'strings', `columns`, operators and...
# words
token_pattern = re.compile(
    r"\s*(?:(?P<number>\d+\.?\d*(?:[eE][-+]?\d+)?)|(?P<string>'[^']*'|\"[^\"]*\")|(?P<column>`[^`]+`)"
    r"|(?P<operator><=|>=|==|!=|<|>|\+|-|\*|/|\(|\)|,)|(?P<word>[A-Za-z_][A-Za-z0-9_]*))"
)
keywords = {"and", "or", "not", "missing", "true", "false"}

# splits a rule into tokens; unknown text raises an error
def tokenize(expression):
    tokens = []
    position = 0
    expression = expression.strip()
    while position < len(expression):
        match = token_pattern.match(expression, position)
        if not match or match.end() == position:
            raise ValueError(f"Unexpected text at position {position}: {expression[position:position + 10]!r}")
        kind = match.lastgroup
        tokens.append((kind, match.group(kind)))
        position = match.end()
    return tokens

# a small recursive parser for the conditions and rules of the agent; it...
# never uses eval
class RuleParser:
    def __init__(self, frame, expression, raw=None):
        self.frame = frame
        self.raw = raw
        self.tokens = tokenize(expression)
        self.position = 0
        self.columns = set()

    # returns the next token without using it
    def peek(self):
        return self.tokens[self.position] if self.position < len(self.tokens) else (None, None)

    # uses the next token, and checks its text when a value is given
    def take(self, value=None):
        kind, text = self.peek()
        if kind is None or (value is not None and text != value):
            raise ValueError(f"Expected {value or 'a value'}, found {text!r}")
        self.position += 1
        return kind, text

    # parses the whole rule and checks that nothing is left over
    def parse(self):
        result = self.parse_or()
        if self.position != len(self.tokens):
            raise ValueError(f"Unexpected token {self.peek()[1]!r}")
        return result

    # a or b
    def parse_or(self):
        left = self.parse_and()
        while self.peek() == ("word", "or"):
            self.take()
            left = as_bool(left) | as_bool(self.parse_and())
        return left

    # a and b
    def parse_and(self):
        left = self.parse_not()
        while self.peek() == ("word", "and"):
            self.take()
            left = as_bool(left) & as_bool(self.parse_not())
        return left

    # not a
    def parse_not(self):
        if self.peek() == ("word", "not"):
            self.take()
            return ~as_bool(self.parse_not())
        return self.parse_comparison()

    # a == b, a != b, a < b, a <= b, a > b, a >= b
    def parse_comparison(self):
        left = self.parse_sum()
        kind, text = self.peek()
        if kind == "operator" and text in {"<", "<=", ">", ">=", "==", "!="}:
            self.take()
            right = self.parse_sum()
            left, right = self.text_side(left, right), self.text_side(right, left)
            operations = {
                "<": lambda a, b: a < b,
                "<=": lambda a, b: a <= b,
                ">": lambda a, b: a > b,
                ">=": lambda a, b: a >= b,
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
            }
            return operations[text](left, right)
        return left

    # a + b, a - b
    def parse_sum(self):
        left = self.parse_term()
        while self.peek()[0] == "operator" and self.peek()[1] in {"+", "-"}:
            _, text = self.take()
            right = self.parse_term()
            left = left + right if text == "+" else left - right
        return left

    # a * b, a / b
    def parse_term(self):
        left = self.parse_factor()
        while self.peek()[0] == "operator" and self.peek()[1] in {"*", "/"}:
            _, text = self.take()
            right = self.parse_factor()
            left = left * right if text == "*" else left / right
        return left

    # numbers, 'strings', columns, brackets, missing(column), true and false
    def parse_factor(self):
        kind, text = self.peek()
        if kind == "number":
            self.take()
            return float(text)
        if kind == "string":
            self.take()
            return text[1:-1]
        if kind == "operator" and text == "-":
            self.take()
            return -self.parse_factor()
        if kind == "operator" and text == "(":
            self.take()
            value = self.parse_or()
            self.take(")")
            return value
        if kind == "word" and text == "missing":
            self.take()
            self.take("(")
            column = self.column_values(*self.take())
            self.take(")")
            return column.isna()
        if kind == "word" and text in {"true", "false"}:
            self.take()
            return text == "true"
        if kind in {"column", "word"}:
            self.take()
            return self.column_values(kind, text)
        raise ValueError(f"Unexpected token {text!r}")

    # a column compared with a 'string' uses its raw text, so hidden tokens...
    # like ' ' can be matched
    def text_side(self, value, other):
        if isinstance(value, pd.Series) and isinstance(other, str) and self.raw is not None and value.name in self.raw.columns:
            raw = self.raw[value.name]
            return raw.astype(str).where(raw.notna(), "")
        return value

    # returns the values of a known column and remembers that the rule uses...
    # it
    def column_values(self, kind, text):
        name = text[1:-1] if kind == "column" else text
        if name not in self.frame.columns:
            raise ValueError(f"Unknown column {name!r}")
        self.columns.add(name)
        return self.frame[name]

# turns a result into True/False values (missing becomes False)
def as_bool(values):
    if isinstance(values, pd.Series):
        return values.fillna(False).astype(bool)
    return bool(values)

# evaluates a rule on a DataFrame and returns the row mask and the used...
# columns
def evaluate_rule(frame, expression, raw=None):
    parser = RuleParser(frame, expression, raw)
    result = parser.parse()
    if not isinstance(result, pd.Series):
        result = pd.Series(np.full(len(frame), bool(result)), index=frame.index)
    return as_bool(result), parser.columns
