import re
import zlib
import numpy as np
import pandas as pd
from decimal import Decimal, InvalidOperation

LEVELS = {0: (0.0, 0.0), 1: (0.25, 0.05), 2: (0.50, 0.15), 3: (0.75, 0.30)}
MIXED_TYPES = ["ids", "packed", "dates", "booleans", "spellings", "mixed_units", "units", "impossible", "fake_missing", "plain_missing"]
ALL_TYPES = MIXED_TYPES + ["layout", "leak"]
TYPE_NUMBERS = {"units": 1, "mixed_units": 2, "dates": 3, "spellings": 4, "fake_missing": 5, "impossible": 6, "booleans": 7, "packed": 8, "ids": 9, "layout": 10, "leak": 11, "plain_missing": 12}
EXPECTED_OPS = {
    "units": "parse_number", "mixed_units": "parse_units", "dates": "parse_datetime", "spellings": "map_categories", "fake_missing": "replace_values",
    "impossible": "set_missing_where", "booleans": "parse_boolean", "packed": "extract_pattern", "ids": "drop_columns", "plain_missing": None,
}
UNIT_FAMILIES = {
    "months": (["month"], ("months", "days", Decimal(30))),
    "age": (["age", "years"], ("years", "months", Decimal(12))),
    "weight": (["weight", "kg", "gms", "gram", "mass"], ("kg", "g", Decimal(1000))),
    "length": (["length", "height", "width", "distance", "dist", "km", "meter", "metre"], ("m", "cm", Decimal(100))),
    "time": (["time", "duration", "hours", "hour", "minutes", "seconds", "days"], ("h", "min", Decimal(60))),
    "volume": (["volume", "litre", "liter"], ("l", "ml", Decimal(1000))),
}
MONEY_HINTS = ["price", "cost", "income", "salary", "amount", "charges", "balance", "value", "prc", "revenue", "sales", "spend", "mnt"]
PERCENT_HINTS = ["pct", "percent", "rate", "ratio", "share"]
TRUE_SPELLINGS = ["Y", "yes", "TRUE", "1", "t"]
FALSE_SPELLINGS = ["N", "no", "FALSE", "0", "f"]
SYNONYMS = {"yes": "y", "no": "n", "male": "M", "female": "F", "united states": "USA", "other": "misc", "unknown": "unk"}
TEXT_TOKENS = ["unknown", "N/A", "-", "?"]
PACK_TEMPLATES = ["{a} / {b}", "{a}x{b}", "L:{a} W:{b}"]
NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

# the raw text of a number written plainly (no exponent, no trailing zeros)
def plain(value):
    text = format(value.normalize(), "f")
    return text if "." not in text else text.rstrip("0").rstrip(".") or "0"

# the exact decimal of a text number, or None
def as_decimal(text):
    try:
        value = Decimal(str(text).strip())
    except InvalidOperation:
        return None
    return value if value.is_finite() else None

# adds thousand separators to the integer part of a plain number text
def with_thousands(text):
    sign = "-" if text.startswith("-") else ""
    whole, _, fraction = text.lstrip("+-").partition(".")
    grouped = f"{int(whole):,}" if whole else "0"
    return sign + grouped + (f".{fraction}" if fraction else "")

# the present (non-empty) values of a column
def present(raw, column):
    return raw[column].astype(str).str.strip() != ""

# true when every present value of a column is a number
def is_numeric(raw, column):
    mask = present(raw, column)
    return bool(mask.any()) and all(as_decimal(value) is not None for value in raw.loc[mask, column])

# true when every present value is a number and none is negative
def non_negative(raw, column):
    return all(as_decimal(value) >= 0 for value in raw.loc[present(raw, column), column])

# true when a column holds dates (90% parse, has digits, not numbers)
def is_date(raw, column):
    values = raw.loc[present(raw, column), column]
    if values.empty or is_numeric(raw, column) or not values.str.contains(r"\d").mean() > 0.9:
        return False
    parsed = pd.to_datetime(values.head(500), errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.9)

# the eligible columns of every type, from the clean table
def eligible(raw, target, column):
    if column == target:
        return set()
    values = raw.loc[present(raw, column), column]
    levels = values.nunique()
    kinds = {"fake_missing", "plain_missing"}
    if is_numeric(raw, column):
        if present(raw, column).all():
            kinds.add("packed")
        if levels >= 20 and non_negative(raw, column):
            kinds |= {"units", "mixed_units"}
        if non_negative(raw, column) and levels >= 5:
            kinds.add("impossible")
    elif is_date(raw, column):
        kinds.add("dates")
    elif 3 <= levels <= 50:
        kinds.add("spellings")
    if levels == 2:
        kinds.add("booleans")
    return kinds

# the unit family of a column from its name: weight, length, time, volume,...
# else a generic thousands/millions family
def unit_family(column):
    name = str(column).lower()
    for family, (hints, units) in UNIT_FAMILIES.items():
        if any(hint in name for hint in hints):
            return family, units
    return "generic", ("k", "M", Decimal(1000))

# type 1: the whole column gets symbols or a suffix: currency "$1,250.00",...
# percent "12.5%" or thousands "1.25k"; minority cells use a second spelling
def corrupt_units(raw, column, rows, rng):
    name = str(column).lower()
    numbers = raw.loc[present(raw, column), column].map(as_decimal)
    if any(hint in name for hint in PERCENT_HINTS) and numbers.abs().max() <= 100:
        style = "percent"
    elif any(hint in name for hint in MONEY_HINTS) or numbers.abs().max() < 1000:
        style = "currency"
    else:
        style = "thousands"
    minority = set(rows.tolist())
    def write(index, text):
        value = as_decimal(text)
        if style == "currency":
            return f"{with_thousands(plain(value))} USD" if index in minority else f"${with_thousands(plain(value))}"
        if style == "percent":
            return f"{plain(value)} %" if index in minority else f"{plain(value)}%"
        return plain(value) if index in minority else f"{plain(value / 1000)}k"
    mask = present(raw, column)
    raw.loc[mask, column] = [write(index, text) for index, text in raw.loc[mask, column].items()]
    return {"style": style, "minority_rows": sorted(minority)}

# undoes type 1
def invert_units(values, params):
    def parse(text):
        if text == "":
            return ""
        clean = text.replace("$", "").replace("USD", "").replace("%", "").replace(",", "").strip()
        if params["style"] == "thousands" and clean.endswith("k"):
            return plain(as_decimal(clean[:-1]) * 1000)
        return plain(as_decimal(clean))
    return values.map(parse)

# type 2: the column is written in a base unit ("1.37 kg") and minority cells...
# in a smaller unit ("1370 g")
def corrupt_mixed_units(raw, column, rows, rng):
    family, (base, small, factor) = unit_family(column)
    minority = set(rows.tolist())
    mask = present(raw, column)
    def write(index, text):
        value = as_decimal(text)
        if family == "generic":
            return f"{plain(value / 1000000)}M" if index in minority else f"{plain(value / 1000)}k"
        return f"{plain(value * factor)} {small}" if index in minority else f"{plain(value)} {base}"
    raw.loc[mask, column] = [write(index, text) for index, text in raw.loc[mask, column].items()]
    factors = {"k": 1000, "M": 1000000} if family == "generic" else {base: 1, small: 1 / float(factor)}
    return {"family": family, "units": {unit: float(value) for unit, value in factors.items()}, "minority_rows": sorted(minority)}

# undoes type 2: each unit has its factor to the base unit
def invert_mixed_units(values, params):
    factors = {unit: Decimal(repr(value)) for unit, value in params["units"].items()}
    def parse(text):
        if text == "":
            return ""
        match = re.match(r"^\s*([+-]?[\d.]+)\s*(\S+)\s*$", text)
        return plain(as_decimal(match.group(1)) * factors[match.group(2)])
    return values.map(parse)

# type 3: dates become ISO text; minority cells use dd/mm/yyyy or "Oct 13 2014"
def corrupt_dates(raw, column, rows, rng):
    mask = present(raw, column)
    dates = pd.to_datetime(raw.loc[mask, column], errors="coerce", format="mixed")
    minority = set(rows.tolist())
    styles = ["%d/%m/%Y", "%b %d %Y"]
    written = []
    for index, date in dates.items():
        if pd.isna(date):
            written.append(raw.at[index, column])
        elif index in minority:
            written.append(date.strftime(styles[int(rng.integers(2))]))
        else:
            written.append(date.strftime("%Y-%m-%d"))
    raw.loc[mask, column] = written
    return {"formats": ["%Y-%m-%d"] + styles, "minority_rows": sorted(minority)}

# undoes type 3 (as ISO dates; the comparison is on the date)
def invert_dates(values, params):
    def parse(text):
        if text == "":
            return ""
        for date_format in params["formats"]:
            try:
                return pd.to_datetime(text, format=date_format).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return text
    return values.map(parse)

# the spelling variants of a category label (case, spaces, abbreviation,...
# typo, synonym); claimed holds the keys taken by other labels, so a variant...
# never stands for two labels
def variants(label, others, rng, claimed):
    candidates = [label.upper(), label.lower(), label.title(), f" {label}  ", label + " "]
    if len(label) > 4:
        candidates.append(label[:3] + ".")
    if len(label) >= 5:
        position = int(rng.integers(1, len(label) - 2))
        candidates.append(label[:position] + label[position + 1] + label[position] + label[position + 2:])
    if label.lower() in SYNONYMS:
        candidates.append(SYNONYMS[label.lower()])
    taken = {other.strip().lower() for other in others if other != label} | {key for key, owner in claimed.items() if owner != label}
    chosen = [candidate for candidate in dict.fromkeys(candidates) if candidate != label and candidate.strip().lower() not in taken]
    for candidate in chosen:
        claimed[candidate.strip().lower()] = label
    return chosen

# type 4: some cells of a category column get another spelling of their label
def corrupt_spellings(raw, column, rows, rng):
    labels = raw.loc[present(raw, column), column].unique().tolist()
    claimed = {}
    options = {label: variants(label, labels, rng, claimed) for label in labels}
    mapping = {}
    for index in rows:
        label = raw.at[index, column]
        if options.get(label):
            variant = options[label][int(rng.integers(len(options[label])))]
            raw.at[index, column] = variant
            mapping[variant] = label
    return {"mapping": mapping}

# undoes types 4 and 7 (a spelling map)
def invert_mapping(values, params):
    return values.map(lambda text: params["mapping"].get(text, text))

# type 7: a two-level column is written with yes/no spellings; minority cells...
# use other spellings
def corrupt_booleans(raw, column, rows, rng):
    levels = sorted(raw.loc[present(raw, column), column].unique().tolist())
    positive = next((level for level in levels if level.lower() in {"yes", "true", "1", "y", "t"}), levels[-1])
    negative = next(level for level in levels if level != positive)
    main = int(rng.integers(len(TRUE_SPELLINGS)))
    minority = set(rows.tolist())
    mapping = {}
    for index in raw.index[present(raw, column)]:
        pair = int(rng.integers(len(TRUE_SPELLINGS))) if index in minority else main
        spelling = TRUE_SPELLINGS[pair] if raw.at[index, column] == positive else FALSE_SPELLINGS[pair]
        mapping[spelling] = raw.at[index, column]
        raw.at[index, column] = spelling
    return {"mapping": mapping, "positive": positive, "negative": negative}

# type 8: two numeric columns are packed into one text column ("3 / 2",...
# "3x2", "L:3 W:2"); minority cells drop the spaces; the originals are removed
def corrupt_packed(raw, first, second, rows, rng):
    template = PACK_TEMPLATES[int(rng.integers(len(PACK_TEMPLATES)))]
    name = f"{first}_{second}"
    minority = set(rows.tolist())
    def write(index):
        a, b = raw.at[index, first], raw.at[index, second]
        if a == "" or b == "":
            return ""
        text = template.format(a=a, b=b)
        return text.replace(" ", "") if index in minority else text
    position = list(raw.columns).index(first)
    packed = [write(index) for index in raw.index]
    raw.drop(columns=[first, second], inplace=True)
    raw.insert(position, name, packed)
    return {"template": template, "column": name, "sources": [first, second], "minority_rows": sorted(minority)}

# undoes type 8: the two numbers of each packed value
def invert_packed(values):
    parts = values.map(lambda text: NUMBER.findall(text) if text else ["", ""])
    return parts.str[0], parts.str[1]

# type 5: some cells become a fake missing token (a numeric sentinel outside...
# the column's range, or a text token)
def corrupt_fake_missing(raw, column, rows, rng):
    if is_numeric(raw, column):
        numbers = raw.loc[present(raw, column), column].map(as_decimal)
        if numbers.min() >= 0:
            token = "-999"
        elif numbers.max() < 9999:
            token = "9999"
        else:
            token = "?"
    else:
        token = TEXT_TOKENS[int(rng.integers(len(TEXT_TOKENS)))]
    raw.loc[rows, column] = token
    return {"token": token}

# type 6: some cells get an impossible value: a sign flip or a x100 scale error...
# above the column maximum
def corrupt_impossible(raw, column, rows, rng):
    numbers = raw[column].map(as_decimal)
    top = max(value for value in numbers if value is not None)
    kinds = {}
    for index in rows:
        value = numbers[index]
        if value is None or value == 0:
            continue
        if rng.random() < 0.5 or value * 100 <= top:
            raw.at[index, column] = plain(-value)
            kinds[int(index)] = "sign"
        else:
            raw.at[index, column] = plain(value * 100)
            kinds[int(index)] = "x100"
    return {"kinds": kinds, "max": float(top)}

# type 12: some cells become empty at random (missing completely at random)
def corrupt_plain_missing(raw, column, rows, rng):
    raw.loc[rows, column] = ""
    return {}

# type 9: adds a random customer id and (from level 2) a saved index column
def add_ids(raw, level, rng):
    raw["customer_id"] = [f"C{value:07d}" for value in rng.permutation(len(raw))]
    added = ["customer_id"]
    if level >= 2:
        raw.insert(0, "Unnamed: 0", [str(position) for position in range(len(raw))])
        added.append("Unnamed: 0")
    return {"added": added}

# a seeded random generator for one dataset, seed and type list (and level,...
# when given)
def generator(name, seed, types, level=None):
    key = zlib.crc32(f"{name}|{seed}|{','.join(types)}|{level}".encode("utf-8"))
    return np.random.default_rng([seed, key])

# the column plan of every type at the highest level: columns are handed out...
# one per type and round in a seeded order, so the types share the columns;...
# each column gets at most one type and at most 80% of the features are used
def column_plan(raw, target, features, active, rng):
    kinds = {column: eligible(raw, target, column) for column in features}
    budget = int(np.floor(0.8 * len(features)))
    wanted = {}
    for kind in active:
        count = sum(kind in kinds[column] for column in features)
        size = max(1, int(round(LEVELS[3][0] * count))) if count else 0
        wanted[kind] = (2 * max(1, size // 2) if count >= 2 else 0) if kind == "packed" else size
    plan = {kind: [] for kind in active}
    used = set()
    progress = True
    while progress and len(used) < budget:
        progress = False
        for kind in [active[index] for index in rng.permutation(len(active))]:
            need = 2 if kind == "packed" else 1
            if len(plan[kind]) >= wanted[kind] or len(used) + need > budget:
                continue
            free = [column for column in features if kind in kinds[column] and column not in used]
            if len(free) < need:
                continue
            picked = [free[index] for index in rng.choice(len(free), size=need, replace=False)]
            plan[kind] += picked
            used.update(picked)
            progress = True
    counts = {kind: sum(kind in kinds[column] for column in features) for kind in active}
    return plan, counts

# corrupts a clean raw table at a level (0 clean, 1 low, 2 medium, 3 high):...
# each type uses its share of its eligible columns (25/50/75%, at least one)...
# and of their cells (5/15/30%); the columns and rows of a level are the first...
# ones of a fixed seeded order, so a higher level contains the lower ones;...
# each column gets at most one type, at most 80% of the features are...
# corrupted, the target never; returns the messy table, the ground truth and...
# the original value of every changed cell
def corrupt(clean, target, name, seed, level, types=None):
    types = list(types or MIXED_TYPES)
    raw = clean.copy().reset_index(drop=True).astype(str)
    truth = {"dataset": name, "seed": seed, "level": level, "types": types, "items": []}
    if level == 0:
        return raw, truth, pd.DataFrame(columns=["row", "column", "original", "corrupted", "type"])
    order = generator(name, seed, types)
    rng = generator(name, seed, types, level)
    column_share, cell_share = LEVELS[level]
    features = [column for column in raw.columns if column != target]
    active = [kind for kind in MIXED_TYPES if kind in types and kind != "ids"]
    plan, counts = column_plan(raw, target, features, active, order)
    row_orders = {column: order.permutation(len(raw)) for kind in active for column in plan[kind]}
    if "ids" in types:
        truth["items"].append({"type": "ids", "number": 9, "columns": [], "params": add_ids(raw, level, rng), "expected_op": "drop_columns"})
    for kind in active:
        size = max(1, int(round(column_share * counts[kind]))) if counts[kind] else 0
        if kind == "packed":
            size = 2 * max(1, size // 2) if counts[kind] >= 2 else 0
        chosen = plan[kind][:size]
        pairs = [chosen[index:index + 2] for index in range(0, len(chosen) - 1, 2)] if kind == "packed" else [[column] for column in chosen]
        for columns in pairs:
            mask = np.ones(len(raw), dtype=bool)
            for column in columns:
                mask &= present(raw, column).to_numpy()
            ordered = [row for row in row_orders[columns[0]] if mask[row]]
            count = max(1, int(round(cell_share * len(ordered)))) if ordered else 0
            rows = np.sort(np.asarray(ordered[:count], dtype=int))
            before = raw[columns].copy()
            if kind == "packed":
                params = corrupt_packed(raw, columns[0], columns[1], rows, rng)
                truth["items"].append({"type": kind, "number": 8, "columns": [params["column"]], "sources": columns, "params": params, "expected_op": "extract_pattern", "rows": rows.tolist()})
            else:
                params = globals()[f"corrupt_{kind}"](raw, columns[0], rows, rng)
                truth["items"].append({"type": kind, "number": TYPE_NUMBERS[kind], "columns": columns, "params": params, "expected_op": EXPECTED_OPS[kind], "rows": rows.tolist()})
            truth["items"][-1]["before"] = before
    cells = cell_table(raw, truth)
    return raw, truth, cells

# the original and corrupted value of every changed cell, one row each
def cell_table(raw, truth):
    records = []
    for item in truth["items"]:
        before = item.pop("before", None)
        if before is None:
            continue
        if item["type"] == "packed":
            column = item["columns"][0]
            for source in item["sources"]:
                for row in range(len(raw)):
                    records.append((row, source, before.at[row, source], raw.at[row, column], "packed"))
            continue
        column = item["columns"][0]
        changed = before[column].to_numpy() != raw[column].to_numpy()
        for row in np.flatnonzero(changed):
            records.append((int(row), column, before.at[row, column], raw.at[row, column], item["type"]))
    return pd.DataFrame(records, columns=["row", "column", "original", "corrupted", "type"])

# restores the clean values of a corrupted table from the ground truth: format...
# types are parsed back from the text; types that destroy values (fake...
# missing, impossible, plain missing) use the stored original cells
def restore(raw, truth, cells):
    restored = raw.copy()
    for item in truth["items"]:
        kind = item["type"]
        if kind == "ids":
            restored = restored.drop(columns=item["params"]["added"])
        elif kind == "packed":
            first, second = invert_packed(restored[item["columns"][0]])
            position = list(restored.columns).index(item["columns"][0])
            restored = restored.drop(columns=item["columns"])
            restored.insert(position, item["sources"][1], second.to_numpy())
            restored.insert(position, item["sources"][0], first.to_numpy())
        elif kind in {"spellings", "booleans"}:
            restored[item["columns"][0]] = invert_mapping(restored[item["columns"][0]], item["params"])
        elif kind in {"units", "mixed_units", "dates"}:
            restored[item["columns"][0]] = globals()[f"invert_{kind}"](restored[item["columns"][0]], item["params"])
        else:
            part = cells[(cells["type"] == kind) & (cells["column"] == item["columns"][0])]
            for row, original in zip(part["row"], part["original"]):
                restored.at[row, item["columns"][0]] = original
    return restored

# writes a table with a messy file layout (type 10, reading test E6b): level 1...
# adds a UTF-8 BOM; level 2 also two title rows and a footer note; level 3...
# also uses ";" with decimal commas and the cp1252 encoding
def write_layout(frame, path, level):
    if level <= 0:
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        return
    delimiter = ";" if level >= 3 else ","
    table = frame.copy()
    if level >= 3:
        for column in table.columns:
            numbers = table[column].map(as_decimal)
            if numbers.notna().all() and table[column].str.contains(".", regex=False).any():
                table[column] = table[column].str.replace(".", ",", regex=False)
    body = table.to_csv(index=False, sep=delimiter, lineterminator="\n")
    if level >= 2:
        body = "Export of the customer table\nCreated by the reporting system\n\n" + body + "\nNote: values are preliminary\n"
    if level >= 3:
        path.write_bytes(body.encode("cp1252", errors="replace"))
    else:
        path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
