import json
import numpy as np
import pandas as pd
from decimal import Decimal
from benchmark import splits, materialize
from benchmark.corruptor import as_decimal, corrupt, restore

# a clean raw table with every kind of column the corruptor uses
def clean_table(rows=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "price": [f"{value:.2f}" for value in rng.uniform(100, 5000, rows)],
        "weight_kg": [f"{value:.3f}" for value in rng.uniform(0.5, 9, rows)],
        "lon": [f"{value:.4f}" for value in rng.uniform(-10, 10, rows)],
        "rooms": [str(value) for value in rng.integers(1, 7, rows)],
        "baths": [str(value) for value in rng.integers(1, 4, rows)],
        "age_years": [str(value) for value in rng.integers(18, 90, rows)],
        "city": rng.choice(["Paris", "Berlin", "Madrid", "Rome", "Vienna"], rows),
        "member": rng.choice(["yes", "no"], rows),
        "signup": pd.date_range("2018-01-01", periods=rows).strftime("%Y-%m-%d"),
        "y": rng.choice(["a", "b"], rows),
    }).astype(str)

# two raw values are equal: same text, numbers within 1e-6 relative, or the...
# same day for dates
def same(a, b):
    if a == b:
        return True
    x, y = as_decimal(a), as_decimal(b)
    if x is not None and y is not None:
        return abs(x - y) <= Decimal("1e-6") * max(abs(x), Decimal(1))
    try:
        return pd.to_datetime(a).date() == pd.to_datetime(b).date()
    except (ValueError, TypeError):
        return False

# the same seed gives the same table and cells, byte for byte
def test_determinism():
    clean = clean_table()
    first = corrupt(clean, "y", "toy", 1, 3)
    second = corrupt(clean, "y", "toy", 1, 3)
    assert first[0].to_csv(index=False) == second[0].to_csv(index=False)
    assert first[1] == second[1]
    assert first[2].equals(second[2])
    assert corrupt(clean, "y", "toy", 2, 3)[0].to_csv(index=False) != first[0].to_csv(index=False)

# a higher level corrupts more cells and contains the lower levels' columns
def test_levels_grow():
    clean = clean_table()
    counts, columns = [], []
    for level in [1, 2, 3]:
        messy, truth, cells = corrupt(clean, "y", "toy", 0, level)
        counts.append(len(cells))
        columns.append({column for item in truth["items"] for column in item.get("sources", item["columns"])})
    assert counts[0] < counts[1] < counts[2]
    assert columns[0] <= columns[1] <= columns[2]
    assert len(corrupt(clean, "y", "toy", 0, 0)[2]) == 0

# the inverse of each corruption restores the clean values, the target is...
# never touched and at most 80% of the features are corrupted
def test_recoverable_and_target_untouched():
    clean = clean_table()
    for level in [1, 2, 3]:
        messy, truth, cells = corrupt(clean, "y", "toy", 0, level)
        assert messy["y"].equals(clean["y"])
        back = restore(messy, truth, cells)
        for column in clean.columns:
            assert all(same(a, b) for a, b in zip(clean[column], back[column])), (level, column)
        touched = {column for item in truth["items"] for column in item.get("sources", item["columns"])}
        assert len(touched) <= 0.8 * (clean.shape[1] - 1)
        assert "y" not in touched

# a single type corrupts only with that type; every type is recoverable alone
def test_single_types():
    clean = clean_table()
    for kind in ["units", "mixed_units", "dates", "spellings", "fake_missing", "impossible", "booleans", "packed", "ids", "plain_missing"]:
        messy, truth, cells = corrupt(clean, "y", "toy", 0, 3, [kind])
        assert {item["type"] for item in truth["items"]} == {kind}, kind
        back = restore(messy, truth, cells)
        assert set(back.columns) == set(clean.columns), kind
        assert all(same(a, b) for column in clean.columns for a, b in zip(clean[column], back[column])), kind

# materialize writes the files, the truth with row positions and the cells
def test_materialize_mess(tmp_path, monkeypatch):
    clean = clean_table()
    spec = {"name": "toy", "group": "A", "target": "y", "task": "binary-classification", "sample_rows": None}
    monkeypatch.setattr(materialize, "benchmark_files_dir", tmp_path)
    monkeypatch.setattr(splits, "benchmark_splits_dir", tmp_path / "splits")
    monkeypatch.setattr(materialize, "dataset_spec", lambda name: spec)
    monkeypatch.setattr(materialize, "load_dataset", lambda name: clean)
    folder = materialize.materialize("toy", 2, 0)
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
    assert folder.name == "seed0" and folder.parent.name == "level2"
    assert truth["mess"] and len(truth["positions"]["test"]) == 60 and (folder / "cells.parquet").exists()
