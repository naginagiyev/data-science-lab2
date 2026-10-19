# Showcase

One messy file, three systems, the same budget. Command (the LLM answers come from the pilot's cache,
so the demo costs almost nothing):

```
uv run python -m benchmark.showcase
```

The file is the level-3 pilot file of `concrete_compressive_strength` (824 training rows): the mess
includes a saved index and an id column, the cement amount written like money ("$393"), the fine
aggregate with magnitude suffixes ("0.7856k", "0.0007893M"), and the age and water amounts packed into
one column ("L:28 W:192.0", sometimes without the space).
Output of a real run on 2026-10-14 (Windows on ARM, x86_64 emulation, 120 s per system):

```
concrete_compressive_strength, mess level 3, 120 s per system, metric rmse (dummy 15.2567):
  autogluon  ok      12.9251   LLM cost $0.0000
  full       ok      4.1062   LLM cost $0.0008
  full+ag    ok      4.2035   LLM cost $0.0027

Main decisions of the full run:
  - Unnamed: 0 dropped (drop: id or saved index column (structural))
  - customer_id dropped (drop: id or saved index column (structural))
  - drop_conflicting_rows: kept by the gate
  - group_rare FineAggregate: removed by the gate
  - parse_number Cement: kept by the gate
  - parse_units FineAggregate: kept by the gate
  - extract_pattern Age_Water: kept by the gate
```

AutoGluon reads the text columns as categories and cannot use the amounts inside them, so it is only a
little better than the dummy model. `full` turns the cement and fine-aggregate amounts back into
numbers, splits the packed age and water values into two numeric columns, drops the two id columns,
and its gate removes a proposed `group_rare` step that made the cross-validated score worse. With
the same prepared data, AutoGluon as the engine (`full+ag`) reaches almost the same score.

The decision list above was printed by `benchmark/showcase.py` from the same run folder; the run
folder of `full` (`report.md`, `decisions.jsonl`, `plans.json`, `gate.json`) holds the full evidence.
This is one file; the pilot results over 8 datasets are in `docs/results.md`.
