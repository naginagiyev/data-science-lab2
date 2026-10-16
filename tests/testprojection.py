import numpy as np
import pandas as pd
from benchmark import projection

DATASETS = ["E-CommereShippingData", "concrete_compressive_strength", "a", "b", "c", "d", "e", "f"]
SYSTEMS = {"dummy": 0.0, "lightgbm": 0.20, "skrub": 0.22, "autogluon": 0.25, "generic": 0.24, "full": 0.30, "full+ag": 0.31}

# synthetic pilot results (roc_auc): full is ahead and the gap grows with the...
# level; seed 1 moves the free systems a little
def results(seed=0, systems=SYSTEMS, levels=(0, 2, 3)):
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in DATASETS:
        for level in levels:
            for system, base in systems.items():
                gain = 0.0 if system in {"dummy"} else base - 0.03 * level * (system not in {"full", "full+ag"})
                rows.append({
                    "system": system, "dataset": dataset, "group": "A", "level": level, "seed": seed, "mess_types": "mixed", "budget_s": 120.0,
                    "status": "ok", "metric": "roc_auc", "score": 0.5 + gain + rng.normal(0, 0.005) * (system != "dummy"), "dummy_score": 0.5,
                    "seconds": 60.0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.03 if system.startswith("full") else 0.0, "run_dir": "", "secondary": "",
                })
    return pd.DataFrame(rows)

# the projection: positive effects with intervals, high power, held-out...
# predictions checked against the actual scores, the same order at both...
# budgets, and a generated report with a recommendation and grid estimates
def test_projection(tmp_path):
    main = results()
    free = {name: value for name, value in SYSTEMS.items() if name in {"dummy", "lightgbm", "skrub", "autogluon", "generic"}}
    seed = results(1, free, (0, 3))
    budget = results(0, {"dummy": 0.0, "full": 0.3, "autogluon": 0.25}, (3,))
    budget = budget[budget["dataset"].isin(DATASETS[:2])].assign(budget_s=300.0)
    runs = [main.assign(grid="pilot-main"), seed.assign(grid="pilot-seed"), budget.assign(grid="pilot-budget")]
    pd.concat(runs, ignore_index=True).to_csv(tmp_path / "runs.csv", index=False)
    predictions = projection.holdout_predictions(main)
    assert len(predictions) == 4 and (predictions["errors_from"] == 6).all()
    predictions.to_csv(tmp_path / "holdout-predictions.csv", index=False)
    result = projection.build(tmp_path, tmp_path / "projection.md", repetitions=200)
    full_vs_autogluon = result["effects"][("full", "autogluon", "2,3")]
    assert full_vs_autogluon["low"] > 0 and full_vs_autogluon["low_seed"] <= full_vs_autogluon["low"]
    assert result["powers"]["full vs autogluon (levels 2+3)"] > 0.8 and 0 < result["noise"] < 0.2
    assert len(result["holdout"]) == 4 and result["holdout"]["actual"].notna().all() and result["holdout"]["inside"].dtype == bool
    assert result["budget"]["full_ahead"].all()
    text = (tmp_path / "projection.md").read_text(encoding="utf-8")
    assert "Held-out check:" in text and "- RQ1 (messy data):" in text and "| main | 1440 |" in text
