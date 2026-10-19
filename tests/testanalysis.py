import json
import numpy as np
import pandas as pd
from benchmark import analysis, figures

# a synthetic results table: 3 systems on 6 datasets at levels 0 and 3; "good"...
# is best everywhere, "weak" is between dummy and good, one weak run failed
def synthetic():
    rows = []
    for index in range(6):
        metric, dummy = ("rmse", 10.0) if index % 2 else ("roc_auc", 0.5)
        for level in [0, 3]:
            scores = {"good": 0.9 - 0.01 * level, "weak": 0.7 - 0.02 * level, "dummy": 0.5} if metric == "roc_auc" else {"good": 4.0 + level, "weak": 7.0 + level, "dummy": 10.0}
            for system, score in scores.items():
                failed = system == "weak" and index == 0 and level == 3
                rows.append({
                    "system": system, "dataset": f"d{index}", "level": level, "seed": 0, "budget_s": 120.0,
                    "status": "failed" if failed else "ok", "metric": metric, "score": np.nan if failed else score, "dummy_score": dummy,
                    "seconds": 10.0 + index, "input_tokens": 100 if system == "good" else 0, "output_tokens": 10 if system == "good" else 0,
                    "cost_usd": 0.01 if system == "good" else 0.0, "run_dir": "",
                })
    return pd.DataFrame(rows)

# the best system scores 1, the dummy 0, a failed run counts as the dummy
def test_normalized_scores():
    data = analysis.normalized(synthetic())
    assert np.allclose(data.loc[data["system"] == "good", "normalized"], 1.0)
    assert np.allclose(data.loc[data["system"] == "dummy", "normalized"], 0.0)
    failed = data[(data["system"] == "weak") & (data["dataset"] == "d0") & (data["level"] == 3)]
    assert float(failed["normalized"].iloc[0]) == 0.0
    table = analysis.mean_normalized(synthetic())
    assert table.loc["good", 0] == 1.0 and 0 < table.loc["weak", 3] < table.loc["weak", 0] < 1

# ranks, Friedman with the Nemenyi critical difference, Holm and Wilcoxon
def test_tests_and_ranks():
    frame = synthetic()
    ranks = analysis.average_ranks(frame)
    assert ranks.loc["good", 0] == 1.0 and ranks.loc["dummy", 3] > ranks.loc["weak", 3]
    result = analysis.friedman_nemenyi(frame, 0)
    assert result["datasets"] == 6 and np.isclose(result["cd"], 2.343 * np.sqrt(3 * 4 / 36)) and result["p_value"] < 0.05
    assert analysis.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    table = analysis.wilcoxon_table(frame, "good", ["weak", "dummy"], [0, 3])
    assert list(table["wins"]) == [12, 12] and list(table["losses"]) == [0, 0] and (table["p_holm"] < 0.01).all()

# the bootstrap interval holds the mean; reliability counts the failed run
def test_bootstrap_and_reliability():
    frame = synthetic()
    interval = analysis.bootstrap_difference(frame, "good", "weak", [3], resamples=2000)
    assert interval["low"] <= interval["mean"] <= interval["high"] and interval["datasets"] == 6 and interval["low"] > 0
    table = analysis.reliability(frame)
    assert table.loc["weak", "success"] == 11 / 12 and table.loc["good", "mean_cost_usd"] == 0.01 and table.loc["good", "mean_tokens"] == 110

# a dropped leak column is found; other non-structural drops are false drops
def test_leak_outcome(tmp_path):
    decisions = [
        {"column": "refund", "action": "drop", "evidence": {"structural": [], "semantic": [{"code": "G03"}]}},
        {"column": "id", "action": "drop", "evidence": {"structural": [{"code": "E03"}]}},
        {"column": "tenure", "action": "drop", "evidence": {"structural": [], "semantic": [{"code": "G03"}]}},
    ]
    (tmp_path / "leaks.json").write_text(json.dumps(decisions), encoding="utf-8")
    outcome = analysis.leak_outcome(tmp_path, {"leak": {"column": "refund"}})
    assert outcome == {"found": True, "false_drops": 1, "leak": "refund"}

# the figures and tables are written from a results table (PDF, PNG, one MD)
def test_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "figures_dir", tmp_path / "figures")
    monkeypatch.setattr(figures, "tables_file", tmp_path / "tables.md")
    monkeypatch.setattr(figures, "TABLES", {})
    frame = synthetic().replace({"system": {"good": "full", "weak": "generic"}})
    figures.score_vs_level(frame)
    result = figures.cd_diagram(pd.concat([frame, frame[frame["system"] == "generic"].assign(system="rules", score=lambda part: part["score"] * 1.001)]), 0)
    figures.cost_vs_score(frame)
    figures.ablations(frame, ["d0", "d1"])
    figures.write_tables()
    names = {path.name for path in (tmp_path / "figures").iterdir()}
    assert {"score-vs-level.pdf", "score-vs-level.png", "cd-level0.png", "cost-vs-score.png", "ablations.png"} <= names
    assert result["ranks"]["full"] == 1.0 and "## cost-vs-score" in (tmp_path / "tables.md").read_text(encoding="utf-8")
