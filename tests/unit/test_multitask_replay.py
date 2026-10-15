from collections import Counter

import pytest

from turnitover.oracle.multitask_replay import replay_groups, stopping_class_weights
from scripts.evaluate_visual_stopping_closed_loop import select_traces
from scripts.audit_multitask_integration import _stopping_per_asset


def test_replay_groups_cover_stopping_once_and_cycle_diagnosis():
    groups = replay_groups(3, 10, seed=7, stopping_per_diagnosis=4)
    assert [len(group["stopping"]) for group in groups] == [4, 4, 2]
    assert sorted(index for group in groups for index in group["stopping"]) == list(range(10))
    assert sorted(group["diagnosis"] for group in groups) == [0, 1, 2]
    assert replay_groups(3, 10, seed=7, stopping_per_diagnosis=4) == groups


def test_stopping_class_weights_balance_total_mass():
    records = [({"stop_target": label}, []) for label in ["stop"] * 3 + ["continue"]]
    weights = stopping_class_weights(records)
    mass = Counter()
    for record, _ in records:
        mass[record["stop_target"]] += weights[record["stop_target"]]
    assert mass["stop"] == pytest.approx(mass["continue"])


def test_closed_loop_selection_stratifies_before_model_inference():
    traces = []
    for index, states in enumerate((
            ["sufficient"] * 6,
            ["identical", "sufficient", "sufficient", "sufficient", "sufficient", "sufficient"],
            ["identical", "identical", "sufficient", "sufficient", "sufficient", "sufficient"],
            ["identical"] * 6,
            ["identical"] * 6,
            ["identical"] * 6,
            ["identical"] * 5 + ["sufficient"],
    )):
        traces.append({"id": f"trace-{index}", "split": "validation",
                       "data_identity": {"asset_id": f"asset-{index}"},
                       "evidence": [{"state": state} for state in states]})
    selected = select_traces(traces)
    assert [row["id"] for row in selected[:3]] == ["trace-0", "trace-1", "trace-2"]
    assert [row["id"] for row in selected[3:]] == ["trace-3", "trace-4", "trace-5"]


def test_stopping_per_asset_separates_active_forced_and_errors():
    rows = [
        {"asset_id": "a", "expected_decision": "stop", "predicted_decision": "continue",
         "termination_kind": "active"},
        {"asset_id": "a", "expected_decision": "continue", "predicted_decision": "stop",
         "termination_kind": "active"},
        {"asset_id": "b", "expected_decision": "stop", "predicted_decision": "stop",
         "termination_kind": "forced"},
    ]
    summary = _stopping_per_asset(rows)
    assert summary["a"] == {
        "n": 2, "correct": 0, "premature_stop": 1, "missed_stop": 1,
        "active_stop_n": 1, "active_stop_correct": 0,
        "forced_stop_n": 0, "forced_stop_correct": 0, "accuracy": 0,
    }
    assert summary["b"]["forced_stop_correct"] == 1
    assert summary["b"]["accuracy"] == 1
