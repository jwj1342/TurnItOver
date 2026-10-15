from turnitover.oracle.stopping_metrics import summarize_pairs, summarize_stopping


def test_stopping_metrics_keep_active_and_forced_stops_separate():
    rows = [
        {"expected_decision": "continue", "predicted_decision": "stop",
         "expected_outcome": None, "predicted_outcome": "fail", "termination_kind": None},
        {"expected_decision": "stop", "predicted_decision": "stop",
         "expected_outcome": "fail", "predicted_outcome": "fail", "termination_kind": "active"},
        {"expected_decision": "stop", "predicted_decision": "continue",
         "expected_outcome": "uncertain", "predicted_outcome": None, "termination_kind": "forced"},
        {"expected_decision": "continue", "predicted_decision": "continue",
         "expected_outcome": None, "predicted_outcome": None, "termination_kind": None},
    ]
    result = summarize_stopping(rows)
    assert result["accuracy"] == .5
    assert result["stop_precision"] == .5
    assert result["stop_recall"] == .5
    assert result["premature_stop"] == 1
    assert result["missed_stop"] == 1
    assert result["active_stop"]["decision_recall"] == 1
    assert result["forced_budget_stop"]["decision_recall"] == 0


def test_pair_metric_requires_correct_orientation_and_checks_visible_text():
    records = {
        "s": {"messages": [{"content": "same"}], "decision_index": 2, "images": ["a", "b"]},
        "c": {"messages": [{"content": "same"}], "decision_index": 2, "images": ["x", "y"]},
    }
    pairs = [{"stop_id": "s", "continue_id": "c"}]
    correct = summarize_pairs(
        [{"id": "s", "predicted_decision": "stop"},
         {"id": "c", "predicted_decision": "continue"}], pairs, records)
    assert correct["all_model_visible_nonimage_input_equal"] is True
    assert correct["correctly_distinguished_rate"] == 1
    swapped = summarize_pairs(
        [{"id": "s", "predicted_decision": "continue"},
         {"id": "c", "predicted_decision": "stop"}], pairs, records)
    assert swapped["opposite_decision_rate"] == 1
    assert swapped["correctly_distinguished_rate"] == 0
