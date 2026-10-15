"""Metrics shared by visual stopping evaluation and metadata baselines."""
from __future__ import annotations

from collections import Counter


def summarize_stopping(rows: list[dict]) -> dict:
    """Summarize stop as the positive class while keeping forced stops separate."""
    n = len(rows)
    if not n:
        return {"n": 0}
    tp = sum(row["expected_decision"] == row["predicted_decision"] == "stop" for row in rows)
    fp = sum(row["expected_decision"] == "continue" and row["predicted_decision"] == "stop"
             for row in rows)
    fn = sum(row["expected_decision"] == "stop" and row["predicted_decision"] != "stop"
             for row in rows)
    tn = sum(row["expected_decision"] == row["predicted_decision"] == "continue" for row in rows)
    invalid = sum(row["predicted_decision"] not in ("continue", "stop") for row in rows)
    expected_stop = tp + fn
    predicted_stop = tp + fp
    confusion = Counter(
        (row["expected_decision"], row["predicted_decision"] or "invalid") for row in rows)
    return {
        "n": n,
        "accuracy": (tp + tn) / n,
        "stop_precision": tp / predicted_stop if predicted_stop else None,
        "stop_recall": tp / expected_stop if expected_stop else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "invalid": invalid,
        "premature_stop": fp,
        "missed_stop": fn,
        "confusion": {f"{a}->{b}": value for (a, b), value in sorted(confusion.items())},
        "active_stop": _stop_slice(rows, "active"),
        "forced_budget_stop": _stop_slice(rows, "forced"),
    }


def _stop_slice(rows: list[dict], kind: str) -> dict:
    selected = [row for row in rows if row.get("termination_kind") == kind]
    correct = sum(row["predicted_decision"] == "stop" for row in selected)
    outcome_correct = sum(
        row["predicted_decision"] == "stop"
        and row.get("predicted_outcome") == row.get("expected_outcome") for row in selected)
    return {
        "n": len(selected),
        "decision_correct": correct,
        "decision_recall": correct / len(selected) if selected else None,
        "outcome_correct": outcome_correct,
        "outcome_accuracy": outcome_correct / len(selected) if selected else None,
    }


def summarize_pairs(rows: list[dict], pairs: list[dict], records: dict[str, dict]) -> dict:
    """Score strict pairs and verify equality of all model-visible non-image input."""
    predictions = {row["id"]: row for row in rows}
    nonvisual_equal = 0
    correct = 0
    opposite = 0
    missing = 0
    for pair in pairs:
        stop_id, continue_id = pair["stop_id"], pair["continue_id"]
        left, right = records[stop_id], records[continue_id]
        same = (
            left["messages"][0]["content"] == right["messages"][0]["content"]
            and left["decision_index"] == right["decision_index"]
            and len(left["images"]) == len(right["images"])
        )
        nonvisual_equal += same
        if stop_id not in predictions or continue_id not in predictions:
            missing += 1
            continue
        stop_prediction = predictions[stop_id]["predicted_decision"]
        continue_prediction = predictions[continue_id]["predicted_decision"]
        opposite += {stop_prediction, continue_prediction} == {"stop", "continue"}
        correct += stop_prediction == "stop" and continue_prediction == "continue"
    n = len(pairs)
    return {
        "n": n,
        "complete_prediction_pairs": n - missing,
        "all_model_visible_nonimage_input_equal": nonvisual_equal == n,
        "nonvisual_equal_pairs": nonvisual_equal,
        "opposite_decisions": opposite,
        "opposite_decision_rate": opposite / n if n else None,
        "correctly_distinguished": correct,
        "correctly_distinguished_rate": correct / n if n else None,
        "definition": "Both members must be correct: stop/fail evidence versus continue/no evidence.",
    }
