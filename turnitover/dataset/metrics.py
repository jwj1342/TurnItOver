"""Shared multilabel detection counts; missing predictions remain false negatives."""
from collections import Counter


def count_labels(classes: dict[str, Counter], expected: set[str], predicted: set[str]) -> None:
    for label in expected | predicted:
        counts = classes.setdefault(label, Counter())
        outcome = "tp" if label in expected & predicted else "fp" if label in predicted else "fn"
        counts[outcome] += 1


def detection_metrics(classes: dict[str, Counter]) -> dict[str, dict]:
    result = {}
    for label, counts in classes.items():
        tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
        result[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        }
    return result
