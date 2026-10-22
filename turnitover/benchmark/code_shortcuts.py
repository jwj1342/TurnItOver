"""Read-only audit of code-level shortcuts: how well do simple rules on the candidate code alone tell defect from clean?

A sample is the candidate AssetSpec (what ``emit_program`` turns into the code a model reads); the reference is never
used. Balanced accuracy is the metric (mean of defect recall and clean recall) so that chance is 0.5 whatever the
clean/defect ratio. Fixed rules are stated a priori and reported at their best direction, ``max(b, 1 - b)``, which is
what an attacker who flips the rule on a few labelled samples would get. Fitted rules learn a class-conditional
likelihood from the other samples (leave-one-out) and are reported as they are.
No model is called and no dataset file is written.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from turnitover.corruptions.kinematics import coordinate_axis

TIDY_DECIMALS = 3
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261022
FLAG_THRESHOLD = 0.65  # best-direction balanced accuracy at or above this is reported as clearly above chance


def joint_features(spec: dict) -> list[dict]:
    """Per joint: axis, sign, anchor, limits and the origin of the part it moves."""
    origins = {p["id"]: p["origin"] for p in spec["parts"]}
    return [{"id": j["id"], "axis": j["axis"], "sign": j["sign"], "anchor": j["anchor"], "limits": j["limits"],
             "part_origin": origins[j["part"]]} for j in spec["joints"]]


def _signature(joint: dict) -> str:
    coordinate = coordinate_axis(joint["axis"])
    return "off" if coordinate is None else f"{'xyz'[coordinate[0]]}{'+' if coordinate[1] > 0 else '-'}"


def _tidy(values) -> bool:
    return all(round(float(v), TIDY_DECIMALS) == float(v) for v in values)


def _axis_off_coordinate(joints) -> bool:
    return any(coordinate_axis(j["axis"]) is None for j in joints)


def _sign_negative(joints) -> bool:
    return any(j["sign"] < 0 for j in joints)


def _axis_dominant_negative(joints) -> bool:
    return any(j["axis"][int(np.argmax(np.abs(j["axis"])))] < 0 for j in joints)


def _origin_tidy(joints) -> bool:
    return all(_tidy(j["anchor"]) and _tidy(j["part_origin"]) for j in joints)


FIXED_RULES = {
    "axis_off_coordinate": _axis_off_coordinate,
    "sign_negative": _sign_negative,
    "axis_dominant_negative": _axis_dominant_negative,
    "origin_tidy": _origin_tidy,
}
FITTED_KEYS = {
    "axis_signature_prior": lambda joints: tuple(sorted(_signature(j) for j in joints)),
    "axis_index_prior": lambda joints: tuple(sorted(_signature(j)[0] for j in joints)),
}


def balanced_accuracy(truth: np.ndarray, predicted: np.ndarray) -> float:
    return 0.5 * (float(np.mean(predicted[truth])) + float(np.mean(~predicted[~truth])))


def bootstrap_interval(truth: np.ndarray, predicted: np.ndarray) -> list[float]:
    """95% percentile interval of balanced accuracy, resampling defects and cleans separately (fixed seed)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    hits, correct = predicted[truth], ~predicted[~truth]
    draws = [0.5 * (rng.choice(hits, len(hits)).mean() + rng.choice(correct, len(correct)).mean())
             for _ in range(BOOTSTRAP)]
    return [round(float(x), 3) for x in np.percentile(draws, [2.5, 97.5])]


def loo_predictions(keys: list, truth: np.ndarray) -> np.ndarray:
    """Predict defect when the sample's key is relatively more common among defects than among cleans.

    The sample itself is left out of the counts. Ties predict clean.
    """
    defect, clean = Counter(k for k, t in zip(keys, truth) if t), Counter(k for k, t in zip(keys, truth) if not t)
    n_defect, n_clean = int(truth.sum()), int((~truth).sum())
    predicted = []
    for key, is_defect in zip(keys, truth):
        d, c = defect[key] - is_defect, clean[key] - (not is_defect)
        predicted.append(d / (n_defect - is_defect) > c / (n_clean - (not is_defect)))
    return np.array(predicted)


def _report(truth: np.ndarray, predicted: np.ndarray, *, fixed: bool) -> dict:
    b = balanced_accuracy(truth, predicted)
    if fixed and b < 0.5:
        predicted, b = ~predicted, 1 - b  # best direction
    return {"balanced_accuracy": round(b, 3), "ci95": bootstrap_interval(truth, predicted),
            "defect_recall": round(float(np.mean(predicted[truth])), 3),
            "clean_recall": round(float(np.mean(~predicted[~truth])), 3),
            "above_chance": b >= FLAG_THRESHOLD}


def audit_stratum(clean: list[dict], defects: list[dict]) -> dict:
    """Rules on ``clean`` plus ``defects`` samples; each sample is ``{"joints": [...]}``."""
    samples = clean + defects
    truth = np.array([False] * len(clean) + [True] * len(defects))
    rules = {name: _report(truth, np.array([rule(s["joints"]) for s in samples]), fixed=True)
             for name, rule in FIXED_RULES.items()}
    for name, key in FITTED_KEYS.items():
        rules[name] = _report(truth, loo_predictions([key(s["joints"]) for s in samples], truth), fixed=False)
    best = max(rules, key=lambda r: rules[r]["balanced_accuracy"])
    return {"n_defect": len(defects), "n_clean": len(clean), "rules": rules, "best_rule": best,
            "best_balanced_accuracy": rules[best]["balanced_accuracy"],
            "above_chance": [r for r, v in rules.items() if v["above_chance"]]}


def audit_dataset(samples: list[dict]) -> dict:
    """``samples``: ``{"label": "clean" | defect id, "stratum": str | None, "joints": [...]}`` -> per-stratum results."""
    clean = [s for s in samples if s["label"] == "clean"]
    groups: dict[tuple, list[dict]] = {}
    for s in samples:
        if s["label"] != "clean":
            groups.setdefault((s["label"], s["stratum"]), []).append(s)
    return {f"{defect}/{stratum}": audit_stratum(clean, rows) for (defect, stratum), rows in sorted(groups.items())}
