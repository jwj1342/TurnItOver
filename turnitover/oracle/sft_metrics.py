"""Pure scoring for frozen observations; invalid predictions retain their rows."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import random

from turnitover.oracle.sft_contracts import parse_stopping
from turnitover.verifier.contracts import Context, parse_decision, parse_json
from turnitover.taxonomy import load_taxonomy

PROTOCOL_VERSION = "fixed-observation-v1"


def public_input_digest(records):
    values = [{"id": row["id"], "user": row["messages"][0],
               "images": row["images"], "image_sha256": row["image_sha256"]} for row in records]
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def object_id(record):
    """The original source object, shared by every corruption and episode."""
    return record["data_identity"]["source_identity_sha256"]


def signature(finding):
    return finding.defect_id, finding.joint_id, tuple(sorted(finding.parts))


def score_prediction(record, raw, task="diagnosis"):
    row = {"id": record["id"], "episode_id": record["episode_id"],
           "object_id": object_id(record), "data_identity": record["data_identity"],
           "raw": raw, "task": task, "error": None}
    target_value = json.loads(record["messages"][1]["content"])
    if task == "stopping":
        target = parse_stopping(target_value)
        try:
            prediction = parse_stopping(raw)
        except ValueError as exc:
            prediction = None
            row["error"] = str(exc)
        row.update(expected=target, predicted=prediction, valid=prediction is not None)
        return row
    payload = json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])
    context = Context(**payload["context"])
    observed = {item["step"] for item in payload["history"]}
    target = parse_decision(target_value, context, observed, final_only=True)
    try:
        prediction = parse_decision(parse_json(raw), context, observed, final_only=True)
    except ValueError as exc:
        prediction = None
        row["error"] = str(exc)
    expected, predicted = {}, {}
    for finding in target.findings:
        expected.setdefault(signature(finding), set()).update(finding.evidence_steps)
    for finding in prediction.findings if prediction else ():
        predicted.setdefault(signature(finding), set()).update(finding.evidence_steps)
    # One supplied joint is a closed choice, never an open localization success.
    given_joint = (len(context.joints) <= 1 or payload.get("target_joint_id") is not None
                   or payload.get("joint_id") is not None)
    row.update(valid=prediction is not None, expected_status=target.status,
               predicted_status=prediction.status if prediction else None,
               expected_findings=[list(item) for item in expected],
               predicted_findings=[list(item) for item in predicted],
               supported_findings=sum(key in expected and steps <= expected[key]
                                      for key, steps in predicted.items()),
               open_joint_localization=not given_joint,
               clean=record.get("is_clean") if type(record.get("is_clean")) is bool else None,
               expected_evidence=[{"finding": list(k), "steps": sorted(v)} for k, v in expected.items()],
               predicted_evidence=[{"finding": list(k), "steps": sorted(v)} for k, v in predicted.items()])
    return row


def _ratio(n, d):
    return n / d if d else None


def _prf(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn, "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn), "f1": _ratio(2 * tp, 2 * tp + fp + fn)}


def _sets(row, field):
    return {(f[0], f[1], tuple(f[2])) for f in row[field]}


def _set_metric(rows, transform, *, joints_only=False):
    counts = Counter()
    for row in rows:
        expected = {transform(f) for f in _sets(row, "expected_findings") if not joints_only or f[1] is not None}
        predicted = {transform(f) for f in _sets(row, "predicted_findings") if not joints_only or f[1] is not None}
        counts.update(tp=len(expected & predicted), fp=len(predicted - expected), fn=len(expected - predicted))
    return _prf(counts["tp"], counts["fp"], counts["fn"])


def summarize(rows, task="diagnosis", defect_ids=None):
    if not rows:
        raise ValueError("Cannot score an empty evaluation")
    if any(row["task"] != task for row in rows):
        raise ValueError("Do not mix diagnosis and stopping summaries")
    n = len(rows)
    report = {"n": n, "independent_objects": len({row["object_id"] for row in rows}),
              "valid_output_rate": sum(row["valid"] for row in rows) / n,
              "invalid_outputs": sum(not row["valid"] for row in rows)}
    if task == "stopping":
        report.update(decision_accuracy=sum(row["valid"] and row["predicted"]["decision"] == row["expected"]["decision"] for row in rows) / n,
                      exact_stopping_accuracy=sum(row["valid"] and all(row["predicted"][k] == row["expected"][k] for k in ("decision", "outcome")) for row in rows) / n,
                      premature_stop_rate=sum(row["valid"] and row["predicted"]["decision"] == "stop" and row["expected"]["decision"] == "continue" for row in rows) / n)
        return report
    labels = sorted(defect_ids if defect_ids is not None else load_taxonomy().ids)
    per_defect = {}
    for label in labels:
        tp = fp = fn = 0
        for row in rows:
            expected = any(f[0] == label for f in row["expected_findings"])
            predicted = any(f[0] == label for f in row["predicted_findings"])
            tp += expected and predicted
            fp += predicted and not expected
            fn += expected and not predicted
        per_defect[label] = _prf(tp, fp, fn)
    f1s = [value["f1"] if value["f1"] is not None else 0.0 for value in per_defect.values()]
    predicted_count = sum(len(row["predicted_findings"]) for row in rows)
    clean_rows = [row for row in rows if row["clean"] is True]
    unsafe_rows = [row for row in rows if row["expected_status"] != "pass"]
    joint_rows = [row for row in rows if row["open_joint_localization"]]
    report.update(status_accuracy=sum(row["predicted_status"] == row["expected_status"] for row in rows) / n,
                  exact_verdict_accuracy=sum(row["valid"] and row["predicted_status"] == row["expected_status"] and _sets(row, "expected_findings") == _sets(row, "predicted_findings") for row in rows) / n,
                  defect_macro_f1=sum(f1s) / len(f1s) if f1s else None,
                  defect_labels=labels, defect_per_class=per_defect,
                  finding_micro=_set_metric(rows, lambda f: f),
                  part_localization_micro=_set_metric(rows, lambda f: (f[0], f[2])),
                  joint_localization_micro=_set_metric(joint_rows, lambda f: (f[0], f[1]), joints_only=True) if joint_rows else None,
                  open_joint_cases=len(joint_rows),
                  evidence_support_rate=_ratio(sum(row["supported_findings"] for row in rows), predicted_count),
                  predicted_findings=predicted_count,
                  evidence_supported_findings=sum(row["supported_findings"] for row in rows),
                  clean_false_positive_count=sum(row["predicted_status"] == "fail" for row in clean_rows),
                  clean_cases=len(clean_rows), clean_false_positive_rate=_ratio(sum(row["predicted_status"] == "fail" for row in clean_rows), len(clean_rows)),
                  false_pass_cases=len(unsafe_rows),
                  false_pass_count=sum(row["predicted_status"] == "pass" for row in unsafe_rows),
                  false_pass_rate=_ratio(sum(row["predicted_status"] == "pass" for row in unsafe_rows), len(unsafe_rows)))
    return report


def object_bootstrap(rows, metric="status_accuracy", *, samples=1000, seed=0, task="diagnosis", defect_ids=None):
    """Resample original objects with replacement, keeping all their cases together."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[row["object_id"]].append(row)
    if len(groups) < 2:
        return {"samples": samples, "seed": seed, "independent_objects": len(groups), "ci95": None}
    labels = sorted(defect_ids if defect_ids is not None else load_taxonomy().ids)
    rng = random.Random(seed)
    keys = sorted(groups)
    values = []
    for _ in range(samples):
        sample = [row for key in rng.choices(keys, k=len(keys)) for row in groups[key]]
        value = summarize(sample, task, labels).get(metric)
        if value is not None:
            values.append(value)
    values.sort()
    return {"samples": samples, "seed": seed, "unit": "original_source_object", "independent_objects": len(groups),
            "metric": metric, "ci95": [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]] if values else None}
