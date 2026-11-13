"""Recompute paper metrics from complete predictions and private frozen inputs."""
from __future__ import annotations

from collections import defaultdict
import json

from turnitover.oracle.paired_observations import OBSERVATION_FORMAT, is_paired, pair_support
from turnitover.oracle.sft_metrics import score_prediction, summarize, summarize_by_difficulty
from turnitover.verifier.contracts import parse_json


def _index(rows, kind):
    result = {}
    for row in rows:
        uid = row.get("id")
        if not isinstance(uid, str) or not uid or uid in result:
            raise ValueError(f"Missing or duplicate {kind} ID")
        result[uid] = row
    return result


def _signature(label):
    return (label["defect_id"],
            label.get("params", {}).get("joint") if label["defect_id"] == "kinematics.joint_axis" else None,
            tuple(sorted(label["parts"])))


def _finding(value):
    return value[0], value[1], tuple(sorted(value[2]))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _injected_summary(rows, defect_ids):
    report = summarize(rows, "diagnosis", defect_ids)
    # Status targets describe observable evidence, not the hidden injection set.
    return {key: report[key] for key in ("n", "defect_macro_f1", "defect_labels",
                                        "defect_per_class", "part_localization_micro")}


def _paper_summary(cases):
    predicted = sum(len(case["predicted"]) for case in cases)
    supported = 0
    sources = set()
    for case in cases:
        sources.add(case["support_source"])
        for evidence in case["predicted"]:
            sig = _finding(evidence["finding"])
            support = case["support"].get(sig, set())
            supported += bool(support) and set(evidence["steps"]) == support
    complete = all(case["complete_exposure"] for case in cases)
    defective = [case for case in cases if case["labels"]]
    any_exposed = sum(bool(case["exposed_indices"]) for case in defective) if complete else None
    all_exposed = sum(len(case["exposed_indices"]) == len(case["labels"])
                      for case in defective) if complete else None
    unexposed = (sum(evidence["finding"][0] not in case["exposed_types"]
                     for case in cases for evidence in case["predicted"])
                 if complete else None)
    reason = None if complete else "Complete per-observation exposure mapping is unavailable."
    return {"n": len(cases), "predicted_findings": predicted,
            "strict_supported_findings": supported,
            "strict_evidence_support_rate": _ratio(supported, predicted),
            "strict_evidence_support_definition": "Correct full finding signature and cited steps exactly equal all supporting shown steps; denominator is all parsed predicted findings.",
            "strict_evidence_support_source": sorted(sources),
            "defective_cases": len(defective),
            "exposure_any_count": any_exposed, "exposure_all_count": all_exposed,
            "exposure_any_rate": _ratio(any_exposed, len(defective)) if complete else None,
            "exposure_all_rate": _ratio(all_exposed, len(defective)) if complete else None,
            "exposure_definition": "Any/all injected label indices exposed in shown observations; clean cases excluded.",
            "unexposed_findings": unexposed,
            "unexposed_finding_rate": _ratio(unexposed, predicted) if complete else None,
            "unexposed_finding_definition": "Predicted defect type has no exposed injected label of the same type in shown observations; denominator is all parsed predicted findings, regardless of part or joint match.",
            "exposure_available": complete, "exposure_unavailable_reason": reason,
            "unexposed_unavailable_reason": reason}


def score_snapshot(records, annotations, predictions, defect_ids, task="diagnosis"):
    """Score raw outputs, retaining legacy targets separately from injection truth.

    Annotation rows may cover the entire dataset. Prediction IDs must exactly
    cover the supplied evaluation records. No previously calculated metric is
    trusted. A paired exposure table is validated even if no finding is emitted.
    """
    if task not in {"diagnosis", "stopping"}:
        raise ValueError("Unknown scoring task")
    defect_ids = list(defect_ids)
    records = list(records)
    by_id = _index(records, "record")
    predictions = _index(predictions, "prediction")
    if set(predictions) != set(by_id):
        raise ValueError("Predictions must exactly cover evaluation record IDs")
    if not records:
        raise ValueError("Cannot score an empty evaluation")
    rows = []
    for record in records:
        raw = predictions[record["id"]].get("raw")
        if not isinstance(raw, str):
            raise ValueError("Prediction raw output must be a string")
        rows.append(score_prediction(record, raw, task))
    target = summarize(rows, task, defect_ids)
    tiers = summarize_by_difficulty(rows, task, defect_ids)
    result = {**target, "by_difficulty": tiers,
              "target_metrics": {"overall": target, "by_difficulty": tiers}}
    if task == "stopping":
        return result
    annotations = _index(annotations.values() if isinstance(annotations, dict) else annotations,
                         "annotation")
    if not defect_ids or len(set(defect_ids)) != len(defect_ids):
        raise ValueError("Need a nonempty unique frozen defect taxonomy")
    injected_rows, cases = [], []
    for record, row in zip(records, rows, strict=True):
        annotation = annotations.get(record["episode_id"])
        if annotation is None:
            raise ValueError("Record lacks an annotation")
        labels = annotation["labels"]
        if any(label["defect_id"] not in defect_ids for label in labels):
            raise ValueError("Injected label is outside the frozen defect taxonomy")
        expected = sorted({_signature(label) for label in labels}, key=repr)
        injected_rows.append({**row, "expected_findings": [list(sig) for sig in expected]})
        support = defaultdict(set)
        exposed = set()
        complete = is_paired(record)
        if complete:
            payload = json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])
            if (payload["observation_format"] != OBSERVATION_FORMAT
                    or [item["step"] for item in payload["history"]] != [0, 1, 2]):
                raise ValueError("Paired exposure mapping requires shown pair steps 0/1/2")
            evidence = record["evidence"]
            covered = evidence.get("covered_label_indices")
            if (not isinstance(covered, list) or len(set(covered)) != len(covered)
                    or any(type(index) is not int or not 0 <= index < len(labels) for index in covered)):
                raise ValueError("Invalid covered label indices")
            per_label = pair_support(evidence, len(labels), covered)
            for index, steps in per_label.items():
                support[_signature(labels[index])].update(steps)
            exposed = set(per_label)
            source = "audited_pair_label_indices"
        else:
            for evidence in row["expected_evidence"]:
                support[_finding(evidence["finding"])].update(evidence["steps"])
            source = "target_expected_evidence_not_complete_exposure_audit"
        predicted = ([{"finding": [finding["defect_id"], finding["joint_id"], finding["parts"]],
                       "steps": finding["evidence_steps"]}
                      for finding in parse_json(row["raw"])["verdict"]["findings"]]
                     if row["valid"] else [])
        cases.append({"row": row, "predicted": predicted, "labels": labels, "support": support,
                      "support_source": source, "complete_exposure": complete,
                      "exposed_indices": exposed,
                      "exposed_types": {labels[index]["defect_id"] for index in exposed}})
    injected_tiers = {}
    paper_tiers = {}
    for tier in ("easy", "medium", "hard"):
        selected = [row for row in injected_rows if row.get("difficulty") == tier]
        if selected:
            injected_tiers[tier] = _injected_summary(selected, defect_ids)
            paper_tiers[tier] = _paper_summary([case for case in cases
                                              if case["row"].get("difficulty") == tier])
    paper = _paper_summary(cases)
    result["paper_metrics"] = {**paper, "overall": paper, "by_difficulty": paper_tiers,
                               "all_injected_metrics": {
                                   "overall": _injected_summary(injected_rows, defect_ids),
                                   "by_difficulty": injected_tiers}}
    return result
