"""Offline label scoring for frozen native candidate snapshots."""
from __future__ import annotations

import json
from pathlib import Path

from turnitover.oracle.candidate_snapshot import digest, jsonl, load_candidate_snapshot


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _parse(raw, taxonomy, observation_count):
    if not isinstance(raw, str):
        raise ValueError("Prediction raw must be a JSON string")
    value = json.loads(raw, object_pairs_hook=_pairs)
    if not isinstance(value, dict) or set(value) != {"verdict"}:
        raise ValueError("Expected one verdict object")
    verdict = value["verdict"]
    if not isinstance(verdict, dict) or verdict.get("status") not in ("pass", "fail", "uncertain"):
        raise ValueError("Invalid verdict status")
    findings = verdict.get("findings")
    if not isinstance(findings, list):
        raise ValueError("Findings must be a list")
    for finding in findings:
        if not isinstance(finding, dict) or finding.get("defect_id") not in taxonomy:
            raise ValueError("Unknown or missing finding taxonomy label")
        if "parts" in finding and (not isinstance(finding["parts"], list)
                or any(not isinstance(part, str) or not part for part in finding["parts"])):
            raise ValueError("Invalid finding parts")
        if finding.get("joint_id") is not None and not isinstance(finding["joint_id"], str):
            raise ValueError("Invalid joint ID")
        if "evidence_observations" in finding:
            observations = finding["evidence_observations"]
            if (not isinstance(observations, list)
                    or any(type(index) is not int or not 1 <= index <= observation_count for index in observations)
                    or len(set(observations)) != len(observations)):
                raise ValueError("Invalid one-based evidence observations")
            converted = [index - 1 for index in observations]
            if "evidence_steps" in finding and sorted(finding["evidence_steps"]) != sorted(converted):
                raise ValueError("Conflicting observation evidence indices")
            finding["evidence_steps"] = converted
        if "evidence_steps" in finding:
            steps = finding["evidence_steps"]
            if (not isinstance(steps, list) or any(type(step) is not int or not 0 <= step < observation_count
                                                for step in steps) or len(set(steps)) != len(steps)):
                raise ValueError("Invalid zero-based observation evidence steps")
    if verdict["status"] == "fail" and not findings:
        raise ValueError("Fail verdict requires a finding")
    if verdict["status"] != "fail" and findings:
        raise ValueError("Only fail verdicts may contain findings")
    return verdict


def _metrics(rows, taxonomy):
    per_label = {}
    for label in taxonomy:
        tp = sum(label in row["expected_labels"] and label in row["predicted_labels"] for row in rows)
        fp = sum(label not in row["expected_labels"] and label in row["predicted_labels"] for row in rows)
        fn = sum(label in row["expected_labels"] and label not in row["predicted_labels"] for row in rows)
        support = sum(label in row["expected_labels"] for row in rows)
        denominator = 2 * tp + fp + fn
        per_label[label] = {"tp": tp, "fp": fp, "fn": fn, "support": support,
                            "f1": 2 * tp / denominator if denominator else 0.0}
    supported = [value["f1"] for value in per_label.values() if value["support"]]
    defective = [row for row in rows if not row["clean"]]
    clean = [row for row in rows if row["clean"]]
    false_passes = sum(row["predicted_status"] == "pass" for row in defective)
    unavailable = {"value": None, "eligible_cases": 0, "total_cases": len(rows),
                   "reason": "Native candidate annotations do not provide uniform audited part/joint localization truth."}
    return {"cases": len(rows), "invalid_predictions": sum(not row["valid"] for row in rows),
            "per_label": per_label, "defect_macro_f1": sum(v["f1"] for v in per_label.values()) / len(taxonomy),
            "supported_label_macro_f1": sum(supported) / len(supported) if supported else None,
            "supported_label_count": len(supported), "taxonomy_label_count": len(taxonomy),
            "zero_denominator_f1": 0.0, "clean_cases": len(clean), "defective_cases": len(defective),
            "false_passes": false_passes,
            "false_pass_rate": false_passes / len(defective) if defective else None,
            "clean_pass_rate": sum(r["predicted_status"] == "pass" for r in clean) / len(clean) if clean else None,
            "localization": unavailable,
            "exposure": {"value": None, "eligible_cases": 0, "total_cases": len(rows),
                         "reason": "No complete semantic per-observation exposure map is frozen; missing evidence is unknown."},
            "evidence_support_rate": None, "unexposed_finding_rate": None}


def score_candidate_snapshot(snapshot, predictions_path, out):
    """Score exact case coverage using only snapshot files and supplied predictions."""
    root, manifest, cases = load_candidate_snapshot(snapshot)
    predictions_path, out = Path(predictions_path), Path(out)
    prediction_bytes = predictions_path.read_bytes()
    predictions = [json.loads(line, object_pairs_hook=_pairs)
                   for line in prediction_bytes.decode("utf-8").splitlines() if line.strip()]
    indexed = {}
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("Prediction row must be an object")
        uid = prediction.get("id", prediction.get("case_id"))
        if (not isinstance(uid, str) or not uid or uid in indexed
                or ("id" in prediction and "case_id" in prediction and prediction["id"] != prediction["case_id"])):
            raise ValueError("Missing, duplicate, or conflicting prediction ID")
        indexed[uid] = prediction
    if set(indexed) != {case["id"] for case in cases}:
        raise ValueError("Predictions must cover every snapshot case exactly once")
    if out.exists():
        raise ValueError("Fresh scoring output required")
    taxonomy = manifest["taxonomy_ids"]
    rows = []
    for case in cases:
        prediction = indexed[case["id"]]
        parsed, error = None, None
        try:
            parsed = _parse(prediction.get("raw"), taxonomy, len(case["observations"]))
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
        rows.append({"id": case["id"], "source_key": case["source_key"], "tier": case["tier"],
                     "clean": case["clean"], "expected_labels": case["labels"],
                     "predicted_labels": sorted({f["defect_id"] for f in parsed["findings"]}) if parsed else [],
                     "predicted_status": parsed["status"] if parsed else None,
                     "valid": parsed is not None, "error": error, "raw": prediction.get("raw"),
                     "parsed": parsed, "usage": prediction.get("usage"),
                     "input_tokens": prediction.get("input_tokens"), "output_tokens": prediction.get("output_tokens"),
                     "human_review_status": case.get("human_review_status"),
                     "training_eligible": case.get("training_eligible"), "formal_admitted": case.get("formal_admitted")})
    report = {"schema": "native-candidate-scores-v1", "status": "complete", "split": manifest["split"],
              "snapshot_manifest_sha256": digest((root / "manifest.json").read_bytes()),
              "predictions_sha256": digest(prediction_bytes), "taxonomy_ids": taxonomy,
              "label_scope": manifest["label_scope"], "metric_scale": "fraction_0_to_1",
              "summary": _metrics(rows, taxonomy),
              "by_tier": {tier: _metrics([r for r in rows if r["tier"] == tier], taxonomy)
                          for tier in sorted({r["tier"] for r in rows})},
              "human_review_status": manifest.get("human_review_status"),
              "training_eligible_cases": manifest.get("training_eligible_cases"),
              "formal_admitted_cases": manifest.get("formal_admitted_cases")}
    out.mkdir(parents=True)
    (out / "predictions.jsonl").write_bytes(prediction_bytes)
    (out / "cases.private.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
