"""Deterministic post-hoc scoring for frozen verifier runs."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re

from turnitover.core.jsonio import write_json_atomic
from turnitover.verifier.contracts import VERIFIER_SCHEMA_VERSION


def score_verifier(gold_path: Path, results_root: Path, output_path: Path) -> dict:
    """Compare ``<results_root>/<case id>/result.json`` with private JSONL gold."""
    if output_path.exists():
        raise ValueError("Scoring output must not exist")
    gold = _read_gold(gold_path)
    counts = Counter(n=len(gold))
    classes: dict[str, Counter] = {}
    rows = []
    token_totals = Counter()
    latency_ms = 0.0
    observations = budget_spent = 0
    token_cases = latency_cases = 0

    for case_id, target in gold.items():
        expected = _signatures(target["findings"])
        expected_ids = {item[0] for item in expected}
        is_defect = bool(expected)
        counts["defect" if is_defect else "clean"] += 1
        result_path = results_root / case_id / "result.json"
        result, result_error = _read_result(result_path)
        if result is not None:
            if result.get("schema_version") != VERIFIER_SCHEMA_VERSION:
                raise ValueError(f"Verifier schema mismatch for {case_id}; rerun under v{VERIFIER_SCHEMA_VERSION}")
            if result.get("program_sha") != target["program_sha"]:
                raise ValueError(f"Candidate program SHA mismatch for {case_id}")
        outcome, predicted, result_error = _prediction(result, result_error)
        predicted_ids = {item[0] for item in predicted}

        if outcome == "uncertain":
            counts["uncertain"] += 1
        elif outcome == "error":
            counts["error"] += 1
        if (is_defect and outcome == "fail") or (not is_defect and outcome == "pass"):
            counts["detection_correct"] += 1
        if is_defect and outcome == "fail":
            counts["defect_detected"] += 1
        if is_defect and outcome == "pass":
            counts["false_accept"] += 1
        if not is_defect and outcome == "fail":
            counts["clean_false_reject"] += 1
        exact_match = expected == predicted and ((is_defect and outcome == "fail") or
                                                 (not is_defect and outcome == "pass"))
        if exact_match:
            counts["exact_match"] += 1

        for defect_id in expected_ids | predicted_ids:
            c = classes.setdefault(defect_id, Counter())
            c["tp" if defect_id in expected_ids & predicted_ids else
              "fp" if defect_id in predicted_ids else "fn"] += 1
        expected_joints = {(d, j) for d, j, _ in expected if j is not None}
        predicted_joints = {(d, j) for d, j, _ in predicted if j is not None}
        counts["joint_targets"] += len(expected_joints)
        counts["joint_correct"] += len(expected_joints & predicted_joints)

        usage, has_usage = _usage(result)
        token_totals.update(usage)
        token_cases += has_usage
        case_latency, has_latency = _latency(result)
        latency_ms += case_latency
        latency_cases += has_latency
        trajectory = result.get("trajectory", []) if result else []
        case_observations = len(trajectory) if isinstance(trajectory, list) else 0
        case_spent = result.get("spent", 0) if result else 0
        observations += case_observations if type(case_observations) is int else 0
        budget_spent += case_spent if type(case_spent) is int else 0
        rows.append({"id": case_id, "outcome": outcome,
                     "expected": _json_signatures(expected), "predicted": _json_signatures(predicted),
                     "detection_correct": ((is_defect and outcome == "fail") or
                                           (not is_defect and outcome == "pass")),
                     "exact_match": exact_match, "error": result_error,
                     "observations": case_observations, "budget_spent": case_spent,
                     "tokens": usage, "inference_time_ms": case_latency})

    n, n_defect, n_clean = counts["n"], counts["defect"], counts["clean"]
    per_defect = {}
    for defect_id, c in sorted(classes.items()):
        tp, fp, fn = (c[key] for key in ("tp", "fp", "fn"))
        per_defect[defect_id] = {"tp": tp, "fp": fp, "fn": fn,
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn)}
    report = {
        "version": 1, "status": "complete", "verifier_schema_version": VERIFIER_SCHEMA_VERSION,
        "n": n, "clean_cases": n_clean, "defect_cases": n_defect,
        "detection_accuracy": _ratio(counts["detection_correct"], n),
        "defect_recall": _ratio(counts["defect_detected"], n_defect),
        "false_accept_rate": _ratio(counts["false_accept"], n_defect),
        "clean_false_reject_rate": _ratio(counts["clean_false_reject"], n_clean),
        "exact_match_accuracy": _ratio(counts["exact_match"], n),
        "joint_localization_accuracy": _ratio(counts["joint_correct"], counts["joint_targets"]),
        "uncertain_rate": _ratio(counts["uncertain"], n), "error_rate": _ratio(counts["error"], n),
        "average_observations": observations / n, "average_budget_spent": budget_spent / n,
        "tokens": {"input": token_totals["input"], "output": token_totals["output"],
                   "total": token_totals["total"], "average_total": token_totals["total"] / n,
                   "reported_case_rate": token_cases / n},
        "inference_time_ms": {"total": latency_ms, "average": latency_ms / n,
                              "reported_case_rate": latency_cases / n},
        "per_defect": per_defect, "per_case": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, report)
    return report


def _read_gold(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("id")
            if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", case_id):
                raise ValueError(f"Invalid case id on gold line {line_number}")
            if case_id in rows:
                raise ValueError(f"Duplicate gold case id: {case_id}")
            if not isinstance(row.get("program_sha"), str) or not row["program_sha"]:
                raise ValueError(f"Missing candidate program SHA for {case_id}")
            target = row.get("target")
            if not isinstance(target, dict) or target.get("status") not in {"pass", "fail"}:
                raise ValueError(f"Invalid target for {case_id}")
            findings = target.get("findings")
            if not isinstance(findings, list) or (target["status"] == "pass") != (not findings):
                raise ValueError(f"Target status/findings disagree for {case_id}")
            _signatures(findings)
            rows[case_id] = {"program_sha": row["program_sha"], "findings": findings}
    if not rows:
        raise ValueError("Gold file contains no cases")
    return rows


def _read_result(path: Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, "missing_result"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "invalid_result_json"
    if not isinstance(result, dict):
        return None, "invalid_result_json"
    return result, None


def _prediction(result: dict | None, error: str | None):
    if result is None or result.get("status") != "complete":
        return "error", set(), error or "incomplete_result"
    verdict = result.get("verdict")
    if not isinstance(verdict, dict) or verdict.get("status") not in {"pass", "fail", "uncertain"}:
        return "error", set(), "invalid_verdict"
    findings = verdict.get("findings")
    if not isinstance(findings, list):
        return "error", set(), "invalid_findings"
    try:
        predicted = _signatures(findings)
    except ValueError:
        return "error", set(), "invalid_findings"
    status = verdict["status"]
    if (status == "pass" and predicted) or (status == "fail" and not predicted):
        return "error", set(), "inconsistent_verdict"
    return status, predicted, None


def _signatures(findings: list) -> set[tuple[str, str | None, frozenset[str]]]:
    result = set()
    for finding in findings:
        if not isinstance(finding, dict) or not isinstance(finding.get("defect_id"), str):
            raise ValueError("Finding needs a defect_id")
        parts, joint_id = finding.get("parts"), finding.get("joint_id")
        if not isinstance(parts, list) or any(not isinstance(part, str) for part in parts):
            raise ValueError("Finding parts must be strings")
        if finding["defect_id"] == "kinematics.joint_axis":
            if not isinstance(joint_id, str) or not joint_id:
                raise ValueError("Joint-axis finding needs joint_id")
        elif joint_id is not None:
            raise ValueError("Non-joint finding cannot name a joint")
        signature = (finding["defect_id"], joint_id, frozenset(parts))
        if signature in result:
            raise ValueError("Duplicate finding")
        result.add(signature)
    return result


def _usage(result: dict | None) -> tuple[Counter, int]:
    total = Counter()
    reported = False
    calls = result.get("model_calls", []) if result else []
    for call in calls if isinstance(calls, list) else []:
        usage = call.get("usage") if isinstance(call, dict) else None
        if not isinstance(usage, dict):
            continue
        incoming_value = _number(usage, "input_tokens", "prompt_tokens", "promptTokenCount")
        outgoing_value = _number(usage, "output_tokens", "completion_tokens", "candidatesTokenCount")
        combined = _number(usage, "total_tokens", "totalTokenCount")
        if incoming_value is None and outgoing_value is None and combined is None:
            continue
        incoming, outgoing = incoming_value or 0, outgoing_value or 0
        total.update(input=incoming, output=outgoing, total=combined if combined is not None else incoming + outgoing)
        reported = True
    return total, int(reported)


def _latency(result: dict | None) -> tuple[float, int]:
    calls = result.get("model_calls", []) if result else []
    values = [call.get("elapsed_ms") for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []
    values = [float(value) for value in values if type(value) in (int, float) and math.isfinite(value) and value >= 0]
    return sum(values), int(bool(values))


def _number(mapping: dict, *keys: str):
    for key in keys:
        value = mapping.get(key)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            return value
    return None


def _json_signatures(signatures):
    return [{"defect_id": defect, "joint_id": joint, "parts": sorted(parts)}
            for defect, joint, parts in sorted(signatures, key=lambda item: (item[0], item[1] or "", sorted(item[2])))]


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None
