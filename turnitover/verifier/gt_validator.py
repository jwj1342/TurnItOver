"""Privileged deterministic validation of candidate code against a clean program."""
from __future__ import annotations

import math

from turnitover.checkers.evidence import collect_evidence, resolve_states
from turnitover.checkers.geometry import GeometryAlignmentChecker
from turnitover.checkers.runtime_budget import RuntimeBudgetChecker
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession


GT_VALIDATION_SCHEMA_VERSION = 1


def validate_code(candidate, reference, render, views, max_triangles, max_draw_calls):
    """Validate after model inference; this result must never enter the verifier context."""
    base = {"schema_version": GT_VALIDATION_SCHEMA_VERSION,
            "candidate_sha": candidate.sha, "reference_sha": reference.sha,
            "privileged": True, "used_by_verifier": False, "used_by_judge": False}
    try:
        with ObservationSession(render, views) as session:
            reference_info = session.load(reference, framing=None)
            reference_contract = _contract(reference_info)
            states = resolve_states(("rest", "joint:*:mid", "joint:*:limit"),
                                    tuple(reference_info.joints))
            gold = collect_evidence(session, reference.sha, states)

            candidate_info = session.load(candidate, framing=reference_info.framing)
            contract_check = _check_contract(_contract(candidate_info), reference_contract)
            evidence_states = states if set(candidate_info.joints) == set(reference_info.joints) else ("rest",)
            observed = collect_evidence(session, candidate.sha, evidence_states)
            checks = [contract_check,
                      to_dict(GeometryAlignmentChecker().check(observed, gold)),
                      to_dict(RuntimeBudgetChecker(max_triangles, max_draw_calls).check(observed, None))]
            return {**base, "status": "complete",
                    "code_correct": all(check["passed"] for check in checks),
                    "checks": checks}
    except Exception as exc:
        return {**base, "status": "error", "code_correct": None,
                "error_type": type(exc).__name__, "error": str(exc), "checks": []}


def _contract(info):
    return {"parts": sorted(info.parts),
            "joints": {joint_id: {"type": value["type"], "part": value["part"],
                                  "limits": [float(x) for x in value["limits"]]}
                       for joint_id, value in sorted(info.joints.items())}}


def _check_contract(candidate: dict, reference: dict) -> dict:
    differences = []
    if candidate["parts"] != reference["parts"]:
        differences.append("parts")
    if set(candidate["joints"]) != set(reference["joints"]):
        differences.append("joint_ids")
    for joint_id in sorted(set(candidate["joints"]) & set(reference["joints"])):
        actual, expected = candidate["joints"][joint_id], reference["joints"][joint_id]
        for field in ("type", "part"):
            if actual[field] != expected[field]:
                differences.append(f"joint:{joint_id}:{field}")
        if any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
               for a, b in zip(actual["limits"], expected["limits"])):
            differences.append(f"joint:{joint_id}:limits")
    return {"checker_id": "program.contract", "passed": not differences,
            "differences": differences, "candidate": candidate, "reference": reference}
