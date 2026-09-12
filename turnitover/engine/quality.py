"""Conservative label qualification from privileged checker results, not judge observations."""
from turnitover.core.sample import CheckResult, CorruptionSpec


def validate_label(label: CorruptionSpec, checks: tuple[CheckResult, ...]) -> dict:
    by_id = {c.checker_id: c for c in checks}
    if any(i.name == "compile" and not i.passed for c in checks for i in c.invariants):
        return {"defect_id": label.defect_id, "status": "invalid", "reason": "candidate_execution_failed"}
    if label.defect_id == "runtime.triangle_budget":
        candidates = [i for i in getattr(by_id.get("runtime.budget"), "invariants", ()) if i.name == "triangles"]
    elif label.defect_id in ("structure.part_offset", "kinematics.joint_axis"):
        def relevant_state(state):
            if label.defect_id == "structure.part_offset":
                return state == "rest"
            return state is not None and state.startswith(f"joint:{label.params['joint']}:")
        candidates = [i for i in getattr(by_id.get("geometry.alignment"), "invariants", ())
                      if relevant_state(i.state) and set(i.parts) & set(label.parts) and i.name in ("bbox_iou", "chamfer_m")]
    else:
        candidates = []
    if not candidates:
        status, reason = "unsupported", "required_checker_or_state_missing"
    elif any(not i.passed for i in candidates):
        status, reason = "valid", "target_constraint_violated"
    else:
        status, reason = "invalid", "perturbation_does_not_violate_target_constraint"
    return {"defect_id": label.defect_id, "status": status, "reason": reason}


def qualify(labels, checks, isolated_checks) -> dict:
    validations = []
    for label, isolated in zip(labels, isolated_checks, strict=True):
        individual = validate_label(label, isolated)
        final = validate_label(label, checks)
        validations.append({**final, "isolated_status": individual["status"], "isolated_reason": individual["reason"]})
    if labels:
        accepted = all(v["status"] == v["isolated_status"] == "valid" for v in validations)
    else:
        accepted = bool(checks) and all(c.passed and c.invariants for c in checks)
    return {"status": "accepted" if accepted else "quarantined", "labels": validations,
            "scope": "configured_constraints_only", "observation_grounded": False}
