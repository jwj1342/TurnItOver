"""Independent evidence qualification for visible-only teacher verdicts.

The audit separates four questions: were actions legal, did each action yield new
defect-relevant evidence, was the acquired evidence sufficient, and was the verdict
correct. Live Chromium observations are bound to a frozen witness manifest by the
actual rendered view and joint values plus exact pixels; detent names are never trusted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from turnitover.core.actions import Detent, detent_value
from turnitover.oracle.pass_evidence import assess_single_reference_pass
from turnitover.teacher.witnesses import (clamp, protocol_record, rest_state, row_sha256, same_state,
                                          supersedes_errors)
from turnitover.verifier.contracts import Context, parse_decision


IDENTITY_KEYS = (
    "formal_training", "catalog_role", "dataset_revision", "source_dataset", "source_asset_id",
    "source_identity_sha256", "source_content_sha256", "spec_sha256", "dataset",
    "source_identity", "annotation_id", "model_id", "archive_sha256",
)
TRAINING_SPLITS = ("train", "validation")
PRIVATE_PARAM_KEYS = ("angle_deg", "original_axis", "magnitude_m")


def _label_key(row: dict) -> tuple:
    joint = row.get("joint_id")
    if joint is None:
        joint = row.get("params", {}).get("joint")
    return row["defect_id"], tuple(sorted(row.get("parts", []))), joint


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_witnesses(episode: dict, manifest: dict, output_root: Path) -> list[dict]:
    """Replay saved observations as actual (view, joint values) states and bind them to frozen pixels."""
    context = Context(**episode["context"])
    limits = {joint: tuple(bounds) for joint, bounds in manifest["candidate_joint_limits"].items()}
    view, state = context.views[0], rest_state(limits)
    rest = dict(state)
    references = (protocol_record(manifest, episode.get("reference_protocol")) or {}).get("states", [])
    frozen_hierarchy = (manifest.get("rest_hierarchy") or {}).get("candidate")
    rows = []
    for record in episode.get("trajectory", []):
        decision = record["decision"]
        row = {"step": record["step"], "decision_source": record.get("decision_source", "teacher")}
        try:
            parse_decision({"action": decision}, context, set())
            row["legal"] = True
        except Exception as exc:  # saved actions are re-validated independently of the runner
            rows.append({**row, "legal": False, "bound": False, "reason": f"illegal:{exc}"})
            continue
        if decision["type"] == "query_runtime":
            runtime = {**row, "bound": False, "reason": "runtime_only", "image": False,
                       "property": decision["property"]}
            if decision["property"] == "hierarchy" and frozen_hierarchy is not None and same_state(state, rest):
                runtime["payload_matches_frozen_rest_hierarchy"] = record.get("payload") == frozen_hierarchy
            rows.append(runtime)
            continue
        if decision["type"] == "request_view":
            view = decision["view_id"]
        else:
            joint = decision["joint_id"]
            requested = clamp(detent_value(Detent(decision["detent"]), limits[joint]), limits[joint])
            actual = (record.get("payload") or {}).get("value")
            row["requested_value"], row["actual_value"] = requested, actual
            row["actuation_consistent"] = type(actual) in (int, float) and abs(actual - requested) <= 1e-6
            state = {**state, joint: actual if row["actuation_consistent"] else requested}
        row["state"] = {"view": view, "joints": dict(state)}
        matches = [w for w in manifest["witnesses"] if w["view"] == view and same_state(w["state"], state)]
        live = output_root / record["image_ref"] if record.get("image_ref") else None
        live_sha = _sha(live) if live and live.is_file() else None
        bound = [w for w in matches if w["sha256"] == live_sha]
        comparable = any(r["label"]["view"] == view and same_state(r["state"], state) for r in references)
        exposures = set.intersection(*(set(w["exposures"]) for w in bound)) if bound else set()
        perceptible = (set.intersection(*(set(w.get("perceptible_exposures", w["exposures"])) for w in bound))
                       if bound else set())
        row.update(live_sha256=live_sha, witness_indices=[w["index"] for w in matches],
                   witness_keys=[w["key"] for w in matches], bound=bool(bound),
                   reference_comparable=comparable, exposures=sorted(exposures),
                   perceptible_exposures=sorted(perceptible))
        if not matches:
            row["reason"] = "state_outside_frozen_witness_grid"
        elif not bound:
            row["reason"] = "pixels_differ_from_frozen_witness"
        elif not row.get("actuation_consistent", True):
            row.update(bound=False, exposures=[], perceptible_exposures=[],
                       reason="actual_joint_value_differs_from_request")
        rows.append(row)
    return rows


def identity_errors(episode: dict, private_gold: dict, annotation: dict, manifest: dict) -> list[str]:
    """Gold, annotation, frozen manifest, and Episode must name one Train/Validation row."""
    identity, errors = episode.get("data_identity", {}), []
    expected = {
        "source_id": private_gold.get("source_id"), "asset_id": private_gold.get("asset_id"),
        "split_group": private_gold.get("split_group"), "candidate_sha": private_gold.get("program_sha"),
        "reference_sha": private_gold.get("reference_sha"),
        **private_gold.get("provenance", {}), **annotation.get("data_identity", {}),
    }
    for key in ("source_id", "asset_id", "split_group", "candidate_sha", "reference_sha", *IDENTITY_KEYS):
        value = expected.get(key)
        if value is not None and identity.get(key) != value:
            errors.append(f"identity_mismatch:{key}")
    for key, value in (("id", episode.get("episode_id")), ("asset_id", identity.get("asset_id")),
                       ("split_group", identity.get("split_group")),
                       ("candidate_sha", identity.get("candidate_sha")),
                       ("reference_sha", identity.get("reference_sha"))):
        if annotation.get(key) != value:
            errors.append(f"annotation_identity_mismatch:{key}")
        if manifest.get(key) != value:
            errors.append(f"manifest_identity_mismatch:{key}")
    if private_gold.get("id") != episode.get("episode_id"):
        errors.append("gold_identity_mismatch:id")
    if manifest.get("row_sha256", {}).get("gold") != row_sha256(private_gold):
        errors.append("gold_row_sha_differs_from_frozen_manifest")
    if manifest.get("row_sha256", {}).get("annotation") != row_sha256(annotation):
        errors.append("annotation_row_sha_differs_from_frozen_manifest")
    split = identity.get("split")
    if not (split == annotation.get("split") == manifest.get("split")) or split not in TRAINING_SPLITS:
        errors.append("non_training_or_inconsistent_split")
    if manifest.get("in_frozen_test") is not False:
        errors.append("row_not_verified_absent_from_frozen_test")
    if identity.get("formal_training") is not True:
        errors.append("not_formal_training")
    record = episode.get("frozen_witness_manifest") or {}
    path = Path(record["path"]) if record.get("path") else None
    if path is None or not path.is_file() or _sha(path) != record.get("sha256"):
        errors.append("frozen_manifest_sha_or_content_mismatch")
    elif json.loads(path.read_text()) != manifest:
        # Re-auditing under a later manifest is allowed only if it extends the recorded one field-identically.
        if manifest.get("supersedes") != {"path": str(path), "sha256": record["sha256"]} or supersedes_errors(
                json.loads(path.read_text()), manifest):
            errors.append("frozen_manifest_sha_or_content_mismatch")
    runtime = episode.get("runtime_joint_limits")
    if runtime is not None and {k: list(v) for k, v in runtime.items()} != manifest["candidate_joint_limits"]:
        errors.append("runtime_joint_limits_differ_from_frozen_manifest")
    return errors


def reference_errors(episode: dict, manifest: dict, output_root: Path) -> list[str]:
    references = sorted(output_root.glob("reference-*"))
    hashes = [_sha(path) for path in references]
    protocol = protocol_record(manifest, episode.get("reference_protocol"))
    if protocol is not None:
        expected = [row["sha256"] for row in protocol["states"]]
        if episode.get("reference_states") != [row["label"] for row in protocol["states"]]:
            return ["reference_state_labels_differ_from_frozen_protocol"]
    elif episode.get("reference_protocol") not in (None, "static_front"):
        return ["reference_protocol_not_in_frozen_manifest"]
    else:
        expected = [manifest["static_reference_sha256"]]
    return [] if hashes == expected else ["reference_pixels_differ_from_frozen_protocol"]


def leakage_errors(output_root: Path, episode: dict, private_gold: dict) -> list[str]:
    """Model-visible prompts must not contain Gold text, corruption parameters, or program source."""
    secrets = [finding["description"] for finding in private_gold.get("target", {}).get("findings", [])]
    secrets += list(PRIVATE_PARAM_KEYS)
    secrets += [private_gold[key][:200] for key in ("candidate_program", "reference_program") if key in private_gold]
    secrets += [private_gold.get("program_sha"), private_gold.get("reference_sha")]
    errors = []
    for call in episode.get("model_calls", []):
        prompt = (output_root / call["path"] / "prompt.txt").read_text()
        if any(secret and secret in prompt for secret in secrets):
            errors.append(f"private_information_in_prompt:{call['path']}")
    return errors


def _diagnosis(status, findings_ok, objective, sufficient) -> str:
    if status is None:
        return "no_legal_verdict"
    if not objective:
        return {"fail": "false_fail_on_clean", "pass": "pass_on_clean",
                "uncertain": "uncertain_on_clean"}.get(status, "unknown")
    if status == "fail":
        if not sufficient:
            return "fail_without_sufficient_evidence"
        return "correct_fail" if findings_ok else "incorrect_fail"
    if status == "uncertain":
        return "uncertain_after_sufficient_evidence" if sufficient else "trustworthy_uncertain"
    return "pass_on_defect"


def audit_teacher_verdict(episode: dict, private_gold: dict, annotation: dict | None,
                          output_root, *, manifest: dict | None = None) -> dict:
    """Separate objective defect truth from what the teacher's cited pixels support."""
    output_root = Path(output_root)
    reasons, steps = [], []
    if private_gold.get("quality", {}).get("status") != "accepted":
        reasons.append("private_gold_not_checker_accepted")
    if annotation is None:
        reasons.append("missing_counterfactual_annotation")
    if manifest is None:
        reasons.append("missing_frozen_witness_manifest")
    if annotation is not None and manifest is not None:
        reasons.extend(identity_errors(episode, private_gold, annotation, manifest))
        reasons.extend(reference_errors(episode, manifest, output_root))
        steps = bind_witnesses(episode, manifest, output_root)
    reasons.extend(leakage_errors(output_root, episode, private_gold))

    verdict = episode.get("verdict")
    status = verdict.get("status") if verdict else None
    labels = (annotation or {}).get("labels", [])
    gold_keys = {_label_key(row) for row in private_gold.get("target", {}).get("findings", [])}
    label_keys = {_label_key(label): index for index, label in enumerate(labels)}
    if annotation is not None and gold_keys != set(label_keys):
        reasons.append("annotation_labels_differ_from_checker_gold")
    # Evidence counts only when bound to frozen pixels AND comparable with a visible reference state.
    # With a perceptibility tier in the manifest, only perceptible exposures can be credited; mere
    # Oracle exposure (some pixels differ, e.g. a 1-px rim shift) is reported but never sufficient.
    perceptibility = bool((manifest or {}).get("perceptibility"))
    credit_key = "perceptible_exposures" if perceptibility else "exposures"
    comparable = {row["step"]: row for row in steps if row.get("bound") and row.get("reference_comparable")}
    usable = {row["step"]: set(comparable[row["step"]][credit_key]) if row["step"] in comparable else set()
              for row in steps}
    exposed_only = set().union(*(set(row["exposures"]) for row in comparable.values()))
    covered, prior = set().union(*usable.values()) if usable else set(), set()
    for row in steps:
        exposed = usable.get(row["step"], set())
        row["new_objective_evidence"] = sorted(exposed - prior)
        row["defect_relevant"] = bool(exposed)
        prior |= exposed
    sufficient = bool(labels) and covered == set(range(len(labels)))
    imperceptible_only = sorted(exposed_only - covered)
    # Reporting only: exposure is a pixel proxy, so record how salient the credited evidence was.
    signals = (annotation or {}).get("signals", {}).get("against_reference", [])
    salience = max((signals[index]["changed_fraction"] for row in steps
                    if usable.get(row["step"]) for index in row["witness_indices"]
                    if index < len(signals)), default=None)
    exposed_without_reference = sorted(
        {i for row in steps if row.get("bound") and not row.get("reference_comparable")
         for i in row["exposures"]} - covered)

    rest = rest_state({k: tuple(v) for k, v in (manifest or {}).get("candidate_joint_limits", {}).items()})
    motion_joints = sorted({joint for row in steps if row.get("bound") and row.get("reference_comparable")
                            for joint, value in row["state"]["joints"].items() if abs(value - rest[joint]) > 1e-6})
    supported_findings, localization = [], []
    if verdict is None:
        reasons.append("missing_legal_terminal_verdict")
    elif status == "fail":
        for finding in verdict.get("findings", []):
            matches_joint = [label for label in labels if label["defect_id"] == finding["defect_id"]]
            localization.append({
                "defect_id": finding["defect_id"], "joint_id": finding.get("joint_id"),
                "parts": finding.get("parts"),
                "defect_type_correct": bool(matches_joint),
                "joint_correct": any(_label_key(label)[2] == finding.get("joint_id") for label in matches_joint),
                "parts_correct": any(_label_key(label)[1] == tuple(sorted(finding.get("parts", [])))
                                     for label in matches_joint),
                "parts_include_gold": any(set(_label_key(label)[1]) <= set(finding.get("parts", []))
                                          for label in matches_joint),
                "evidence_steps": finding["evidence_steps"]})
            label_index = label_keys.get(_label_key(finding))
            if label_index is None:
                reasons.append(f"finding_not_objective_truth:{finding['defect_id']}")
                continue
            supported = [step for step in finding["evidence_steps"] if label_index in usable.get(step, set())]
            if supported != finding["evidence_steps"] or not supported:
                reasons.append(f"finding_lacks_cited_counterfactual_evidence:{finding['defect_id']}")
            else:
                supported_findings.append({"defect_id": finding["defect_id"],
                                           "joint_id": finding.get("joint_id"),
                                           "evidence_steps": supported, "label_index": label_index})
        if labels and {f["label_index"] for f in supported_findings} != set(range(len(labels))):
            reasons.append("fail_does_not_cover_all_objective_defects")
    elif status == "pass":
        history = [{**row, "action": row["decision"]} for row in episode["trajectory"]]
        payload = {"context": episode["context"], "history": history,
                   "image_order": [path.name for path in sorted(output_root.glob("reference-*"))]}
        pass_result = assess_single_reference_pass(payload, output_root)
        if labels:
            reasons.append("pass_contradicts_objective_defect")
        if not pass_result["eligible"]:
            reasons.append(f"pass_scope_not_supported:{pass_result['reason']}")
    elif status == "uncertain":
        if verdict.get("findings"):
            reasons.append("uncertain_must_not_emit_findings")
        if covered:
            reasons.append("uncertain_after_sufficient_objective_evidence")
    else:
        reasons.append("unknown_verdict_status")

    findings_ok = status == "fail" and not any(r.startswith(("finding_", "fail_")) for r in reasons)
    return {
        "accepted": not reasons, "status": status,
        "objective_defect_ids": sorted(label["defect_id"] for label in labels),
        "supported_findings": supported_findings, "localization": localization,
        "action_audit": steps,
        "actions_legal": all(row.get("legal") for row in steps),
        "evidence_sufficient": sufficient,
        "objective_evidence_covered": sorted(covered),
        "evidence_standard": "perceptible_exposure" if perceptibility else "oracle_exposure",
        "objective_exposure_imperceptible_only": imperceptible_only,
        "max_credited_changed_fraction": salience,
        "objective_exposure_without_visible_reference": exposed_without_reference,
        "diagnosis_outcome": _diagnosis(status, findings_ok, labels, sufficient),
        "comparable_motion_joints": motion_joints,
        "rejection_reasons": reasons,
        "truth_basis": "prepared checker-qualified private gold",
        "evidence_basis": ("live Chromium pixels and actual joint values bound to a frozen witness "
                           "manifest and a visible reference state"),
        "teacher_input_excluded_private_gold": not any(r.startswith("private_information") for r in reasons),
    }
