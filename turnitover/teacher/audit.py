"""Independent integrity and evidence audit for a saved teacher episode."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from turnitover.core.jsonio import write_json_atomic
from turnitover.teacher.evidence import audit_teacher_verdict
from turnitover.verifier.contracts import Context, parse_decision


VISIBLE_KEYS = {"context", "remaining", "stop_required", "image_order", "history", "taxonomy_ids"}


def audit_episode(output: Path, private_gold: dict | None = None,
                  private_annotation: dict | None = None, *, manifest: dict | None = None,
                  audit_name: str = "audit.json") -> dict:
    episode = json.loads((output / "episode.json").read_text())
    context = Context(**episode["context"])
    errors, informative, hashes = [], [], set()
    reference_sha256, observation_sha256 = [], []
    for path in sorted(output.glob("reference-*")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        reference_sha256.append(digest)
        hashes.add(digest)
    spent = 0
    for expected_step, record in enumerate(episode["trajectory"]):
        try:
            action = parse_decision({"action": record["decision"]}, context, set())
        except Exception as exc:
            errors.append(f"illegal saved action at step {expected_step}: {exc}")
            continue
        if record["step"] != expected_step:
            errors.append(f"non-causal step index at {expected_step}")
        spent += 1
        if record.get("image_ref"):
            path = output / record["image_ref"]
            if not path.is_file():
                errors.append(f"missing observation image at step {expected_step}")
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                observation_sha256.append(digest)
                informative.append(digest not in hashes)
                hashes.add(digest)
        else:
            informative.append(bool(record.get("payload")))
    for call in episode.get("model_calls", []):
        request = json.loads((output / call["path"] / "request.json").read_text())
        if set(request) != VISIBLE_KEYS:
            errors.append(f"model-visible request has unexpected fields: {sorted(set(request) - VISIBLE_KEYS)}")
        image_order = request["image_order"]
        image_paths = json.loads((output / call["path"] / "images.json").read_text())
        if len(image_paths) != len(image_order):
            errors.append("model image list and declared order differ in length")
        reference_paths = [path.name for path in sorted(output.glob("reference-*"))]
        if image_paths != reference_paths + [item["image_ref"] for item in request["history"]
                                             if item.get("image_ref")]:
            errors.append("model images are not the references followed by causal observations")
        expected_image_steps = [item["step"] for item in request["history"] if item.get("image_ref")]
        if [item.get("step") for item in image_order if item["kind"] == "observation"] != expected_image_steps:
            errors.append("model image order is not the causal history order")
        for index, visible in enumerate(request["history"]):
            if index >= len(episode["trajectory"]):
                errors.append("model-visible history includes a future observation")
                break
            saved = episode["trajectory"][index]
            for key in ("step", "action", "image_ref", "payload"):
                if visible.get(key) != saved.get(key):
                    errors.append(f"model-visible history differs from saved trajectory at step {index}")
                    break
        if request["remaining"] != episode["budget"] - len(request["history"]):
            errors.append("model-visible remaining budget is inconsistent with its causal history")
        if request["stop_required"] is not (request["remaining"] == 0):
            errors.append("model-visible stop instruction is inconsistent with remaining budget")
    if spent != episode["spent"] or spent > episode["budget"]:
        errors.append("saved budget accounting is inconsistent")

    evidence_supported, evidence_reason = None, "No diagnosis finding was emitted."
    verdict = episode.get("verdict")
    if verdict and verdict.get("findings"):
        references_are_informative = all(
            all(0 <= step < len(informative) and informative[step] for step in finding["evidence_steps"])
            for finding in verdict["findings"])
        if not references_are_informative:
            evidence_supported = False
            evidence_reason = "A finding cites a missing or pixel-duplicate observation."
        else:
            evidence_reason = "Pixel novelty alone cannot establish semantic support; checker or human review is required."
    handled = {"budget_exhausted", "teacher_finished", "model_call_limit", "invalid_decision",
               "model_error", "observation_error"}
    invalid_outputs = [item for item in episode["decisions"] if not item["valid"]]
    invalid_actions = [item for item in invalid_outputs
                       if isinstance(item.get("decision"), dict) and "action" in item["decision"]]
    model_calls = episode.get("model_calls", [])
    response_received = any(
        item.get("status") == "received" or "finish_reason" in item for item in model_calls)
    responses_accepted = bool(model_calls) and all(
        item.get("status") == "received" for item in model_calls)
    audit = {"schema_version": 1, "episode_id": episode["episode_id"],
             "pipeline_ok": not errors and episode.get("termination") in handled,
             "teacher_actions_legal": not invalid_actions,
             "teacher_outputs_legal": responses_accepted and not invalid_outputs,
             "teacher_response_received": response_received,
             "teacher_verdict_present": verdict is not None,
             "budget_exhausted": bool(episode.get("budget_exhausted")),
             "teacher_verdict_missing": bool(episode.get("teacher_verdict_missing")),
             "diagnosis_evidence_supported": evidence_supported,
             "diagnosis_evidence_reason": evidence_reason,
             "informative_observations": sum(informative), "observations": len(informative),
             "observation_new_information": informative, "errors": errors,
             "reference_sha256": reference_sha256, "observation_sha256": observation_sha256,
             "scope": "integrity_and_observation_novelty_not_semantic_ground_truth"}
    if "data_identity" in episode:
        audit["data_identity"] = episode["data_identity"]
    if private_gold is not None:
        gold_ids = sorted(item["defect_id"] for item in private_gold.get("target", {}).get("findings", []))
        predicted = sorted(item["defect_id"] for item in (verdict or {}).get("findings", []))
        audit["private_posthoc_comparison"] = {"gold_defect_ids": gold_ids,
                                                "teacher_defect_ids": predicted,
                                                "exact_defect_ids": gold_ids == predicted,
                                                "teacher_input_excluded_private_gold": True}
        audit["independent_evidence_audit"] = audit_teacher_verdict(
            episode, private_gold, private_annotation, output, manifest=manifest)
    audit["episode_sha256"] = hashlib.sha256((output / "episode.json").read_bytes()).hexdigest()
    write_json_atomic(output / audit_name, audit)
    return audit
