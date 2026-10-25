"""Controlled terminal-verdict views over an audited Episode export."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from turnitover.oracle.supervision import load_examples
from turnitover.verifier.contracts import Context, parse_decision


SUPERVISION_MODES = ("ordinary", "evidence")


def _finding(label, evidence_steps):
    defect_id = label["defect_id"]
    if defect_id == "kinematics.joint_axis":
        description = "The part moves along a different axis from the reference articulation."
        fix = "Correct the joint axis to match the reference motion."
        joint_id = label.get("params", {}).get("joint")
    elif defect_id == "structure.part_offset":
        description = "The part is displaced relative to the reference configuration."
        fix = "Correct the part placement relative to its parent."
        joint_id = None
    else:
        raise ValueError(f"Unsupported ordinary diagnosis label: {defect_id}")
    return {"defect_id": defect_id, "parts": list(label["parts"]), "joint_id": joint_id,
            "severity": label["severity"], "confidence": 1.0,
            "evidence_steps": evidence_steps, "description": description, "suggested_fix": fix}


def _annotations(root: Path, split: str) -> dict[str, dict]:
    rows = {}
    for line in (root / "annotations.private.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] != split:
            continue
        if row["id"] in rows:
            raise ValueError(f"Duplicate private annotation: {row['id']}")
        rows[row["id"]] = row
    return rows


def _label_key(label):
    return (label["defect_id"], tuple(label["parts"]),
            label.get("params", {}).get("joint") if label["defect_id"] == "kinematics.joint_axis" else None)


def _check_annotation_identity(record, annotation):
    identity = record["data_identity"]
    annotation_identity = annotation.get("data_identity", {})
    required = ("source_dataset", "source_asset_id", "source_identity_sha256",
                "source_content_sha256", "formal_training")
    if any(key not in annotation_identity for key in required):
        raise ValueError("Annotation is missing source identity")
    if (annotation["id"] != record["episode_id"]
            or annotation["reference_sha"] != identity["reference_sha"]
            or annotation["candidate_sha"] != identity["candidate_sha"]
            or any(identity.get(key) != value for key, value in annotation["data_identity"].items())):
        raise ValueError("Episode and private annotation provenance differ")


def _ordinary_target(record: dict, annotation: dict) -> dict:
    _check_annotation_identity(record, annotation)
    payload = json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])
    observed_steps = [row["step"] for row in payload["history"]]
    labels = annotation["labels"]
    if labels:
        if not observed_steps:
            raise ValueError("Ordinary diagnosis requires at least one observation")
        verdict = {
            "status": "fail", "confidence": 1.0,
            "findings": [_finding(label, observed_steps) for label in labels],
            "summary": "The final diagnosis identifies the annotated defects.",
            "limitations": [
                "This ordinary diagnosis target is not filtered by the current visual evidence."
            ],
        }
    else:
        verdict = {
            "status": "uncertain", "confidence": 0.0, "findings": [],
            "summary": "The observations do not establish that the candidate is clean.",
            "limitations": ["The current observations are insufficient to support pass."],
        }
    target = {"verdict": verdict}
    context = Context(**payload["context"])
    parse_decision(target, context, set(observed_steps), final_only=True)
    return target


def load_terminal_training_examples(root, split, supervision="evidence"):
    """Load equal-input terminal rows for ordinary or evidence-qualified SFT."""
    if supervision not in SUPERVISION_MODES:
        raise ValueError(f"Unknown SFT supervision mode: {supervision}")
    annotations = _annotations(Path(root), split)
    seen = set()
    for record, images in load_examples(root, split, "episodes",
                                        require_formal_training=True, terminal_only=True):
        annotation = annotations.get(record["episode_id"])
        if annotation is None:
            raise ValueError(f"Missing private annotation: {record['episode_id']}")
        _check_annotation_identity(record, annotation)
        if supervision == "ordinary":
            record = copy.deepcopy(record)
            record["messages"][1]["content"] = json.dumps(_ordinary_target(record, annotation))
            record["condition"] = "ordinary_final_diagnosis"
        seen.add(record["episode_id"])
        validate_target(record, "diagnosis", supervision, annotation)
        yield record, images
    if annotations is not None and seen != set(annotations):
        raise ValueError("Private annotations and terminal Episode rows differ")


TASKS = ("diagnosis", "stopping")


def load_training_examples(root, split, task="diagnosis", supervision="evidence"):
    """Select one task; ordinary differs only in terminal diagnosis targets."""
    if task not in TASKS or supervision not in SUPERVISION_MODES:
        raise ValueError("Unknown task or supervision")
    examples = (load_terminal_training_examples(root, split, supervision) if task == "diagnosis"
                else load_examples(root, split, "stopping", require_formal_training=True))
    annotations = _annotations(Path(root), split) if task == "stopping" else None
    for record, images in examples:
        record = copy.deepcopy(record)
        if task == "stopping":
            annotation = annotations.get(record["episode_id"])
            if annotation is None:
                raise ValueError("Stopping row lacks audited annotation")
            _check_annotation_identity(record, annotation)
            validate_target(record, task, supervision, annotation)
        record["sft_objective"] = task
        yield record, images


def validate_target(record, task, supervision="evidence", annotation=None):
    """Reject unaudited fail/pass targets; never infer a pass from hidden clean labels."""
    from turnitover.oracle.supervision_validation import public_payload
    payload = public_payload(record)
    target = json.loads(record["messages"][1]["content"])
    covered = record.get("evidence", {}).get("covered_label_indices", [])
    if annotation is None:
        raise ValueError("Supervision needs audited annotations for validation")
    labels = annotation["labels"]
    if (not isinstance(covered, list)
            or any(type(index) is not int or not 0 <= index < len(labels) for index in covered)
            or len(set(covered)) != len(covered)):
        raise ValueError("Invalid covered label indices")
    if task == "diagnosis":
        verdict = parse_decision(target, Context(**payload["context"]),
                                 {row["step"] for row in payload["history"]}, final_only=True)
        if verdict.status == "pass":
            raise ValueError("These exports do not establish positive pass evidence")
        if supervision == "evidence":
            expected = {_label_key(labels[index]) for index in covered}
            actual = {(f.defect_id, tuple(f.parts), f.joint_id) for f in verdict.findings}
            if actual != expected or len(verdict.findings) != len(covered):
                raise ValueError("Evidence findings differ from audited visible labels")
    else:
        from turnitover.oracle.sft_contracts import parse_stopping
        decision = parse_stopping(target)
        if decision["outcome"] == "pass" or (decision["outcome"] == "fail" and not covered):
            raise ValueError("Stop outcome lacks audited visible evidence")


def supervision_digest(records) -> str:
    """Hash public input and assistant target text without serializing image bytes."""
    digest = hashlib.sha256()
    for record in records:
        payload = {
            "id": record["id"],
            "objective": record["sft_objective"],
            "images": record["images"],
            "image_sha256": record["image_sha256"],
            "messages": record["messages"],
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()
