"""Identity and public-input checks for fixed-observation supervision exports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def _reject_private_fields(value):
    """Reject known teacher-only metadata, including inside observation payloads."""
    private = {"labels", "corruptions", "exposures", "witness_indices", "data_identity",
               "future_history", "private_target", "teacher_metadata", "ground_truth",
               "covered_label_indices", "uncovered_label_indices", "gold"}
    if isinstance(value, dict):
        if private.intersection(value):
            raise ValueError("Private teacher metadata in public input")
        for child in value.values():
            _reject_private_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_private_fields(child)


def validate_identity(identity, manifest):
    if not isinstance(identity, dict):
        raise ValueError("Episode is missing data identity")
    if identity.get("formal_training") != manifest.get("formal_training"):
        raise ValueError("Episode data purpose differs from manifest")
    for key, length in (("source_identity_sha256", 64), ("source_content_sha256", 64),
                        ("reference_sha", 16), ("candidate_sha", 16)):
        if not isinstance(identity.get(key), str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, identity[key]):
            raise ValueError(f"Episode has invalid {key}")
    for key in ("source_dataset", "source_asset_id", "asset_id", "split_group"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            raise ValueError(f"Episode has invalid {key}")

    expected = hashlib.sha256(f"{identity['source_dataset']}:{identity['source_asset_id']}".encode()).hexdigest()
    if identity["source_identity_sha256"] != expected:
        raise ValueError("Source identity SHA mismatch")


def audit_split_identities(root: Path, kind: str, manifest: dict):
    """Read only export metadata; no held-out targets are returned to callers."""
    assignments, provenance, episodes = {}, {}, {}
    for split, export in ((split, export) for split in ("train", "validation", "test")
                          for export in ("episodes", "stopping")):
        path = root / f"{split}.{export}.jsonl"
        if not path.exists():
            continue
        seen = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            identity = record.get("data_identity")
            validate_identity(identity, manifest)
            if record["id"] in seen:
                raise ValueError("Duplicate supervision record")
            seen.add(record["id"])
            keys = [("group", identity["split_group"]),
                    ("source", identity["source_identity_sha256"]),
                    ("content", identity["source_content_sha256"]),
                    ("asset", identity["source_dataset"], identity["asset_id"]),
                    ("reference", identity["reference_sha"])]
            for key in keys:
                if key in assignments and assignments[key] != split:
                    raise ValueError("Original object leakage across splits")
                assignments[key] = split
            source = identity["source_identity_sha256"]
            value = (identity["source_content_sha256"], identity["source_dataset"],
                     identity["asset_id"], identity["split_group"], identity["reference_sha"])
            if source in provenance and provenance[source] != value:
                raise ValueError("Inconsistent source identity")
            provenance[source] = value
            episode = record.get("episode_id", record["id"])
            value = (split, identity)
            if episode in episodes and episodes[episode] != value:
                raise ValueError("Inconsistent episode identity")
            episodes[episode] = value
            groups = manifest.get("groups", {})
            if identity["split_group"] in groups and groups[identity["split_group"]] != split:
                raise ValueError("Split differs from manifest")


def public_payload(record):
    messages = record.get("messages")
    if (not isinstance(messages, list) or len(messages) != 2
            or [row.get("role") for row in messages] != ["user", "assistant"]):
        raise ValueError("Expected one user/assistant pair")
    payload = json.loads(messages[0]["content"].rsplit("\n", 1)[-1])
    _reject_private_fields(payload)
    allowed = {"context", "history", "image_order", "remaining", "taxonomy_ids",
               "training_objective", "target_joint_id", "joint_id"}
    if set(payload) - allowed:
        raise ValueError("Unknown public input fields")
    history = payload["history"]
    steps = [row["step"] for row in history]
    if any(type(step) is not int or step < 0 for step in steps) or steps != sorted(set(steps)):
        raise ValueError("History steps must be unique and causal")
    if payload.get("image_order") != record["images"]:
        raise ValueError("Public image order differs from image roster")
    if record["images"][1:] != [row["image_ref"] for row in history]:
        raise ValueError("History observations differ from image roster")
    return payload
