"""Export audited interactive-teacher calls into the existing Episode JSONL contract.

Action supervision and diagnosis supervision are qualified separately: an action row needs a
legal, frozen-witness-bound, reference-comparable observation of a new state; a
diagnosis row needs the whole Episode to pass the independent evidence audit. Both need
pipeline integrity, consistent identity, no prompt leakage, and a Train/Validation split.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from turnitover.core.jsonio import write_json_atomic
from turnitover.teacher.evidence import TRAINING_SPLITS


BLOCKING_PREFIXES = ("identity_mismatch", "annotation_identity_mismatch", "manifest_identity_mismatch",
                     "gold_identity_mismatch", "gold_row_sha", "annotation_row_sha", "non_training",
                     "row_not_verified", "not_formal_training", "runtime_joint_limits", "frozen_manifest",
                     "reference_", "private_information", "private_gold_not", "missing_counterfactual",
                     "missing_frozen_witness",
                     "annotation_labels_differ")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_verified(src: Path, dst: Path, expected_sha: str) -> None:
    """Copy one image atomically and verify both ends, including existing destinations."""
    if not src.is_file():
        raise ValueError(f"Missing source image: {src}")
    if _sha(src) != expected_sha:
        raise ValueError(f"Source image changed while exporting: {src}")
    if dst.is_file() and _sha(dst) == expected_sha:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dst.parent, prefix=f".{dst.name}.", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(src, temporary)
        if _sha(src) != expected_sha or _sha(temporary) != expected_sha:
            raise ValueError(f"Image SHA-256 mismatch while exporting: {src}")
        temporary.replace(dst)
    finally:
        temporary.unlink(missing_ok=True)
    if _sha(dst) != expected_sha:
        raise ValueError(f"Exported image SHA-256 mismatch: {dst}")


def _copy_images(source: Path, target: Path, episode_id: str, call: dict) -> tuple[list[str], list[str], list[dict]]:
    call_dir = source / call["path"]
    relative_paths = json.loads((call_dir / "images.json").read_text())
    exported, hashes, provenance = [], [], []
    for index, relative in enumerate(relative_paths):
        src = (source / relative).resolve()
        try:
            source_relative = src.relative_to(source.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(f"Image path escapes Episode directory: {relative}") from exc
        source_sha = _sha(src) if src.is_file() else None
        if source_sha is None:
            raise ValueError(f"Missing source image: {src}")
        suffix = src.suffix or ".png"
        filename = f"{call_dir.name}-{index:02d}-{source_sha}{suffix.lower()}"
        dst = target / "images" / episode_id / filename
        _copy_verified(src, dst, source_sha)
        exported_path = dst.relative_to(target).as_posix()
        exported.append(exported_path)
        hashes.append(source_sha)
        provenance.append({"source": source_relative, "source_sha256": source_sha,
                           "exported": exported_path, "exported_sha256": _sha(dst)})
    return exported, hashes, provenance


def _episode_blockers(episode: dict, audit: dict, episode_path: Path) -> list[str]:
    evidence = audit.get("independent_evidence_audit", {})
    reasons = [r for r in evidence.get("rejection_reasons", []) if r.startswith(BLOCKING_PREFIXES)]
    if not evidence:
        reasons.append("missing_independent_evidence_audit")
    if not audit.get("pipeline_ok"):
        reasons.append("pipeline_integrity_audit_failed")
    if audit.get("episode_sha256") != _sha(episode_path):
        reasons.append("episode_changed_after_audit")
    identity = episode.get("data_identity", {})
    if identity.get("split") not in TRAINING_SPLITS or identity.get("formal_training") is not True:
        reasons.append("test_or_nonformal_split_forbidden")
    return reasons


def qualified_action(step: dict, seen_states: list) -> bool:
    """Label-independent: a legal teacher action reaching a new, frozen-bound, reference-comparable state,
    or a first rest-pose hierarchy query whose payload matches the frozen one.

    Hidden labels never decide which visible prefixes are supervised; defect relevance is metadata only.
    """
    if step.get("decision_source") != "teacher" or not step.get("legal"):
        return False
    if step.get("reason") == "runtime_only":
        key = {"runtime": step.get("property")}
        if not step.get("payload_matches_frozen_rest_hierarchy") or key in seen_states:
            return False
        seen_states.append(key)
        return True
    if not (step.get("bound") and step.get("reference_comparable")) or step["state"] in seen_states:
        return False
    seen_states.append(step["state"])
    return True


def export_episodes(sources: list[Path], output: Path, audit_name: str = "audit.json") -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Interactive SFT export directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    streams = {(split, kind): (output / f"{split}.{kind}.jsonl").open("w", encoding="utf-8")
               for split in TRAINING_SPLITS
               for kind in ("actions", "diagnosis", "episodes")}
    rejected = (output / "rejected.jsonl").open("w", encoding="utf-8")
    image_manifest = (output / "image-manifest.jsonl").open("w", encoding="utf-8")
    counts = {"episodes_seen": 0, "episodes_blocked": 0, "diagnosis_accepted": 0,
              "diagnosis_rejected": 0, "action_rows": 0, "action_rows_rejected": 0, "terminal_rows": 0}
    provenance, seen_ids = [], set()
    try:
        for source in sources:
            counts["episodes_seen"] += 1
            episode_path, audit_path = source / "episode.json", source / audit_name
            episode, audit = json.loads(episode_path.read_text()), json.loads(audit_path.read_text())
            evidence = audit.get("independent_evidence_audit", {})
            blockers = _episode_blockers(episode, audit, episode_path)
            if blockers:
                counts["episodes_blocked"] += 1
                rejected.write(json.dumps({"episode_id": episode.get("episode_id"), "source": str(source),
                                           "scope": "episode", "reasons": blockers}) + "\n")
                continue
            identity = episode["data_identity"]
            split = identity["split"]
            diagnosis_ok = bool(evidence.get("accepted")) and not episode.get("teacher_verdict_missing")
            if not diagnosis_ok:
                counts["diagnosis_rejected"] += 1
                rejected.write(json.dumps({"episode_id": episode["episode_id"], "source": str(source),
                                           "scope": "diagnosis",
                                           "outcome": evidence.get("diagnosis_outcome"),
                                           "reasons": evidence.get("rejection_reasons")
                                           or ["missing_legal_terminal_verdict"]}) + "\n")
            teacher_steps = iter(row for row in evidence.get("action_audit", [])
                                 if row.get("decision_source") == "teacher")
            seen_states = []
            for decision_index, decision in enumerate(row for row in episode["decisions"] if row["valid"]):
                raw = decision["decision"]
                terminal = "verdict" in raw
                if terminal and not diagnosis_ok:
                    continue
                step = {}
                if not terminal:
                    step = next(teacher_steps, {})
                    if not qualified_action(step, seen_states):
                        counts["action_rows_rejected"] += 1
                        continue
                call = episode["model_calls"][decision["call"]]
                images, image_hashes, image_provenance = _copy_images(
                    source, output, episode["episode_id"], call)
                run_id = hashlib.sha256(json.dumps({
                    "source_name": source.name,
                    "episode_sha256": _sha(episode_path),
                    "audit_sha256": _sha(audit_path),
                    "image_sha256": image_hashes,
                }, sort_keys=True).encode()).hexdigest()[:16]
                row_id = (f"{episode['episode_id']}:{identity.get('experiment_mode', 'active')}:"
                          f"{'verdict' if terminal else f'action:{decision_index}'}")
                if row_id in seen_ids:  # repeated conditions on one row: qualify by the source run directory
                    row_id = f"{row_id}@{source.name}-{run_id}"
                if row_id in seen_ids:
                    raise ValueError(f"Duplicate export row id: {row_id}")
                seen_ids.add(row_id)
                row = {
                    "id": row_id,
                    "episode_id": episode["episode_id"], "decision_index": decision_index,
                    "source_episode": str(source), "source_run_id": run_id,
                    "condition": "visible_only_interactive_teacher",
                    "experiment_mode": identity.get("experiment_mode"),
                    "reference_protocol": identity.get("reference_protocol"),
                    "common_input": identity.get("common_input", "none"),
                    "prompt_addenda": identity.get("prompt_addenda", []),
                    "terminal": terminal, "termination_target": "verdict" if terminal else None,
                    "termination_reason": episode.get("termination") if terminal else None,
                    "images": images, "image_sha256": image_hashes,
                    "image_provenance": image_provenance,
                    "messages": [
                        {"role": "user", "content": (source / call["path"] / "prompt.txt").read_text()},
                        {"role": "assistant", "content": json.dumps(raw)},
                    ],
                    "data_identity": identity,
                    "source_episode_sha256": _sha(episode_path),
                    "source_audit_sha256": _sha(audit_path),
                    "episode_diagnosis_accepted": diagnosis_ok,
                    "action_new_objective_evidence": None if terminal else step.get("new_objective_evidence"),
                    "audit_conclusion": {key: evidence.get(key) for key in (
                        "status", "diagnosis_objective", "diagnosis_outcome", "evidence_sufficient",
                        "objective_evidence_complete", "supported_findings",
                        "objective_evidence_covered", "rejection_reasons")},
                }
                streams[(split, "episodes")].write(json.dumps(row) + "\n")
                streams[(split, "diagnosis" if terminal else "actions")].write(json.dumps(row) + "\n")
                for image in image_provenance:
                    image_manifest.write(json.dumps({"row_id": row_id, "episode_id": episode["episode_id"],
                                                     "source_episode": str(source), **image}) + "\n")
                counts["terminal_rows" if terminal else "action_rows"] += 1
            counts["diagnosis_accepted"] += diagnosis_ok
            provenance.append({"episode_id": episode["episode_id"], "source": str(source),
                               "experiment_mode": identity.get("experiment_mode"),
                               "source_asset_id": identity.get("source_asset_id"),
                               "split": split, "split_group": identity.get("split_group"),
                               "candidate_sha": identity.get("candidate_sha"),
                               "reference_sha": identity.get("reference_sha"),
                               "diagnosis_objective": evidence.get("diagnosis_objective"),
                               "diagnosis_accepted": diagnosis_ok,
                               "episode_sha256": _sha(episode_path), "audit_sha256": _sha(audit_path)})
    finally:
        for stream in streams.values():
            stream.close()
        rejected.close()
        image_manifest.close()
    manifest = {"schema_version": 3, "status": "complete", "formal_training": True,
                "kind": "visible_only_interactive_teacher", "counts": counts,
                "splits": list(TRAINING_SPLITS), "audit_name": audit_name, "sources": provenance}
    write_json_atomic(output / "manifest.json", manifest)
    return manifest
