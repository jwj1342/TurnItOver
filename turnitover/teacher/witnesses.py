"""Frozen witness manifests: candidate/reference pixels and actual joint states fixed before any teacher call.

A manifest binds one prepared Train/Validation row, its checker-qualified Gold, and its
``oracle-first-failure-v4`` annotation to the full motion-view witness grid. Each witness
stores the actual (limit-clamped) joint values the browser renders, so audits compare
states rather than detent names: ``rest`` is value 0 clamped to limits, ``zero`` is the
lower limit, and the two coincide only when the lower limit is 0.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from turnitover.core.actions import Detent, detent_value
from turnitover.oracle.witnesses import Witness
from turnitover.storage.reader import iter_samples


SCHEMA_VERSION = 1
MOTION_VIEWS = ("front", "oblique_fr")
STATE_TOLERANCE = 1e-6
REFERENCE_PROTOCOL = "front_motion_v1"


def sample_framing(raw_root: Path, source_id: str, sample_id: str):
    """Frozen generation framing shared by the Oracle evidence run and the teacher session."""
    for tar_path in sorted(Path(raw_root).glob("*.tar")):
        for sample, _ in iter_samples(tar_path):
            if sample.source_id == source_id and sample.sample_id == sample_id:
                return sample.framing
    raise ValueError("Prepared row has no matching raw sample framing")


def row_sha256(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clamp(value: float, limits) -> float:
    lo, hi = limits
    return min(hi, max(lo, value))


def rest_state(limits: dict) -> dict[str, float]:
    return {joint: clamp(0.0, bounds) for joint, bounds in limits.items()}


def witness_state(row: dict, limits: dict) -> dict[str, float]:
    """Joint values rendered for a grid row: every joint reset, then the witness joint set."""
    state = rest_state(limits)
    if row.get("joint") is not None:
        state[row["joint"]] = clamp(detent_value(Detent(row["detent"]), tuple(limits[row["joint"]])),
                                    limits[row["joint"]])
    return state


def same_state(left: dict, right: dict) -> bool:
    return set(left) == set(right) and all(abs(left[k] - right[k]) <= STATE_TOLERANCE for k in left)


def reference_protocol_rows(grid: list[dict]) -> list[dict]:
    """Label-independent multi-state reference: front rest, then every joint's grid detents at front."""
    return [row for row in grid if row["view"] == "front"]


def reference_label(row: dict) -> dict:
    return {"view": row["view"], "joint_id": row.get("joint"), "detent": row.get("detent") or "rest"}


def fixed_protocol_actions(grid: list[dict]) -> list[dict]:
    """Front rest, then each joint's grid detents with an explicit rest reset between joints."""
    actions = [{"type": "request_view", "view_id": "front"}]
    active = None
    for row in reference_protocol_rows(grid):
        joint = row.get("joint")
        if joint is None:
            continue
        if active not in (None, joint):
            actions.append({"type": "actuate_joint", "joint_id": active, "detent": "rest"})
        actions.append({"type": "actuate_joint", "joint_id": joint, "detent": row["detent"]})
        active = joint
    return actions


def _browser_limits(session, program, framing) -> dict[str, tuple[float, float]]:
    info = session.load(program, framing=framing)
    try:
        return {joint: tuple(row["limits"]) for joint, row in info.joints.items()}
    finally:
        session.dispose()


def _render(session, program, framing, view: str, state: dict) -> bytes:
    session.load(program, framing=framing)
    try:
        session.set_joints(state)
        return session.request_view(view).png
    finally:
        session.dispose()


def build_manifest(session, *, uid: str, split: str, public: dict, gold: dict, annotation: dict,
                   candidate, reference, framing, annotation_root: Path, out_dir: Path,
                   source_files: dict[str, Path], test_ids: set[str],
                   prepared_reference: Path) -> dict:
    """Render and freeze every motion-view witness; refuse rows whose identities disagree."""
    errors = []
    if candidate.sha != gold["program_sha"] or candidate.sha != annotation["candidate_sha"]:
        errors.append("candidate_sha_mismatch")
    if reference.sha != gold["reference_sha"] or reference.sha != annotation["reference_sha"]:
        errors.append("reference_sha_mismatch")
    for key in ("asset_id", "split_group"):
        if gold.get(key) != annotation.get(key):
            errors.append(f"gold_annotation_mismatch:{key}")
    if not (uid == public["id"] == gold["id"] == annotation["id"]):
        errors.append("row_id_mismatch")
    if split != annotation["split"] or split not in ("train", "validation"):
        errors.append("split_mismatch_or_forbidden")
    if uid in test_ids:
        errors.append("row_present_in_frozen_test")
    if gold.get("quality", {}).get("status") != "accepted":
        errors.append("gold_not_checker_accepted")
    if errors:
        raise ValueError(f"Refusing to freeze {uid}: {errors}")

    candidate_limits = _browser_limits(session, candidate, framing)
    reference_limits = _browser_limits(session, reference, framing)
    if set(candidate_limits) != set(reference_limits):
        raise ValueError(f"Candidate and reference joints differ: {uid}")
    saved = {}
    for trajectory in annotation["trajectories"].values():
        for index, observation in zip(trajectory["witness_indices"], trajectory["history"], strict=True):
            saved.setdefault(index, set()).add(file_sha256(annotation_root / observation["image_ref"]))

    out_dir.mkdir(parents=True, exist_ok=False)
    witnesses = []
    for index, row in enumerate(annotation["grid"]):
        if row["view"] not in MOTION_VIEWS:
            continue
        state = witness_state(row, candidate_limits)
        png = _render(session, candidate, framing, row["view"], state)
        digest = hashlib.sha256(png).hexdigest()
        if index in saved and saved[index] != {digest}:
            raise ValueError(f"Fresh witness render differs from frozen v4 pixels: {uid}/{index}")
        witnesses.append({"index": index, "key": Witness(row["view"], row.get("joint"),
                                                         Detent(row["detent"]) if row.get("detent") else None).key,
                          "view": row["view"], "joint": row.get("joint"), "detent": row.get("detent"),
                          "state": state, "sha256": digest,
                          "v4_pixel_verified": index in saved,
                          "exposures": annotation["exposures"][index]})
    references = []
    for position, row in enumerate(reference_protocol_rows(annotation["grid"])):
        state = witness_state(row, reference_limits)
        png = _render(session, reference, framing, row["view"], state)
        path = out_dir / f"reference-{position:02d}.png"
        path.write_bytes(png)
        references.append({"file": path.name, "label": reference_label(row), "state": state,
                           "sha256": hashlib.sha256(png).hexdigest()})
    frozen_reference = file_sha256(annotation_root / "images" / uid / "reference.png")
    if references[0]["sha256"] != frozen_reference:
        raise ValueError(f"Reference front/rest differs from the frozen v4 reference: {uid}")

    comparable = {w["index"] for w in witnesses
                  if any(w["view"] == r["label"]["view"] and same_state(w["state"], r["state"])
                         for r in references)}
    support = []
    for label_index, label in enumerate(annotation["labels"]):
        exposing = [w["index"] for w in witnesses if label_index in w["exposures"]]
        support.append({"label_index": label_index, "defect_id": label["defect_id"],
                        "exposing_witnesses": exposing,
                        "static_front_exposing": [i for i in exposing
                                                  if annotation["grid"][i]["joint"] is None
                                                  and annotation["grid"][i]["view"] == "front"],
                        "reference_comparable_exposing": sorted(set(exposing) & comparable)})
    manifest = {
        "schema_version": SCHEMA_VERSION, "id": uid, "split": split,
        "asset_id": gold["asset_id"], "split_group": gold["split_group"],
        "candidate_sha": candidate.sha, "reference_sha": reference.sha,
        "framing": {"center": list(framing.center), "radius": framing.radius},
        "row_sha256": {"input": row_sha256(public), "gold": row_sha256(gold),
                       "annotation": row_sha256(annotation)},
        "source_files": {name: {"path": str(path), "sha256": file_sha256(path)}
                         for name, path in source_files.items()},
        "in_frozen_test": False, "gold_quality": gold["quality"]["status"],
        "objective_defect_ids": [label["defect_id"] for label in annotation["labels"]],
        "candidate_joint_limits": {k: list(v) for k, v in candidate_limits.items()},
        "reference_joint_limits": {k: list(v) for k, v in reference_limits.items()},
        "witnesses": witnesses,
        "reference_protocol": {"name": REFERENCE_PROTOCOL, "label_independent": True,
                               "states": references},
        "fixed_protocol_actions": fixed_protocol_actions(annotation["grid"]),
        "reference_support": support,
        "v4_reference_sha256": frozen_reference,
        "static_reference_sha256": file_sha256(prepared_reference),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
