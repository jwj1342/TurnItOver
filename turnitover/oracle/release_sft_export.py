"""Convert TurnItOver release ZIPs into fixed-observation SFT exports."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zipfile

from turnitover.oracle.episodes import episode_examples


HARD_DEFECTS = {
    "axis_direction_3deg": "kinematics.joint_axis",
    "joint_anchor_offset": "kinematics.joint_anchor",
    "moving_part_initial_rotation": "structure.part_rotation",
}
SEVERITY = {"easy": 0.9, "medium": 0.6, "hard": 0.35}


@dataclass(frozen=True)
class ReleaseCase:
    level: str
    archive: Path
    root: str
    gold: dict
    public: dict


def _jsonl(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def _release(path: Path) -> tuple[dict, list[ReleaseCase]]:
    manifest = json.loads((path / "release-manifest.json").read_text(encoding="utf-8"))
    archive = next(path.glob("*-data.zip"))
    with zipfile.ZipFile(archive) as source:
        root = source.namelist()[0].split("/", 1)[0]
        gold = _jsonl(source.read(f"{root}/private/gold.jsonl"))
        public = {row["case_id"]: row for row in _jsonl(source.read(f"{root}/review_manifest.jsonl"))}
    return manifest, [ReleaseCase(root, archive, root, row, public.get(row["case_id"], {})) for row in gold]


def _provenance(case: ReleaseCase) -> tuple[str, str, str, str, str]:
    gold, source = case.gold, case.gold.get("source", {})
    source_identity = source.get("source_identity") or gold.get("source_identity") or ""
    fallback_asset = str(gold["asset_id"]).removeprefix("PM_")
    identity_dataset, identity_asset = (source_identity.split(":", 1)
                                        if ":" in source_identity else ("PartNet-Mobility", fallback_asset))
    dataset = source.get("source_dataset") or source.get("dataset") or identity_dataset
    if dataset in {"PartNet-Mobility", "sapien-sim/PartNetMobility"}:
        dataset = "sapien-sim/PartNetMobility"
    source_asset = str(source.get("source_asset_id") or identity_asset)
    split_group = f"{dataset}:{source_asset}"
    content = (source.get("original_spec_sha256") or gold.get("original_spec_sha256")
               or source.get("original_geometry_fingerprint") or source.get("archive_sha256")
               or gold.get("source_archive_sha256"))
    if not isinstance(content, str) or len(content) != 64:
        content = hashlib.sha256(split_group.encode()).hexdigest()
    return dataset, source_asset, str(gold["asset_id"]), split_group, content


def _split(group: str) -> str:
    value = int(hashlib.sha256(f"turnitover-sft-v1:{group}".encode()).hexdigest()[:8], 16)
    return "validation" if value % 5 == 0 else "train"


def _labels(case: ReleaseCase) -> list[dict]:
    gold = case.gold
    defect_id = gold.get("defect_id") or HARD_DEFECTS.get(gold.get("kind"))
    if not defect_id:
        return []
    parts = list(gold.get("parts") or ([gold["part_id"]] if gold.get("part_id") else ["moving_part"]))
    params = dict(gold.get("parameters") or {})
    if defect_id == "kinematics.joint_axis":
        params["joint"] = gold.get("joint_id") or params.get("joint") or "joint_0"
    return [{"defect_id": defect_id, "parts": parts, "severity": SEVERITY[case.level], "params": params}]


def _members(case: ReleaseCase) -> tuple[str, list[tuple[str, str]]]:
    cid, prefix = case.gold["case_id"], f"{case.root}/"
    if case.level == "easy":
        return case.public.get("view", "front"), [
            (f"frame_{index}", prefix + relative) for index, relative in enumerate(case.public["images"])]
    if case.level == "medium":
        roles = ("reference_rest", "reference_mid", "reference_limit",
                 "candidate_rest", "candidate_mid", "candidate_limit")
        return case.public.get("view", "front"), [
            (role, f"{prefix}native/{cid}/{role}.png") for role in roles]
    views = (case.gold["initial_view"], case.gold["supplementary_view"])
    members = []
    for view in views:
        for kind in ("reference", "candidate"):
            for state in ("rest", "mid", "limit"):
                role = f"{view}_{kind}_{state}"
                members.append((role, f"{prefix}native/{cid}/{view}/{kind}_{state}.png"))
    return views[0], members


def _copy_images(case: ReleaseCase, output: Path) -> tuple[str, list[dict], list[str], list[str]]:
    view, members = _members(case)
    paths, digests = [], []
    with zipfile.ZipFile(case.archive) as source:
        for index, (role, member) in enumerate(members):
            data = source.read(member)
            relative = f"images/{case.gold['case_id']}/{index:02d}-{role}.png"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            paths.append(relative)
            digests.append(hashlib.sha256(data).hexdigest())
    history = [{"step": index, "image_ref": relative,
                "action": {"type": "request_view", "view_id": view}}
               for index, relative in enumerate(paths[1:])]
    return view, history, paths, digests


def _identity(case: ReleaseCase, split_group: str, content: str) -> dict:
    dataset, source_asset, asset, _, _ = _provenance(case)
    return {"source_dataset": dataset, "source_asset_id": source_asset, "asset_id": asset,
            "split_group": split_group,
            "source_identity_sha256": hashlib.sha256(f"{dataset}:{source_asset}".encode()).hexdigest(),
            "source_content_sha256": content, "reference_sha": case.gold["reference_sha"],
            "candidate_sha": case.gold["candidate_sha"], "formal_training": True}


def _context(case: ReleaseCase, labels: list[dict], view: str) -> dict:
    parts = sorted({part for label in labels for part in label["parts"]}) or ["object"]
    joint = case.gold.get("joint_id")
    if not joint and labels:
        joint = labels[0].get("params", {}).get("joint")
    return {"task": "compare reference and candidate articulation", "parts": parts,
            "joints": [joint] if joint else [], "views": [view], "runtime_properties": [],
            "action_types": ["request_view"], "max_triangles": 1, "max_draw_calls": 1}


def export_releases(release_paths: list[Path], output: Path, dataset_version: str) -> dict:
    """Build one complete formal-training export from release directories."""
    manifests, cases = [], []
    for path in release_paths:
        manifest, release_cases = _release(path)
        manifests.append(manifest)
        cases.extend(release_cases)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    episodes = {"train": [], "validation": []}
    stopping = {"train": [], "validation": []}
    annotations, groups = [], {}
    for case in sorted(cases, key=lambda item: item.gold["case_id"]):
        _, _, _, group, content = _provenance(case)
        split = _split(group)
        groups[group] = split
        labels = _labels(case)
        view, history, paths, digests = _copy_images(case, output)
        covered = [0] if labels else []
        terminal = episode_examples(
            case.gold["case_id"], _context(case, labels, view), paths[0], history,
            list(range(len(history))), [covered] * len(history), labels, len(history),
            objective="comprehensive_within_budget")[-1]
        identity = _identity(case, group, content)
        terminal.update(data_identity=identity, image_sha256=digests[:len(terminal["images"])])
        episodes[split].append(terminal)
        stop = json.loads(json.dumps(terminal))
        stop["id"] = f"{case.gold['case_id']}:stopping"
        decision = ({"decision": "stop", "outcome": "fail", "reason": "Observed motion exposes the defect."}
                    if labels else
                    {"decision": "continue", "outcome": None, "reason": "Current observations establish no defect."})
        stop["messages"][1]["content"] = json.dumps({"stop_decision": decision})
        stopping[split].append(stop)
        annotations.append({"id": case.gold["case_id"], "split": split,
                            "reference_sha": identity["reference_sha"],
                            "candidate_sha": identity["candidate_sha"],
                            "data_identity": identity, "labels": labels})
    for split in episodes:
        for kind, rows in (("episodes", episodes[split]), ("stopping", stopping[split])):
            (output / f"{split}.{kind}.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (output / "annotations.private.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in annotations), encoding="utf-8")
    manifest = {"status": "complete", "formal_training": True, "dataset_version": dataset_version,
                "groups": groups, "source_releases": [row["tag"] for row in manifests],
                "counts": {split: {"episodes": len(episodes[split]), "stopping": len(stopping[split])}
                           for split in episodes}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
