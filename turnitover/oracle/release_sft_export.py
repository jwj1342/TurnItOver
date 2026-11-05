"""Convert TurnItOver release ZIPs into fixed-observation SFT exports."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zipfile

from turnitover.oracle.episodes import episode_examples
from turnitover.oracle.evidence import pixel_signal, pixels


HARD_DEFECTS = {
    "axis_direction_3deg": "kinematics.joint_axis",
    "joint_anchor_offset": "kinematics.joint_anchor",
    "moving_part_initial_rotation": "structure.part_rotation",
}
HARD_COMPOSITE_DEFECTS = {
    "joint_anchor": "kinematics.joint_anchor",
    "joint_axis": "kinematics.joint_axis",
    "part_offset": "structure.part_offset",
    "part_rotation": "structure.part_rotation",
}
SEVERITY = {"easy": 0.9, "medium": 0.6, "hard": 0.35}
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class ReleaseCase:
    level: str
    archive: Path
    root: str
    gold: dict
    public: dict


@dataclass(frozen=True)
class ImageMember:
    role: str
    path: str
    view: str
    kind: str
    state: str


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


def _candidate_bundle(path: Path) -> tuple[dict, list[ReleaseCase]]:
    """Load the private Hard v1.1 candidate ZIP without treating it as a release."""
    if not path.is_file():
        raise ValueError(f"Candidate bundle is not a file: {path}")
    with zipfile.ZipFile(path) as source:
        roots = {name.split("/", 1)[0] for name in source.namelist() if "/" in name}
        if len(roots) != 1:
            raise ValueError("Candidate bundle must contain exactly one root directory")
        root = roots.pop()
        gold = _jsonl(source.read(f"{root}/private/gold.jsonl"))
        pool = {row["case_id"]: row for row in _jsonl(source.read(f"{root}/pool_manifest.jsonl"))}
        bundle_manifest = json.loads(source.read(f"{root}/manifest.json"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"tag": bundle_manifest.get("policy", root), "kind": "candidate_bundle",
                "archive_sha256": digest, "bundle_status": bundle_manifest.get("status")}
    return manifest, [ReleaseCase("hard", path, root, row, pool.get(row["case_id"], {})) for row in gold]


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
    bucket = value % 20
    return "train" if bucket < 14 else "validation" if bucket < 17 else "test"


def _labels(case: ReleaseCase) -> list[dict]:
    gold = case.gold
    if gold.get("defects"):
        labels = []
        for defect in gold["defects"]:
            defect_id = HARD_COMPOSITE_DEFECTS.get(defect.get("field"))
            if not defect_id:
                raise ValueError(f"Unsupported composite defect: {defect.get('field')}")
            labels.append({"defect_id": defect_id,
                           "parts": [defect.get("part_id") or "moving_part"],
                           "severity": SEVERITY[case.level],
                           "params": {**dict(defect.get("parameters") or {}),
                                      "joint": defect.get("joint_id") or "joint_0"}})
        return labels
    defect_id = gold.get("defect_id") or HARD_DEFECTS.get(gold.get("kind"))
    if not defect_id:
        return []
    parts = list(gold.get("parts") or ([gold["part_id"]] if gold.get("part_id") else ["moving_part"]))
    params = dict(gold.get("parameters") or {})
    if defect_id == "kinematics.joint_axis":
        params["joint"] = gold.get("joint_id") or params.get("joint") or "joint_0"
    return [{"defect_id": defect_id, "parts": parts, "severity": SEVERITY[case.level], "params": params}]


def _members(case: ReleaseCase) -> list[ImageMember]:
    cid, prefix = case.gold["case_id"], f"{case.root}/"
    view = case.public.get("view", "front")
    if case.level == "easy":
        roles = (("reference_rest", "reference", "rest"), ("reference_limit", "reference", "limit"),
                 ("candidate_rest", "candidate", "rest"), ("candidate_limit", "candidate", "limit"))
        return [ImageMember(role, prefix + relative, view, kind, state)
                for relative, (role, kind, state) in zip(case.public["images"], roles, strict=True)]
    if case.level == "medium":
        roles = ("reference_rest", "reference_mid", "reference_limit",
                 "candidate_rest", "candidate_mid", "candidate_limit")
        return [ImageMember(role, f"{prefix}native/{cid}/{role}.png", view, *role.split("_"))
                for role in roles]
    views = (case.gold.get("initial_view") or case.public["initial_view"],
             case.gold.get("supplementary_view") or case.public["supplementary_view"])
    members = []
    for view in views:
        for kind in ("reference", "candidate"):
            for state in ("rest", "mid", "limit"):
                role = f"{view}_{kind}_{state}"
                members.append(ImageMember(role, f"{prefix}native/{cid}/{view}/{kind}_{state}.png",
                                           view, kind, state))
    return members


def _copy_images(case: ReleaseCase, output: Path) -> tuple[list[ImageMember], list[dict], list[str], list[str], list[bytes]]:
    members = _members(case)
    paths, digests, images = [], [], []
    with zipfile.ZipFile(case.archive) as source:
        for index, member in enumerate(members):
            data = source.read(member.path)
            relative = f"images/{case.gold['case_id']}/{index:02d}-{member.role}.png"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            paths.append(relative)
            digests.append(hashlib.sha256(data).hexdigest())
            images.append(data)
    history = [{"step": index, "image_ref": relative,
                "action": {"type": "request_view", "view_id": member.view}}
               for index, (member, relative) in enumerate(zip(members[1:], paths[1:], strict=True))]
    return members, history, paths, digests, images


def _identity(case: ReleaseCase, split_group: str, content: str) -> dict:
    dataset, source_asset, asset, _, _ = _provenance(case)
    return {"source_dataset": dataset, "source_asset_id": source_asset, "asset_id": asset,
            "split_group": split_group,
            "source_identity_sha256": hashlib.sha256(f"{dataset}:{source_asset}".encode()).hexdigest(),
            "source_content_sha256": content, "reference_sha": case.gold["reference_sha"],
            "candidate_sha": case.gold["candidate_sha"], "formal_training": True}


def _asset_index(asset_roots: list[Path]) -> dict[tuple[str, str], Path]:
    return {(path.stem, hashlib.sha256(path.read_bytes()).hexdigest()): path
            for root in asset_roots for path in root.rglob("PM_*.json")}


def _roster(case: ReleaseCase, assets: dict[tuple[str, str], Path]) -> tuple[list[str], list[str]]:
    expected = case.gold.get("source", {}).get("original_spec_sha256") or case.gold.get("original_spec_sha256")
    spec_path = assets.get((case.gold["asset_id"], expected))
    if spec_path:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        return [row["id"] for row in spec["parts"]], [row["id"] for row in spec["joints"]]
    raise ValueError(f"No exact part/joint roster for {case.gold['case_id']}")


def _exposures(members: list[ImageMember], images: list[bytes], labels: list[dict]) -> list[list[int]]:
    if not labels:
        return [[] for _ in members[1:]]
    references = {(member.view, member.state): image for member, image in zip(members, images, strict=True)
                  if member.kind == "reference"}
    return [[0] if member.kind == "candidate"
            and pixel_signal(pixels(references[(member.view, member.state)]), pixels(image))["exposed"] else []
            for member, image in zip(members[1:], images[1:], strict=True)]


def _context(parts: list[str], joints: list[str], members: list[ImageMember]) -> dict:
    views = list(dict.fromkeys(member.view for member in members))
    return {"task": "compare reference and candidate articulation", "parts": parts,
            "joints": joints, "views": views, "runtime_properties": [],
            "action_types": ["request_view"], "max_triangles": 1, "max_draw_calls": 1}


def _case_split(case: ReleaseCase, split_plan: dict[str, str] | None, *, allow_hash_split: bool) -> str:
    _, _, _, group, _ = _provenance(case)
    # Easy/Medium source metadata may carry an upstream acquisition split.  Only
    # the Hard candidate's object split is the frozen three-tier assignment.
    frozen = case.gold.get("source", {}).get("split") if case.level == "hard" else None
    split = frozen or (split_plan or {}).get(group)
    if split in SPLITS:
        return split
    if allow_hash_split:
        return _split(group)
    raise ValueError(f"No frozen split for source group {group}")


def _export_cases(manifests: list[dict], cases: list[ReleaseCase], output: Path, dataset_version: str,
                  asset_roots: list[Path], *, split_plan: dict[str, str] | None = None,
                  allow_hash_split: bool, expected_difficulty_counts: dict[str, int] | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    assets = _asset_index(asset_roots)
    episodes = {split: [] for split in SPLITS}
    stopping = {split: [] for split in SPLITS}
    annotations, groups, group_splits = [], {}, {}
    for case in sorted(cases, key=lambda item: item.gold["case_id"]):
        _, _, _, group, content = _provenance(case)
        split = _case_split(case, split_plan, allow_hash_split=allow_hash_split)
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Source group assigned to multiple splits: {group}")
        group_splits[group] = split
        groups[group] = split
        labels = _labels(case)
        parts, joints = _roster(case, assets)
        members, history, paths, digests, images = _copy_images(case, output)
        exposures = _exposures(members, images, labels)
        terminal = episode_examples(
            case.gold["case_id"], _context(parts, joints, members), paths[0], history,
            list(range(len(history))), exposures, labels, len(history),
            objective="comprehensive_within_budget")[-1]
        identity = _identity(case, group, content)
        terminal.update(data_identity=identity, difficulty=case.level,
                        image_sha256=digests[:len(terminal["images"])])
        episodes[split].append(terminal)
        stop = json.loads(json.dumps(terminal))
        stop["id"] = f"{case.gold['case_id']}:stopping"
        visible = terminal["evidence"]["covered_label_indices"]
        decision = ({"decision": "stop", "outcome": "fail", "reason": "Observed motion exposes the defect."}
                    if visible else
                    {"decision": "continue", "outcome": None, "reason": "Current observations establish no defect."})
        stop["messages"][1]["content"] = json.dumps({"stop_decision": decision})
        stopping[split].append(stop)
        annotations.append({"id": case.gold["case_id"], "split": split,
                            "reference_sha": identity["reference_sha"],
                            "candidate_sha": identity["candidate_sha"],
                            "data_identity": identity, "difficulty": case.level, "labels": labels})
    for split in episodes:
        for kind, rows in (("episodes", episodes[split]), ("stopping", stopping[split])):
            (output / f"{split}.{kind}.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (output / "annotations.private.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in annotations), encoding="utf-8")
    difficulty_counts = {level: sum(row["difficulty"] == level for row in annotations) for level in SEVERITY}
    if expected_difficulty_counts and difficulty_counts != expected_difficulty_counts:
        raise ValueError(f"Unexpected per-tier case counts: {difficulty_counts}; expected {expected_difficulty_counts}")
    taxonomy = sorted({label["defect_id"] for row in annotations for label in row["labels"]})
    manifest = {"status": "complete", "formal_training": True, "dataset_version": dataset_version,
                "groups": groups, "source_releases": [row["tag"] for row in manifests],
                "experiment_taxonomy": taxonomy,
                "counts": {split: {"episodes": len(episodes[split]), "stopping": len(stopping[split]),
                                    "difficulty": {level: sum(row["difficulty"] == level
                                                              for row in episodes[split])
                                                   for level in SEVERITY}}
                           for split in SPLITS}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def export_releases(release_paths: list[Path], output: Path, dataset_version: str,
                    asset_roots: list[Path]) -> dict:
    """Build one complete formal-training export from release directories."""
    manifests, cases = [], []
    for path in release_paths:
        manifest, release_cases = _release(path)
        manifests.append(manifest)
        cases.extend(release_cases)
    return _export_cases(manifests, cases, output, dataset_version, asset_roots, allow_hash_split=True)


def export_three_tier_900(release_paths: list[Path], hard_candidate: Path, output: Path,
                          dataset_version: str, asset_roots: list[Path], split_plan: dict[str, str]) -> dict:
    """Export the frozen 300-case Easy/Medium/Hard SFT corpus.

    Easy and Medium must be assigned by ``split_plan``.  Hard v1.1 retains the
    object-level split embedded in its private Gold records.
    """
    manifests, cases = [], []
    for path in release_paths:
        manifest, release_cases = _release(path)
        manifests.append(manifest)
        cases.extend(release_cases)
    manifest, hard_cases = _candidate_bundle(hard_candidate)
    manifests.append(manifest)
    cases.extend(hard_cases)
    return _export_cases(manifests, cases, output, dataset_version, asset_roots,
                         split_plan=split_plan, allow_hash_split=False,
                         expected_difficulty_counts={"easy": 300, "medium": 300, "hard": 300})
