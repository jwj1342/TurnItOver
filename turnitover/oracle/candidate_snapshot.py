"""Freeze native candidate inputs and private truth without changing admission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from turnitover.taxonomy import load_taxonomy

SCHEMA = "native-candidate-evaluation-v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inside(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Expected a relative snapshot path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes dataset or snapshot")
    return path


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bound_bytes(root, entry):
    data = inside(root, entry["path"]).read_bytes()
    if digest(data) != entry["sha256"]:
        raise ValueError(f"Hash mismatch: {entry['path']}")
    return data


def prepare_candidate_snapshot(dataset, out, split="validation", *, manifest_sha256):
    """Freeze only the explicitly selected split of the hash-pinned native release."""
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    dataset, out = Path(dataset).resolve(), Path(out).resolve()
    data = (dataset / "manifest.private.json").read_bytes()
    if digest(data) != manifest_sha256:
        raise ValueError("Source manifest hash mismatch")
    source = json.loads(data)
    if source.get("format") != "tio-750-complete-candidate-custody-v1":
        raise ValueError("Unsupported native candidate format")
    taxonomy = list(load_taxonomy().ids)
    selected = []
    seen = set()
    for entry in source["cases"]:
        parts = Path(entry["path"]).parts
        if len(parts) < 3 or parts[:2] != ("cases", split):
            continue  # Never open other splits' case JSON or Gold.
        uid = entry.get("id")
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError("Missing or duplicate selected case ID")
        seen.add(uid)
        selected.append(entry)
    if not selected:
        raise ValueError("Selected split has no cases")
    expected_count = source.get("counts", {}).get("split", {}).get(split)
    if expected_count is not None and expected_count != len(selected):
        raise ValueError("Selected split count differs from source manifest")
    if out.exists():
        raise ValueError("Fresh snapshot output required")
    out.mkdir(parents=True)
    files = {}

    def write(relative, content):
        path = inside(out, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files[relative] = digest(content)

    write("source-manifest.private.json", data)
    cases, inputs = [], []
    for entry in selected:
        relative = entry["path"]
        metadata = source["files"][relative]
        raw_case = bound_bytes(dataset, {"path": relative, **metadata})
        case = json.loads(raw_case)
        if case.get("id") != entry["id"] or case.get("split") != split:
            raise ValueError("Selected case identity or split mismatch")
        labels = case.get("labels")
        if (not isinstance(labels, list) or any(label not in taxonomy for label in labels)
                or len(labels) != len(set(labels))):
            raise ValueError("Invalid candidate labels")
        if type(case.get("clean")) is not bool or case["clean"] != (not labels):
            raise ValueError("Clean flag and labels disagree")
        if not isinstance(case.get("source_key"), str) or not case["source_key"]:
            raise ValueError("Missing source group")
        gold_entry = case["assets"]["gold_file"]
        gold_bytes = bound_bytes(dataset, gold_entry)
        gold = json.loads(gold_bytes)
        if gold.get("case_id") != case["id"] or set(gold.get("defect_ids", [])) != set(labels):
            raise ValueError("Gold identity or labels disagree")
        if case["clean"] and (case["assets"]["clean_program"]["sha256"] !=
                              case["assets"]["candidate_program"]["sha256"]):
            raise ValueError("Clean candidate differs from frozen reference")
        write(relative, raw_case)
        gold_path = f"gold/{gold_entry['sha256']}.json"
        write(gold_path, gold_bytes)
        observations, images = [], []
        for observation in case.get("observations", []):
            public = {key: observation[key] for key in ("state", "view_id")}
            for role in ("reference_image", "candidate_image"):
                image_entry = observation[role]
                image = bound_bytes(dataset, image_entry)
                destination = f"images/{digest(image)}.png"
                if destination not in files:
                    write(destination, image)
                public[role] = destination
                images.append(destination)
            observations.append(public)
        if not observations:
            raise ValueError("Case has no image pairs")
        inputs.append({"id": case["id"], "observations": observations, "images": images})
        cases.append({"id": case["id"], "path": relative, "gold_path": gold_path,
                      "images": images})
    write("inputs.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in inputs).encode())
    manifest = {"schema": SCHEMA, "status": "complete", "split": split,
                "case_count": len(cases), "cases": cases, "taxonomy_ids": taxonomy,
                "label_scope": "fixed_full_taxonomy_including_zero_support_labels",
                "source_manifest_sha256": manifest_sha256, "source_status": source.get("status"),
                "human_review_status": source.get("human_review_status"),
                "training_eligible_cases": source.get("training_eligible_cases"),
                "formal_admitted_cases": source.get("formal_admitted_cases"), "files": files}
    content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (out / "manifest.json").write_bytes(content)
    (out / "manifest.sha256").write_text(digest(content) + "\n", encoding="utf-8")
    return manifest


def load_candidate_snapshot(snapshot):
    root = Path(snapshot).resolve()
    raw = (root / "manifest.json").read_bytes()
    if digest(raw) != (root / "manifest.sha256").read_text().strip():
        raise ValueError("Snapshot manifest hash mismatch")
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("Incomplete or unsupported native candidate snapshot")
    for relative, expected in manifest["files"].items():
        if digest(inside(root, relative).read_bytes()) != expected:
            raise ValueError(f"Snapshot hash mismatch: {relative}")
    if manifest["files"].get("source-manifest.private.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Source manifest binding differs")
    cases = []
    for entry in manifest["cases"]:
        for path in (entry["path"], entry["gold_path"], *entry["images"]):
            if path not in manifest["files"]:
                raise ValueError("Unbound snapshot case file")
        case = json.loads(inside(root, entry["path"]).read_text())
        if case["id"] != entry["id"] or case["split"] != manifest["split"]:
            raise ValueError("Snapshot case identity differs")
        cases.append(case)
    ids = [case["id"] for case in cases]
    if len(ids) != manifest["case_count"] or len(set(ids)) != len(ids):
        raise ValueError("Snapshot case count or uniqueness differs")
    return root, manifest, cases
