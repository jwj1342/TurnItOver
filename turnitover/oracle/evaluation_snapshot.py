"""Self-contained, hash-checked scoring inputs, separate from model prompts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = 1


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _unique(rows, label):
    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Missing or duplicate {label} ID")


def _inside(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Snapshot paths must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path outside snapshot or dataset")
    return path


def create_snapshot(output, dataset, records, split, task, defect_ids):
    """Freeze actual evaluation records and images; return verified snapshot data.

    Only the top-level image paths are rewritten. Original image paths live in
    a separate mapping file; messages and private annotations are never merged.
    The final manifest is the completion marker and is published atomically.
    """
    dataset = Path(dataset).resolve()
    records = copy.deepcopy(list(records))
    _unique(records, "record")
    if not records:
        raise ValueError("Cannot snapshot an empty evaluation")
    episode_ids = {record.get("episode_id") for record in records}
    if any(not isinstance(value, str) or not value for value in episode_ids):
        raise ValueError("Record is missing episode_id")
    dataset_manifest = (dataset / "manifest.json").read_bytes()
    annotation_bytes = (dataset / "annotations.private.jsonl").read_bytes()
    annotations = [json.loads(line) for line in annotation_bytes.decode("utf-8").splitlines() if line.strip()]
    _unique(annotations, "annotation")
    annotations = [row for row in annotations if row["id"] in episode_ids]
    if {row["id"] for row in annotations} != episode_ids:
        raise ValueError("Missing annotation for evaluated episode")
    if any(row.get("split", split) != split for row in annotations):
        raise ValueError("Annotation split differs from evaluation")
    root = Path(output) / "scoring-snapshot"
    root.mkdir(parents=True, exist_ok=False)
    files = {}

    def write(relative, data):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[relative] = _sha(path)

    write("dataset-manifest.json", dataset_manifest)
    write("annotations.private.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n"
                                             for row in annotations).encode())
    mappings = []
    for record in records:
        original = record.get("images")
        hashes = record.get("image_sha256")
        if not isinstance(original, list) or not isinstance(hashes, list) or len(original) != len(hashes):
            raise ValueError("Record image paths and hashes must align")
        mapped = []
        for relative, expected in zip(original, hashes):
            source = _inside(dataset, relative)
            data = source.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != expected:
                raise ValueError(f"Image hash mismatch: {relative}")
            destination = f"images/{digest}"
            if destination not in files:
                write(destination, data)
            mapped.append(destination)
        mappings.append({"id": record["id"], "original_images": original,
                         "snapshot_images": mapped})
        record["images"] = mapped
    for relative, rows in (("records.jsonl", records), ("image-paths.jsonl", mappings)):
        write(relative, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode())
    manifest = {"schema_version": SCHEMA_VERSION, "case_count": len(records),
                "split": split, "task": task, "defect_ids": list(defect_ids),
                "source_sha256": {"dataset_manifest": hashlib.sha256(dataset_manifest).hexdigest(),
                                  "annotations": hashlib.sha256(annotation_bytes).hexdigest()},
                "files": files}
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "manifest.json")
    return load_snapshot(output)


def load_snapshot(output):
    """Load only snapshot files, rejecting missing, altered, or ambiguous inputs.

    Returns root (Path), manifest, dataset_manifest, records, annotations (list),
    annotations_by_id, and image_paths (list). Image paths are relative to root.
    """
    root = (Path(output) / "scoring-snapshot").resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported scoring snapshot schema")
    files = manifest.get("files", {})
    required = {"dataset-manifest.json", "annotations.private.jsonl", "records.jsonl", "image-paths.jsonl"}
    if not required <= files.keys():
        raise ValueError("Snapshot manifest is missing required files")
    for relative, expected in files.items():
        if _sha(_inside(root, relative)) != expected:
            raise ValueError(f"Snapshot hash mismatch: {relative}")
    records = _jsonl(root / "records.jsonl")
    annotations = _jsonl(root / "annotations.private.jsonl")
    mappings = _jsonl(root / "image-paths.jsonl")
    for rows, label in ((records, "record"), (annotations, "annotation"), (mappings, "mapping")):
        _unique(rows, label)
    if not records or len(records) != manifest["case_count"]:
        raise ValueError("Snapshot case count mismatch")
    episode_ids = {record.get("episode_id") for record in records}
    if any(not isinstance(value, str) or not value for value in episode_ids):
        raise ValueError("Snapshot record is missing episode_id")
    if {row["id"] for row in annotations} != episode_ids:
        raise ValueError("Snapshot annotation IDs differ from evaluated episodes")
    if any(row.get("split", manifest["split"]) != manifest["split"] for row in annotations):
        raise ValueError("Snapshot annotation split mismatch")
    mapping_by_id = {row["id"]: row for row in mappings}
    if set(mapping_by_id) != {row["id"] for row in records}:
        raise ValueError("Snapshot image mapping IDs differ")
    for record in records:
        paths, hashes = record["images"], record["image_sha256"]
        mapping = mapping_by_id[record["id"]]
        if paths != mapping["snapshot_images"] or len(paths) != len(hashes) or len(paths) != len(mapping["original_images"]):
            raise ValueError("Snapshot image mapping mismatch")
        for relative, expected in zip(paths, hashes):
            if relative not in files or files[relative] != expected:
                raise ValueError("Snapshot image is missing or has a different hash")
    return {"root": root, "manifest": manifest, "records": records,
            "dataset_manifest": json.loads((root / "dataset-manifest.json").read_text(encoding="utf-8")),
            "annotations": annotations, "annotations_by_id": {row["id"]: row for row in annotations},
            "image_paths": mappings}
