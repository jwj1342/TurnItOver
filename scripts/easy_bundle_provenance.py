"""Check portable and archived references without requiring the sealed archive."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _relative(value: Any) -> bool:
    return (
        isinstance(value, str) and bool(value)
        and not PurePosixPath(value).is_absolute()
        and all(part not in {"", ".", ".."} for part in value.split("/"))
    )


def validate_provenance(root: Path, gold: dict[str, dict[str, Any]], split: dict[str, Any]) -> None:
    provenance = _json(root / "provenance.json")
    source = provenance.get("source", {})
    if (
        not isinstance(source.get("head"), str)
        or re.fullmatch(r"[0-9a-f]{40}", source["head"]) is None
        or source.get("dirty") is not True
        or not _sha(source.get("gold_sha256"))
        or not _sha(source.get("splits_sha256"))
        or not provenance.get("transformations")
    ):
        raise ValueError("provenance lacks source snapshot or transformations")
    current_source = split.get("source", {})
    if current_source != {"gold": "private/gold.jsonl", "sha256": _hash(root / "private/gold.jsonl")}:
        raise ValueError("split does not bind the current relative Gold file")
    if source["splits_sha256"] != _hash(root / "archive/source_splits.json"):
        raise ValueError("source split snapshot hash differs from provenance")

    original_ref = provenance.get("original_split", {})
    if (
        original_ref != split.get("original_split")
        or original_ref.get("path") != "archive/original_splits.json"
        or original_ref.get("sha256") != _hash(root / "archive/original_splits.json")
    ):
        raise ValueError("original split snapshot reference is invalid")
    original_split = _json(root / "archive/original_splits.json")
    current_assignments = {
        entry["source_identity"]: (entry["split"], set(entry["case_ids"]))
        for entry in split["objects"]
    }
    source_assignments = {
        entry["source_identity"]: (entry["split"], set(entry["case_ids"]))
        for entry in _json(root / "archive/source_splits.json")["objects"]
    }
    if source_assignments != current_assignments:
        raise ValueError("source object split assignment changed")
    original_ids = set()
    for entry in original_split["objects"]:
        original_ids.update(entry["case_ids"])
        if current_assignments.get(entry["source_identity"]) != (entry["split"], set(entry["case_ids"])):
            raise ValueError("original object split assignment changed")

    imported_ids = {case_id for case_id, row in gold.items() if row.get("origin") == "medium-v1-reclassified-obvious"}
    if original_ids != set(gold) - imported_ids:
        raise ValueError("original split does not cover exactly the original cases")
    mappings = provenance.get("case_mappings", {})
    for key, expected in (("original125", original_ids), ("imported150", imported_ids)):
        rows = mappings.get(key, [])
        if len(rows) != len(expected) or {row.get("case_id") for row in rows} != expected:
            raise ValueError(f"provenance case mapping differs: {key}")
        if any(row.get("origin") != gold[row["case_id"]].get("origin") for row in rows):
            raise ValueError(f"provenance origin mapping differs: {key}")

    registry = _json(root / "archive/registry.json")
    archives = registry.get("archives", [])
    if registry.get("version") != 1 or len(archives) != 1:
        raise ValueError("archive registry must identify one sealed source archive")
    archive = archives[0]
    archive_ref = provenance.get("archive", {})
    if (
        not archive.get("id") or archive.get("filename") != "filesystem.tar.gz"
        or archive.get("availability") != "local_custody"
        or not _sha(archive.get("sha256"))
        or not isinstance(archive.get("size"), int) or archive["size"] <= 0
        or archive.get("member_index") != "program_members.jsonl"
        or any(archive_ref.get(key) != archive.get(key) for key in ("id", "filename", "sha256", "availability"))
    ):
        raise ValueError("archive registry and provenance do not agree")
    copies = archive.get("custody_copies", [])
    if not copies or any(not row.get("path") or row.get("verified_sha256") != archive["sha256"] for row in copies):
        raise ValueError("archive custody verification metadata is incomplete")

    index_rows = [json.loads(line) for line in (root / "archive/program_members.jsonl").read_text().splitlines() if line.strip()]
    index = {row["case_id"]: row for row in index_rows}
    if len(index_rows) != len(imported_ids) or set(index) != imported_ids:
        raise ValueError("archive member index does not cover the imported cases")
    imported_mappings = {row["case_id"]: row for row in mappings["imported150"]}
    for case_id in imported_ids:
        program = gold[case_id].get("program")
        expected = {key: value for key, value in index[case_id].items() if key != "case_id"}
        if not isinstance(program, dict) or program != expected:
            raise ValueError(f"program archive reference differs from index: {case_id}")
        if (
            program.get("archive_id") != archive["id"]
            or not _relative(program.get("member"))
            or not program["member"].startswith("filesystem/")
            or not _relative(program.get("original_path"))
            or not _sha(program.get("sha256"))
            or not isinstance(program.get("size"), int) or program["size"] <= 0
            or imported_mappings[case_id].get("member") != program["member"]
            or imported_mappings[case_id].get("original_program") != program["original_path"]
        ):
            raise ValueError(f"invalid archived program reference: {case_id}")

    approval = _json(root / "collection_approval.json")
    approved_ids = approval.get("case_ids", [])
    if (
        approval.get("scope") != "collection_inclusion"
        or approval.get("independent_per_case_review") is not False
        or not approval.get("statement")
        or len(approved_ids) != len(imported_ids) or set(approved_ids) != imported_ids
    ):
        raise ValueError("collection inclusion approval is incomplete or misrepresented")
