"""Validate immutable, private inputs for a matched repair evaluation.

The declared validation split must be checked against its authoritative source
before a manifest is frozen. This loader validates the declaration and hashes;
it cannot independently establish the provenance of a split.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

_FIELDS = ("reference_image", "initial_program", "reference_program")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _file(root: Path, relative: object, expected: object, field: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Invalid relative path for {field}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Invalid or missing file for {field}")
    if not isinstance(expected, str) or not _SHA.fullmatch(expected):
        raise ValueError(f"Missing or invalid SHA256 for {field}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"SHA256 mismatch for {field}")
    return path


def load_frozen_manifest(path: Path) -> tuple[dict, list[dict]]:
    """Load a hash-bound validation manifest; paths are relative to its directory.

    Each case contains a safe unique id, source_group, three file paths, and a
    sha256 object with matching keys. Additional case metadata is retained.
    """
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("status") != "frozen":
        raise ValueError("Expected a frozen repair manifest with schema_version 1")
    if manifest.get("split") != "validation":
        raise ValueError("Repair manifest must declare the validation split")
    rows = manifest.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Repair manifest must contain cases")
    cases, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Repair case must be an object")
        case_id = row.get("id")
        if not isinstance(case_id, str) or not _ID.fullmatch(case_id) or case_id in seen:
            raise ValueError("Repair cases must have unique safe ids")
        seen.add(case_id)
        if not isinstance(row.get("source_group"), str) or not row["source_group"].strip():
            raise ValueError(f"Missing source_group for {case_id}")
        hashes = row.get("sha256")
        if not isinstance(hashes, dict):
            raise ValueError(f"Missing hashes for {case_id}")
        files = {field: _file(path.parent, row.get(field), hashes.get(field), field) for field in _FIELDS}
        cases.append({**row, **files})
    return manifest, cases
