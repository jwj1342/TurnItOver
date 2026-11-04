"""Portable Hard candidate schemas and custody boundaries."""

from __future__ import annotations

from pathlib import PurePosixPath

KINDS = {"axis_direction_3deg", "joint_anchor_offset", "moving_part_initial_rotation"}
KIND_COUNTS = {kind: 25 for kind in KINDS}
SOURCE_KEYS = (
    "source_identity",
    "model_id",
    "archive_sha256",
    "original_spec_sha256",
    "original_geometry_fingerprint",
)
STATES = ("rest", "mid", "limit")
VIEWS = ("initial_view", "supplementary_view")
ABSOLUTE_SOURCE_KEYS = {
    "source_archive",
    "source_spec",
    "source_urdf",
    "source_catalog",
}


def safe_path(value: str):
    path = PurePosixPath(value)
    if (
        not isinstance(value, str)
        or path.is_absolute()
        or "\\" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("Unsafe Hard bundle path")
    return path


def sanitize_gold(row: dict) -> dict:
    result = dict(row)
    result.pop("program", None)
    source = dict(result["source"])
    for key in ABSOLUTE_SOURCE_KEYS:
        source.pop(key, None)
    result["source"] = source
    result["construction_program"] = "local_custody"
    return result


def sanitize_check(row: dict) -> dict:
    result = dict(row)
    auth = dict(result["source_authenticity"])
    files = []
    for index, item in enumerate(auth.get("source_files", [])):
        files.append(
            {"logical_name": f"source-file-{index:04d}", "sha256": item["sha256"]}
        )
    auth["source_files"] = files
    result["source_authenticity"] = auth
    return result
