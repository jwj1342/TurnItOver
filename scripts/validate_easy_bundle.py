"""Validate the portable Easy dataset bundle without project dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

if __package__:
    from .easy_bundle_contract import validate_blind_review_sources, validate_imported_review
    from .easy_bundle_provenance import validate_provenance
    from .prepare_easy_validation_review import validate_review_bundle
else:
    from easy_bundle_contract import validate_blind_review_sources, validate_imported_review
    from easy_bundle_provenance import validate_provenance
    from prepare_easy_validation_review import validate_review_bundle


MANIFEST_NAME = "bundle_manifest.json"
EXPECTED_CASES = 275
EXPECTED_OBJECTS = 127
EXPECTED_IMPORT_CASES = 150


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe bundle path: {value!r}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"invalid PNG image: {path}")
    return struct.unpack(">II", header[16:24])


def write_manifest(root: Path) -> dict[str, str]:
    files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME
    }
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps({"version": 1, "files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return files


def _validate_manifest(root: Path) -> set[str]:
    """Check the local file inventory independently of release-sized ledgers."""
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("bundle must not contain symbolic links")
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError(f"unsupported or malformed {MANIFEST_NAME}")

    expected_files: set[str] = set()
    for relative, expected_hash in manifest["files"].items():
        rel_path = _safe_relative_path(relative)
        file_path = root.joinpath(*rel_path.parts)
        if not file_path.is_file():
            raise ValueError(f"manifest file is missing: {relative}")
        if not isinstance(expected_hash, str) or _sha256(file_path) != expected_hash:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        expected_files.add(relative)

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unlisted = sorted(actual_files - expected_files)
        raise ValueError(f"file inventory differs from manifest: missing={missing[:3]}, unlisted={unlisted[:3]}")
    return expected_files


def validate_bundle(root: Path) -> dict[str, Any]:
    root = root.resolve()
    expected_files = _validate_manifest(root)

    gold_rows = _read_jsonl(root / "private/gold.jsonl")
    review_rows = _read_jsonl(root / "review_manifest.jsonl")
    human_rows = _read_jsonl(root / "human_review.jsonl")
    check_rows = _read_jsonl(root / "private/import_checks.jsonl")
    split = json.loads((root / "splits.json").read_text(encoding="utf-8"))

    gold_by_id = {row["case_id"]: row for row in gold_rows}
    review_by_id = {row["case_id"]: row for row in review_rows}
    human_by_id = {row["case_id"]: row for row in human_rows}
    if len(gold_by_id) != len(gold_rows) or len(review_by_id) != len(review_rows) or len(human_by_id) != len(human_rows):
        raise ValueError("duplicate case IDs in a root ledger")
    if set(gold_by_id) != set(review_by_id) or set(gold_by_id) != set(human_by_id):
        raise ValueError("Gold, review manifest, and human review IDs do not align")
    if len(gold_rows) != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} Easy cases, found {len(gold_rows)}")
    validate_provenance(root, gold_by_id, split)

    objects_by_split: dict[str, set[str]] = defaultdict(set)
    cases_by_split: Counter[str] = Counter()
    case_splits: dict[str, str] = {}
    for entry in split.get("objects", []):
        source_identity = entry.get("source_identity")
        split_name = entry.get("split")
        if not source_identity or split_name not in {"train", "validation"}:
            raise ValueError("invalid object split entry")
        if source_identity in objects_by_split["train"] | objects_by_split["validation"]:
            raise ValueError(f"source identity appears more than once in split: {source_identity}")
        objects_by_split[split_name].add(source_identity)
        for case_id in entry.get("case_ids", []):
            row = gold_by_id.get(case_id)
            if row is None or row.get("source_identity") != source_identity:
                raise ValueError(f"split case does not match Gold provenance: {case_id}")
            if case_id in case_splits:
                raise ValueError(f"case appears more than once in split: {case_id}")
            case_splits[case_id] = split_name
            cases_by_split[split_name] += 1
    if set(objects_by_split["train"]) & set(objects_by_split["validation"]):
        raise ValueError("source object leaks across Train and Validation")
    if sum(map(len, objects_by_split.values())) != EXPECTED_OBJECTS:
        raise ValueError("split object count does not match the Easy collection")
    if set(split.get("counts", {})) != {"train", "validation"}:
        raise ValueError("split count summary is incomplete")
    for split_name in ("train", "validation"):
        summary = split["counts"][split_name]
        if summary.get("objects") != len(objects_by_split[split_name]) or summary.get("samples") != cases_by_split[split_name]:
            raise ValueError(f"split summary does not match object assignments: {split_name}")
        expected_objects, expected_cases = {"train": (117, 251), "validation": (10, 24)}[split_name]
        if (len(objects_by_split[split_name]), cases_by_split[split_name]) != (expected_objects, expected_cases):
            raise ValueError(f"unexpected release split counts: {split_name}")
    if cases_by_split["train"] + cases_by_split["validation"] != EXPECTED_CASES:
        raise ValueError("split does not cover every Easy case")

    referenced_media: set[str] = set()
    for case_id, review in review_by_id.items():
        media = [review.get("board"), review.get("input_preview"), *review.get("images", [])]
        if len(media) != 6 or any(not isinstance(path, str) for path in media):
            raise ValueError(f"expected board, preview and four state images: {case_id}")
        for value in media:
            rel_path = _safe_relative_path(value)
            media_path = root.joinpath(*rel_path.parts)
            if not media_path.is_file():
                raise ValueError(f"missing review media for {case_id}: {value}")
            width, height = _png_size(media_path)
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid image dimensions for {case_id}: {value}")
            referenced_media.add(value)
        if review.get("status") != human_by_id[case_id].get("status"):
            raise ValueError(f"review status mismatch for {case_id}")
    if len(referenced_media) != EXPECTED_CASES * 6:
        raise ValueError("review media is reused or incomplete")

    imported_ids = {
        case_id for case_id, row in gold_by_id.items()
        if row.get("origin") == "medium-v1-reclassified-obvious"
    }
    if len(imported_ids) != EXPECTED_IMPORT_CASES:
        raise ValueError(f"expected {EXPECTED_IMPORT_CASES} imported cases, found {len(imported_ids)}")
    if any(case_splits.get(case_id) != "train" for case_id in imported_ids):
        raise ValueError("imported objects must remain Train candidates")
    imported_objects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case_id in imported_ids:
        row = gold_by_id[case_id]
        imported_objects[row["source_identity"]].append(row)
        if row.get("difficulty") != "easy_provisional":
            raise ValueError(f"imported case is not marked Easy provisional: {case_id}")
        validate_imported_review(row, human_by_id[case_id])
    if len(imported_objects) != 75 or any(len(rows) != 2 for rows in imported_objects.values()):
        raise ValueError("imported cases must be clean/defect pairs for 75 objects")
    for source_identity, pair in imported_objects.items():
        kinds = [row.get("kind") for row in pair]
        if kinds.count("clean") != 1:
            raise ValueError(f"imported object does not have one Clean and one defect: {source_identity}")

    checks_by_id = {row["case_id"]: row for row in check_rows}
    if len(checks_by_id) != len(check_rows) or set(checks_by_id) != imported_ids:
        raise ValueError("import checker rows do not cover exactly the imported cases")
    for case_id, row in checks_by_id.items():
        gold = gold_by_id[case_id]
        if row.get("candidate_sha") != gold.get("candidate_sha") or row.get("reference_sha") != gold.get("reference_sha"):
            raise ValueError(f"checker hashes do not align with Gold: {case_id}")
        if row.get("code_correct") is not (gold.get("kind") == "clean"):
            raise ValueError(f"checker correctness does not align with case kind: {case_id}")
        if row.get("used_by_student") is not False:
            raise ValueError(f"checker result is not marked as held out: {case_id}")

    blind_review = validate_review_bundle(root / "validation_review")
    validate_blind_review_sources(root, review_by_id)
    return {
        "cases": len(gold_rows),
        "objects": EXPECTED_OBJECTS,
        "train_objects": len(objects_by_split["train"]),
        "train_cases": cases_by_split["train"],
        "validation_objects": len(objects_by_split["validation"]),
        "validation_cases": cases_by_split["validation"],
        "imported_cases": len(imported_ids),
        "imported_objects": len(imported_objects),
        "media_files": len(referenced_media),
        "manifest_files": len(expected_files),
        "human_review_status": dict(Counter(row["status"] for row in human_rows)),
        "independent_validation_review": blind_review,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="explicitly supplied, extracted Easy bundle directory")
    parser.add_argument("--write-manifest", action="store_true", help="write or refresh the relative-path SHA-256 manifest")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest(args.root.resolve())
    print(json.dumps(validate_bundle(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
