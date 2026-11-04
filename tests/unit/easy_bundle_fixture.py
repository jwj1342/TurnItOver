"""Tiny deterministic ledgers for unit checks; never a production release."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from scripts.validate_easy_bundle import write_manifest

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aYuoAAAAASUVORK5CYII="
)
IMPORTED_ORIGIN = "medium-v1-reclassified-obvious"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh(root: Path) -> None:
    split = json.loads((root / "splits.json").read_text())
    split["source"] = {"gold": "private/gold.jsonl", "sha256": file_sha(root / "private/gold.jsonl")}
    write_json(root / "splits.json", split)
    write_manifest(root)


def build_bundle(root: Path) -> Path:
    """Build four synthetic cases, two objects and 24 one-pixel media files."""
    gold, reviews, humans, checks, index, mappings = [], [], [], [], [], []
    archive = dict(id="synthetic-unit-fixture", filename="filesystem.tar.gz",
                   sha256=hashlib.sha256(b"synthetic archive identity").hexdigest(),
                   size=1, availability="local_custody", member_index="program_members.jsonl")
    for imported, origin in ((False, "easy-batch-v1"), (True, IMPORTED_ORIGIN)):
        for kind in ("clean", "direction_flip"):
            case_id = f"{'imported' if imported else 'original'}-{kind}"
            object_id = "fixture:imported" if imported else "fixture:original"
            row = dict(case_id=case_id, source_identity=object_id, origin=origin,
                       kind=kind, difficulty="easy_provisional",
                       expected_status="pass" if kind == "clean" else "fail",
                       training_eligibility="pending_human_review",
                       candidate_sha=hashlib.sha256(case_id.encode()).hexdigest()[:16], reference_sha="1" * 16)
            if imported:
                original = f"programs/{case_id}.json"
                program = dict(archive_id=archive["id"], member=f"filesystem/{original}",
                               sha256=hashlib.sha256(case_id.encode()).hexdigest(), size=1, original_path=original)
                row["program"] = program
                index.append({"case_id": case_id, **program})
                mappings.append(dict(case_id=case_id, origin=origin, original_program=original, member=program["member"]))
                checks.append({key: row[key] for key in ("case_id", "candidate_sha", "reference_sha")})
                checks[-1].update(code_correct=kind == "clean", used_by_student=False)
            gold.append(row)
            media = [f"{directory}/{case_id}{suffix}.png" for directory, suffix in
                     [("boards", ""), ("inputs", ""), *[("frames", f"-{i}") for i in range(4)]]]
            for relative in media:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(PNG)
            reviews.append(dict(case_id=case_id, board=media[0], input_preview=media[1], images=media[2:], status="pending_human_review"))
            humans.append(dict(case_id=case_id, status="pending_human_review", human_judgment={"verdict": None}))
    objects = [dict(source_identity=f"fixture:{name}", split=split_name,
                    case_ids=[row["case_id"] for row in gold if row["source_identity"] == f"fixture:{name}"])
               for name, split_name in (("original", "validation"), ("imported", "train"))]
    write_json(root / "archive/original_splits.json", {"objects": objects[:1]})
    original_ref = dict(path="archive/original_splits.json", sha256=file_sha(root / "archive/original_splits.json"))
    split = dict(objects=objects, counts={name: {"objects": 1, "samples": 2} for name in ("train", "validation")}, original_split=original_ref)
    write_json(root / "archive/source_splits.json", split)
    for name, rows in (("private/gold.jsonl", gold), ("review_manifest.jsonl", reviews),
                       ("human_review.jsonl", humans), ("private/import_checks.jsonl", checks),
                       ("archive/program_members.jsonl", index)):
        write_jsonl(root / name, rows)
    archive["custody_copies"] = [{"path": "synthetic-unavailable-custody", "verified_sha256": archive["sha256"]}]
    write_json(root / "archive/registry.json", {"version": 1, "archives": [archive]})
    write_json(root / "provenance.json", dict(
        source=dict(head="1" * 40, dirty=True, gold_sha256=file_sha(root / "private/gold.jsonl"), splits_sha256=file_sha(root / "archive/source_splits.json")),
        original_split=original_ref, archive={key: archive[key] for key in ("id", "filename", "sha256", "availability")},
        transformations=["deterministic synthetic unit fixture"],
        case_mappings={"original125": [{"case_id": row["case_id"], "origin": row["origin"]} for row in gold[:2]], "imported150": mappings}))
    write_json(root / "collection_approval.json", dict(scope="collection_inclusion", independent_per_case_review=False,
               statement="Synthetic unit-test inclusion only", case_ids=[row["case_id"] for row in gold[2:]]))
    manifest, media_provenance = [], []
    for review in reviews[:2]:
        blind = {key: review[key] for key in ("case_id", "board", "input_preview", "images")}
        manifest.append(blind)
        paths = [("board", blind["board"]), ("input_preview", blind["input_preview"]), *[(f"image_{i}", value) for i, value in enumerate(blind["images"])]]
        for role, relative in paths:
            media_provenance.append(dict(case_id=blind["case_id"], role=role, source_path=relative, derived_path=relative,
                                        source_sha256=file_sha(root / relative), derived_sha256=file_sha(root / relative)))
    write_json(root / "validation_review/case_manifest.json", manifest)
    write_json(root / "validation_review/media_provenance.json", media_provenance)
    write_json(root / "splits.json", split)
    refresh(root)
    return root
