"""Prepare an offline blind review and merge independently returned annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath


REVIEWERS = ("jwj1342", "WinterCCC")
VALIDATION_IDS_SHA256 = "55db11d9f3aa8d2f65a6cd0ca90524a8f93c130cf7b18a994abe9937ecb1062f"
VALIDATION_OBJECTS_SHA256 = "9867caace32a3e06545382acf5529e6178ca4426deca59572a4fa2cf9e362041"
BOARD_SIZE = (912, 630)
BOARD_CROP = (0, 60, 912, 630)
MEDIA_FIELDS = {"case_id", "board", "input_preview", "images"}
RESULT_FIELDS = {"case_id", "reviewer", "review_status", "verdict", "defect_type", "one_glance", "no_clipping_or_occlusion", "notes", "reviewed_at"}
DECISIONS = ("verdict", "defect_type", "one_glance", "no_clipping_or_occlusion")
STATIC_FILES = {"README.md", "reviewer.template.html", "review.js", "review.css"}
ASSETS_DIR = Path(__file__).resolve().parent / "easy_review_assets"
SELECTION_PATH = Path(__file__).resolve().parents[1] / "benchmark/releases/easy-validation-selection.json"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe media path: {value!r}")
    return Path(*path.parts)


def _check_ids(ids: list[str]) -> None:
    digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    if len(ids) != 24 or len(set(ids)) != 24 or digest != VALIDATION_IDS_SHA256:
        raise ValueError("Validation must retain the fixed 24 case IDs")


def blank_result(case_id: str, reviewer: str) -> dict:
    return dict(case_id=case_id, reviewer=reviewer, review_status="pending", verdict=None,
                defect_type=None, one_glance=None, no_clipping_or_occlusion=None,
                notes="", reviewed_at=None)


def prepare_review(root: Path, output: Path, zip_path: Path | None = None) -> dict:
    # Deliberately never read Gold, checker outputs, or historical human reviews.
    from PIL import Image

    if output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()):
        raise ValueError("source data and independent review output must not overlap")
    if zip_path is not None and zip_path.resolve().is_relative_to(output.resolve()):
        raise ValueError("write ZIP outside the distributable directory")
    split = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    objects = [row for row in split["objects"] if row["split"] == "validation"]
    if len(objects) != 10:
        raise ValueError("Validation must retain exactly 10 objects")
    assignments = sorted((row["source_identity"], sorted(row["case_ids"])) for row in objects)
    mapping_sha = hashlib.sha256(json.dumps(assignments, separators=(",", ":")).encode()).hexdigest()
    if mapping_sha != VALIDATION_OBJECTS_SHA256:
        raise ValueError("Validation must retain the fixed 24 case IDs and 10 object assignments")
    ids = sorted(case_id for row in objects for case_id in row["case_ids"])
    _check_ids(ids)
    source_rows = _jsonl(root / "review_manifest.jsonl")
    by_id = {row["case_id"]: row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("duplicate source manifest case IDs")
    output.mkdir(parents=True, exist_ok=True)
    for name in STATIC_FILES:
        if (ASSETS_DIR / name).resolve() != (output / name).resolve():
            shutil.copyfile(ASSETS_DIR / name, output / name)
    manifest, provenance = [], []
    for case_id in ids:
        source = by_id[case_id]
        row = {"case_id": case_id}
        media = [("board", source["board"]), ("input_preview", source["input_preview"])]
        media += [(f"image_{i}", value) for i, value in enumerate(source["images"])]
        if len(media) != 6:
            raise ValueError(f"expected six media files: {case_id}")
        row["images"] = []
        for role, relative in media:
            source_path = root / _relative(relative)
            target_relative = f"media/{case_id}/{role}.png"
            target = output / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            crop = None
            if role == "board":
                with Image.open(source_path) as image:
                    if image.size != BOARD_SIZE:
                        raise ValueError(f"unexpected board dimensions: {case_id}: {image.size}")
                    image.crop(BOARD_CROP).save(target)
                crop = list(BOARD_CROP)
            else:
                shutil.copyfile(source_path, target)
            if role.startswith("image_"):
                row["images"].append(target_relative)
            else:
                row[role] = target_relative
            provenance.append(dict(case_id=case_id, role=role, source_path=relative,
                                   source_sha256=_sha(source_path), crop=crop,
                                   derived_path=target_relative, derived_sha256=_sha(target)))
        manifest.append(row)
    _write_json(output / "case_manifest.json", manifest)
    _write_json(output / "media_provenance.json", provenance)
    template = (output / "reviewer.template.html").read_text(encoding="utf-8")
    payload = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c")
    for reviewer in REVIEWERS:
        _write_jsonl(output / f"{reviewer}.blank.jsonl", [blank_result(cid, reviewer) for cid in ids])
        page = template.replace("{{REVIEWER}}", reviewer).replace("{{CASES_JSON}}", payload)
        (output / f"{reviewer}.html").write_text(page, encoding="utf-8")
    report = validate_review_bundle(output)
    if zip_path is not None:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.write(path, f"validation_review/{path.relative_to(output).as_posix()}")
        report["zip"] = str(zip_path)
        report["zip_sha256"] = _sha(zip_path)
    return report


def validate_review_bundle(bundle: Path) -> dict:
    if any(path.is_symlink() for path in bundle.rglob("*")):
        raise ValueError("blind distributable must not contain symbolic links")
    manifest = json.loads((bundle / "case_manifest.json").read_text(encoding="utf-8"))
    ids = [row["case_id"] for row in manifest]
    _check_ids(ids)
    expected = STATIC_FILES | {"case_manifest.json", "media_provenance.json"}
    media_ids = {}
    for row in manifest:
        if set(row) != MEDIA_FIELDS or len(row["images"]) != 4:
            raise ValueError("blind manifest must contain only case IDs and six media paths")
        for value in [row["board"], row["input_preview"], *row["images"]]:
            relative = _relative(value).as_posix()
            if not relative.startswith(f"media/{row['case_id']}/") or relative in media_ids:
                raise ValueError("invalid or reused blind media path")
            media_ids[relative] = row["case_id"]
            expected.add(relative)
    provenance = json.loads((bundle / "media_provenance.json").read_text(encoding="utf-8"))
    if len(provenance) != 144 or {row["derived_path"] for row in provenance} != set(media_ids):
        raise ValueError("media provenance must cover every media file once")
    for row in provenance:
        if set(row) != {"case_id", "role", "source_path", "source_sha256", "crop", "derived_path", "derived_sha256"}:
            raise ValueError("media provenance contains unexpected fields")
        path = _relative(row["derived_path"])
        _relative(row["source_path"])
        if row["case_id"] != media_ids[path.as_posix()] or _sha(bundle / path) != row["derived_sha256"]:
            raise ValueError("media provenance or derived hash mismatch")
        if row["crop"] != (list(BOARD_CROP) if row["role"] == "board" else None):
            raise ValueError("unexpected title removal transform")
    template = (bundle / "reviewer.template.html").read_text(encoding="utf-8")
    payload = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c")
    for reviewer in REVIEWERS:
        expected.update({f"{reviewer}.blank.jsonl", f"{reviewer}.html"})
        if _jsonl(bundle / f"{reviewer}.blank.jsonl") != [blank_result(cid, reviewer) for cid in ids]:
            raise ValueError("reviewer templates must remain blank and independent")
        page = (bundle / f"{reviewer}.html").read_text(encoding="utf-8")
        expected_page = template.replace("{{REVIEWER}}", reviewer).replace("{{CASES_JSON}}", payload)
        if page != expected_page or f'data-reviewer="{reviewer}"' not in page or "{{" in page:
            raise ValueError("reviewer page differs from its identity or case manifest")
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError(f"unexpected distributable inventory: {sorted(actual ^ expected)}")
    return dict(cases=24, objects=10, media_files=144, reviewers=list(REVIEWERS), files=len(actual))


def _returned_review(path: Path, reviewer: str, ids: set[str]) -> dict[str, dict]:
    rows = _jsonl(path)
    by_id = {row["case_id"]: row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != ids:
        raise ValueError("returned review must cover the fixed IDs once each")
    for row in rows:
        if set(row) != RESULT_FIELDS or row["reviewer"] != reviewer:
            raise ValueError(f"expected an independent return by {reviewer}")
        if row["review_status"] != "completed":
            raise ValueError("pending or blank review cannot be merged")
        verdict, defect = row["verdict"], row["defect_type"]
        allowed = {"normal": {"none"}, "defect": {"direction", "axis", "offset", "other"}, "uncertain": {"uncertain"}}
        if verdict not in allowed or defect not in allowed[verdict]:
            raise ValueError("invalid verdict or defect_type")
        if any(type(row[key]) is not bool for key in DECISIONS[2:]) or not isinstance(row["notes"], str):
            raise ValueError("quality responses must be boolean and notes must be text")
        try:
            timestamp = datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
            if timestamp.utcoffset() is None:
                raise ValueError("timezone required")
        except (TypeError, AttributeError, ValueError) as exc:
            raise ValueError("reviewed_at must be an ISO datetime with timezone") from exc
    return by_id


def merge_reviews(bundle: Path, first: Path, second: Path, output: Path) -> dict:
    validate_review_bundle(bundle)
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("returned answers and reports must stay outside the distributable")
    for name in ("report.json", "disagreement.jsonl"):
        target = output / name
        if any(target.resolve() == path.resolve() or
               (target.exists() and target.samefile(path)) for path in (first, second)):
            raise ValueError("report output would overwrite an independent return")
    ids = {row["case_id"] for row in json.loads((bundle / "case_manifest.json").read_text())}
    left = _returned_review(first, REVIEWERS[0], ids)
    right = _returned_review(second, REVIEWERS[1], ids)
    disagreements, needs_followup = [], []
    for case_id in sorted(ids):
        a, b = left[case_id], right[case_id]
        differing = [field for field in DECISIONS if a[field] != b[field]]
        if differing:
            disagreements.append(dict(case_id=case_id, differing_fields=differing,
                                      reviews={REVIEWERS[0]: a, REVIEWERS[1]: b}))
        if any(r["verdict"] == "uncertain" or not r["one_glance"] or not r["no_clipping_or_occlusion"] for r in (a, b)):
            needs_followup.append(case_id)
    report = dict(cases=24, reviewers=list(REVIEWERS), agreement_cases=24-len(disagreements),
                  disagreement_cases=len(disagreements), needs_followup=needs_followup,
                  returned_sha256={REVIEWERS[0]: _sha(first), REVIEWERS[1]: _sha(second)},
                  policy="Independent returns only; reconcile after both returns. No ledger update or automatic approval.")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "report.json", report)
    _write_jsonl(output / "disagreement.jsonl", disagreements)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--root", type=Path, required=True, help="已解压的数据附件目录，包含 splits.json 和 review_manifest.jsonl")
    prepare.add_argument("--output", type=Path, required=True, help="独立盲审目录，不能放入原始数据目录")
    prepare.add_argument("--zip", type=Path)
    validate = sub.add_parser("validate")
    validate.add_argument("--bundle", type=Path, required=True, help="已解压的独立盲审附件目录")
    merge = sub.add_parser("merge")
    merge.add_argument("--bundle", type=Path, required=True, help="已解压的独立盲审附件目录")
    merge.add_argument("--first", type=Path, required=True, help="jwj1342 returned JSONL")
    merge.add_argument("--second", type=Path, required=True, help="WinterCCC returned JSONL")
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare_review(args.root, args.output, args.zip)
    elif args.command == "validate":
        report = validate_review_bundle(args.bundle)
    else:
        report = merge_reviews(args.bundle, args.first, args.second, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
