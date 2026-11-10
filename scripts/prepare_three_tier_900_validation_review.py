"""Build and seal the anonymous review pack for the three-tier 900 validation split."""
from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

from PIL import Image

from scripts.package_tier_candidates import seal
from scripts.prepare_tier_review import prepare, validate


EXPECTED_COUNTS = {"easy": 40, "medium": 40, "hard": 40}
EPISODES_SUFFIX = "episode_export/validation.episodes.jsonl"
SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")
IMAGE_NAME = re.compile(
    r"^\d{2}-(?:(?P<view>.+)_)?(?P<role>reference|candidate)_"
    r"(?P<state>rest|mid|limit)\.png$"
)
FORBIDDEN_TEXT = (
    '"data_identity"',
    '"difficulty"',
    '"messages"',
    '"source_asset_id"',
    '"source_case_id"',
)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_release_sha(manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text())
    return manifest["artifacts"]["private_candidate_bundle"]["sha256"]


def safe_archive_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name}")


def read_rows(archive: zipfile.ZipFile) -> tuple[str, list[dict]]:
    matches = [name for name in archive.namelist() if name.endswith(EPISODES_SUFFIX)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {EPISODES_SUFFIX}, found {len(matches)}")
    member = matches[0]
    prefix = member[: -len("validation.episodes.jsonl")]
    rows = [json.loads(line) for line in archive.read(member).decode().splitlines() if line.strip()]
    counts = Counter(row.get("difficulty") for row in rows)
    if counts != Counter(EXPECTED_COUNTS):
        raise ValueError(f"Unexpected validation counts: {dict(counts)}")
    episode_ids = [row.get("episode_id") for row in rows]
    if len(rows) != 120 or len(set(episode_ids)) != 120:
        raise ValueError("Validation must contain exactly 120 unique episode IDs")
    if any(not isinstance(value, str) or not SAFE_ID.fullmatch(value) for value in episode_ids):
        raise ValueError("Unsafe validation episode ID")
    return prefix, rows


def parse_image_name(filename: str) -> tuple[str | None, str, str]:
    match = IMAGE_NAME.fullmatch(filename)
    if not match:
        raise ValueError(f"Unexpected validation image name: {filename}")
    return match.group("view"), match.group("role"), match.group("state")


def verify_png(data: bytes, member: str) -> None:
    with Image.open(BytesIO(data)) as image:
        if image.format != "PNG" or image.size != (256, 256):
            raise ValueError(f"Review evidence must be native 256x256 PNG: {member}")
        image.verify()


def build_intake(
    archive: zipfile.ZipFile,
    prefix: str,
    rows: list[dict],
    evidence_root: Path,
    bundle_identity: str,
) -> tuple[list[dict], int]:
    archive_names = set(archive.namelist())
    seen_members: set[str] = set()
    intake: list[dict] = []
    total_images = 0
    for row in sorted(rows, key=lambda item: item["episode_id"]):
        episode_id = row["episode_id"]
        images = row.get("images")
        hashes = row.get("image_sha256")
        expected_by_tier = {"easy": 4, "medium": 6, "hard": 12}[row["difficulty"]]
        if not isinstance(images, list) or not isinstance(hashes, list):
            raise ValueError(f"Missing image inventory for {episode_id}")
        if len(images) != expected_by_tier or len(images) != len(hashes):
            raise ValueError(f"Unexpected image inventory for {episode_id}")

        parsed = [parse_image_name(PurePosixPath(name).name) for name in images]
        views: list[str | None] = []
        for view, _, _ in parsed:
            if view not in views:
                views.append(view)
        expected_views = 2 if row["difficulty"] == "hard" else 1
        if len(views) != expected_views:
            raise ValueError(f"Unexpected view layout for {episode_id}: {views}")

        item = {
            "source_case_id": episode_id,
            "bundle": bundle_identity,
            "primary": [],
            "supplementary": [],
        }
        for image_ref, expected_hash, parsed_name in zip(images, hashes, parsed, strict=True):
            relative = PurePosixPath(image_ref)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "images":
                raise ValueError(f"Unsafe image reference for {episode_id}: {image_ref}")
            member = prefix + image_ref
            safe_archive_name(member)
            if member not in archive_names or member in seen_members:
                raise ValueError(f"Missing or duplicate validation image: {member}")
            seen_members.add(member)
            data = archive.read(member)
            if digest_bytes(data) != expected_hash:
                raise ValueError(f"Image hash mismatch: {member}")
            verify_png(data, member)

            view, role, state = parsed_name
            group = "primary" if view == views[0] else "supplementary"
            target = evidence_root / episode_id / relative.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            caption = (
                f"{'参考' if role == 'reference' else '候选'} / "
                f"{'初始视角' if group == 'primary' else '补充视角'} / {state}"
            )
            item[group].append({"path": str(target), "caption": caption})
            total_images += 1
        intake.append(item)
    if total_images != 880 or len(seen_members) != 880:
        raise ValueError(f"Expected 880 unique validation images, found {total_images}")
    return intake, total_images


def assert_answer_free(pack: Path, source_ids: set[str]) -> None:
    for path in pack.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".html", ".js", ".json", ".md"}:
            continue
        text = path.read_text()
        for token in FORBIDDEN_TEXT:
            if token in text:
                raise ValueError(f"Private field leaked into {path.name}: {token}")
        for source_id in source_ids:
            if source_id in text:
                raise ValueError(f"Source episode ID leaked into {path.name}")


def verify_archive(output: Path, source_ids: set[str]) -> None:
    archive_path = output / "review.zip"
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Sealed review ZIP failed CRC validation")
        for name in archive.namelist():
            safe_archive_name(name)
            lowered = name.lower()
            if any(token in lowered for token in ("mapping.json", "intake.json", "validation.episodes")):
                raise ValueError(f"Private file leaked into review ZIP: {name}")
            if Path(name).suffix.lower() in {".html", ".js", ".json", ".md"}:
                text = archive.read(name).decode()
                if any(token in text for token in FORBIDDEN_TEXT):
                    raise ValueError(f"Private field leaked into review ZIP: {name}")
                if any(source_id in text for source_id in source_ids):
                    raise ValueError(f"Source episode ID leaked into review ZIP: {name}")


def build(
    candidate_zip: Path,
    release_manifest: Path,
    output: Path,
    custody: Path,
    namespace: str,
) -> dict:
    candidate_zip = candidate_zip.resolve(strict=True)
    release_manifest = release_manifest.resolve(strict=True)
    output = output.resolve()
    custody = custody.resolve()
    if output.exists() or custody.exists() or output == custody:
        raise ValueError("Output and custody paths must be new and separate")
    expected_sha = load_release_sha(release_manifest)
    actual_sha = digest_file(candidate_zip)
    if actual_sha != expected_sha:
        raise ValueError(f"Candidate ZIP SHA256 mismatch: {actual_sha}")

    output.parent.mkdir(parents=True, exist_ok=True)
    custody.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="three-tier-900-validation-", dir=output.parent) as temporary:
            temporary_root = Path(temporary)
            evidence_root = temporary_root / "evidence"
            intake_path = temporary_root / "intake.json"
            pack = temporary_root / "three-tier-900-validation-v1"
            with zipfile.ZipFile(candidate_zip) as archive:
                if archive.testzip() is not None:
                    raise ValueError("Candidate ZIP failed CRC validation")
                prefix, rows = read_rows(archive)
                intake, image_count = build_intake(
                    archive, prefix, rows, evidence_root, str(candidate_zip)
                )
            intake_path.write_text(json.dumps(intake, ensure_ascii=False, indent=2) + "\n")
            source_ids = {row["episode_id"] for row in rows}
            prepared = prepare(intake_path, pack, custody, namespace)
            verified = validate(pack)
            if prepared["cases"] != 120 or verified["cases"] != 120:
                raise ValueError("Generated review package does not contain 120 cases")
            assert_answer_free(pack, source_ids)
            sealed = seal([pack], output, review=True)
            verify_archive(output, source_ids)
        return {
            "archive": str(output / "review.zip"),
            "archive_sha256": sealed["sha256"],
            "cases": 120,
            "difficulty": EXPECTED_COUNTS,
            "images": image_count,
            "package_sha256": prepared["package_sha256"],
            "source_sha256": actual_sha,
            "status": "review_pack_verified",
        }
    except BaseException:
        if output.exists():
            shutil.rmtree(output)
        if custody.exists():
            shutil.rmtree(custody)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-zip", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--custody", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
