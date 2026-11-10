import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from scripts.prepare_three_tier_900_validation_review import build


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (256, 256), "white").save(stream, format="PNG")
    return stream.getvalue()


def make_candidate(tmp_path: Path, corrupt_hash: bool = False) -> tuple[Path, Path]:
    candidate = tmp_path / "private-candidates.zip"
    data = png_bytes()
    rows = []
    with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for difficulty, image_count in (("easy", 4), ("medium", 6), ("hard", 12)):
            for index in range(40):
                episode_id = f"{difficulty}-{index:02d}"
                names = []
                if difficulty == "easy":
                    layouts = [(None, "reference", "rest"), (None, "reference", "limit"),
                               (None, "candidate", "rest"), (None, "candidate", "limit")]
                elif difficulty == "medium":
                    layouts = [(None, role, state) for role in ("reference", "candidate")
                               for state in ("rest", "mid", "limit")]
                else:
                    layouts = [(view, role, state) for view in ("front", "left")
                               for role in ("reference", "candidate")
                               for state in ("rest", "mid", "limit")]
                assert len(layouts) == image_count
                for number, (view, role, state) in enumerate(layouts):
                    middle = f"{view}_" if view else ""
                    filename = f"{number:02d}-{middle}{role}_{state}.png"
                    relative = f"images/{episode_id}/{filename}"
                    member = f"three-tier-900-sft-v1/episode_export/{relative}"
                    archive.writestr(member, data)
                    names.append(relative)
                hashes = [hashlib.sha256(data).hexdigest()] * image_count
                if corrupt_hash and difficulty == "easy" and index == 0:
                    hashes[0] = "0" * 64
                rows.append({
                    "difficulty": difficulty,
                    "episode_id": episode_id,
                    "images": names,
                    "image_sha256": hashes,
                    "messages": [{"role": "assistant", "content": "private answer"}],
                })
        archive.writestr(
            "three-tier-900-sft-v1/episode_export/validation.episodes.jsonl",
            "".join(json.dumps(row) + "\n" for row in rows),
        )
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(json.dumps({
        "artifacts": {"private_candidate_bundle": {
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()
        }}
    }))
    return candidate, manifest


def test_builds_verified_answer_free_review_archive(tmp_path):
    candidate, manifest = make_candidate(tmp_path)
    output = tmp_path / "sealed"
    custody = tmp_path / "custody"
    result = build(candidate, manifest, output, custody, "test-three-tier-900-validation-v1")
    assert result["cases"] == 120
    assert result["images"] == 880
    assert result["status"] == "review_pack_verified"
    assert (output / "review.zip").is_file()
    assert (output / "SHA256SUMS").is_file()
    with zipfile.ZipFile(output / "review.zip") as archive:
        names = archive.namelist()
        text = "\n".join(
            archive.read(name).decode()
            for name in names
            if Path(name).suffix in {".html", ".js", ".json", ".md"}
        )
    assert "easy-00" not in text
    assert "private answer" not in text
    assert '"difficulty"' not in text
    mapping = (custody / "mapping.json").read_text()
    assert "easy-00" in mapping
    assert len(list(custody.iterdir())) == 2


def test_rejects_image_hash_mismatch_without_partial_outputs(tmp_path):
    candidate, manifest = make_candidate(tmp_path, corrupt_hash=True)
    output = tmp_path / "sealed"
    custody = tmp_path / "custody"
    with pytest.raises(ValueError, match="Image hash mismatch"):
        build(candidate, manifest, output, custody, "test-three-tier-900-validation-v1")
    assert not output.exists()
    assert not custody.exists()
