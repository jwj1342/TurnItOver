import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_easy_validation_review import (
    BOARD_CROP,
    REVIEWERS,
    SELECTION_PATH,
    blank_result,
    merge_reviews,
    prepare_review,
    validate_review_bundle,
)


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def bundle(tmp_path):
    # Deliberately provide no Gold, old human ledger, checker, or root package.
    source = tmp_path / "source"
    source.mkdir()
    objects = json.loads(SELECTION_PATH.read_text())["objects"]
    (source / "splits.json").write_text(json.dumps({"objects": objects}))
    rows = []
    for obj in objects:
        for cid in obj["case_ids"]:
            row = dict(case_id=cid, board=f"boards/{cid}.png", input_preview=f"inputs/{cid}.png",
                       images=[f"frames/{cid}/{i}.png" for i in range(4)],
                       status="OLD_REVIEW_SECRET", kind="GOLD_ANSWER_SECRET")
            for relative in [row["board"], row["input_preview"], *row["images"]]:
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (912, 630) if relative == row["board"] else (4, 4), "red")
                if relative == row["board"]:
                    image.paste("blue", (0, 60, 912, 630))
                image.save(path)
            rows.append(row)
    _jsonl(source / "review_manifest.jsonl", rows)
    output = tmp_path / "review"
    prepare_review(source, output)
    return source, output


def test_preparation_is_blind_fixed_and_independently_distributable(bundle, tmp_path):
    source, output = bundle
    report = validate_review_bundle(output)
    assert (report["cases"], report["objects"], report["media_files"]) == (24, 10, 144)
    manifest = json.loads((output / "case_manifest.json").read_text())
    for row in manifest:
        assert set(row) == {"case_id", "board", "input_preview", "images"}
        with Image.open(output / row["board"]) as image:
            assert image.size == (912, 570)
            assert image.getpixel((0, 0)) == (0, 0, 255)
    for path in output.glob("*"):
        if path.is_file():
            content = path.read_text()
            assert "OLD_REVIEW_SECRET" not in content
            assert "GOLD_ANSWER_SECRET" not in content
    provenance = json.loads((output / "media_provenance.json").read_text())
    assert all(row["crop"] == list(BOARD_CROP) for row in provenance if row["role"] == "board")
    for reviewer in REVIEWERS:
        rows = [json.loads(line) for line in (output / f"{reviewer}.blank.jsonl").read_text().splitlines()]
        assert rows == [blank_result(row["case_id"], reviewer) for row in manifest]
        assert f'data-reviewer="{reviewer}"' in (output / f"{reviewer}.html").read_text()
    archive = tmp_path / "review.zip"
    prepare_review(source, output, archive)
    with zipfile.ZipFile(archive) as zipped:
        assert len(zipped.namelist()) == report["files"]
        assert not any("gold" in name or "human_review" in name for name in zipped.namelist())


def _returns(bundle, tmp_path):
    manifest = json.loads((bundle / "case_manifest.json").read_text())
    rows, paths = [], []
    for reviewer in REVIEWERS:
        review = [dict(blank_result(row["case_id"], reviewer), review_status="completed",
                       verdict="normal", defect_type="none", one_glance=True,
                       no_clipping_or_occlusion=True, reviewed_at="2026-10-24T08:00:00+00:00")
                  for row in manifest]
        path = tmp_path / f"{reviewer}.jsonl"
        _jsonl(path, review)
        paths.append(path)
        rows.append(review)
    return paths, rows


def test_merge_reports_disagreement_without_updating_sources(bundle, tmp_path):
    source, output = bundle
    paths, rows = _returns(output, tmp_path)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    rows[1][0].update(verdict="defect", defect_type="axis")
    rows[0][1].update(verdict="uncertain", defect_type="uncertain")
    rows[1][1].update(verdict="uncertain", defect_type="uncertain")
    _jsonl(paths[0], rows[0])
    _jsonl(paths[1], rows[1])
    destination = tmp_path / "adjudication"
    report = merge_reviews(output, *paths, destination)
    assert report["disagreement_cases"] == 1
    assert report["agreement_cases"] == 23
    assert report["needs_followup"] == [rows[0][1]["case_id"]]
    differences = [json.loads(line) for line in (destination / "disagreement.jsonl").read_text().splitlines()]
    assert differences[0]["differing_fields"] == ["verdict", "defect_type"]
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    validate_review_bundle(output)


@pytest.mark.parametrize("mutation", ["pending", "duplicate", "missing", "same_reviewer", "verdict", "defect", "time", "quality"])
def test_merge_rejects_invalid_or_nonindependent_returns(bundle, tmp_path, mutation):
    _, output = bundle
    paths, rows = _returns(output, tmp_path)
    if mutation == "pending":
        rows[0][0] = blank_result(rows[0][0]["case_id"], REVIEWERS[0])
    elif mutation == "duplicate":
        rows[0][-1] = rows[0][0]
    elif mutation == "missing":
        rows[0].pop()
    elif mutation == "same_reviewer":
        rows[1][0]["reviewer"] = REVIEWERS[0]
    elif mutation == "verdict":
        rows[0][0]["verdict"] = "approved"
    elif mutation == "defect":
        rows[0][0]["defect_type"] = "axis"
    elif mutation == "time":
        rows[0][0]["reviewed_at"] = "2026-10-24T08:00:00"
    else:
        rows[0][0]["one_glance"] = "yes"
    for path, review in zip(paths, rows):
        _jsonl(path, review)
    with pytest.raises(ValueError):
        merge_reviews(output, *paths, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_preparation_rejects_changed_validation_ids(bundle, tmp_path):
    source, _ = bundle
    path = source / "splits.json"
    split = json.loads(path.read_text())
    split["objects"][0]["case_ids"][0] = "unrecognized-case"
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="fixed 24"):
        prepare_review(source, tmp_path / "changed")


def test_preparation_rejects_changed_object_assignment(bundle, tmp_path):
    source, _ = bundle
    path = source / "splits.json"
    split = json.loads(path.read_text())
    a, b = split["objects"][:2]
    a["case_ids"][0], b["case_ids"][0] = b["case_ids"][0], a["case_ids"][0]
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="object assignments"):
        prepare_review(source, tmp_path / "changed-objects")


def test_validation_rejects_page_with_stale_embedded_cases(bundle):
    _, output = bundle
    path = output / f"{REVIEWERS[0]}.html"
    page = path.read_text()
    changed, count = re.subn(r'(<script[^>]* id="case-data"[^>]*>).*?(</script>)',
                             r'\g<1>[]\2', page, flags=re.DOTALL)
    assert count == 1
    path.write_text(changed)
    with pytest.raises(ValueError, match="case manifest"):
        validate_review_bundle(output)


@pytest.mark.parametrize("filename", ["report.json", "disagreement.jsonl"])
def test_merge_preserves_return_when_output_path_collides(bundle, tmp_path, filename):
    _, output = bundle
    paths, _ = _returns(output, tmp_path)
    destination = tmp_path / "adjudication"
    destination.mkdir()
    collision = destination / filename
    paths[0].rename(collision)
    original = collision.read_bytes()
    with pytest.raises(ValueError, match="overwrite an independent return"):
        merge_reviews(output, collision, paths[1], destination)
    assert collision.read_bytes() == original


@pytest.mark.parametrize("command", ["prepare", "validate", "merge"])
def test_cli_requires_explicit_paths_without_reading_data(command, tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/prepare_easy_validation_review.py"
    result = subprocess.run([sys.executable, "-B", str(script), command], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "required" in result.stderr
    assert not list(tmp_path.iterdir())


def test_preparation_refuses_source_output_overlap(bundle):
    source, _ = bundle
    with pytest.raises(ValueError, match="must not overlap"):
        prepare_review(source, source / "review")
    assert not (source / "review").exists()
