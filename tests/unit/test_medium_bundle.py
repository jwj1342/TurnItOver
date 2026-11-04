"""Portable Medium release failures are checked using synthetic media only."""
from __future__ import annotations

import json
import shutil

import pytest

from tests.unit.medium_bundle_fixture import build_bundle, png, read_rows, write_json, write_rows
from scripts.easy_release_archive import sha256
from scripts.medium_bundle_history import validate_history
from scripts.medium_bundle_inventory import write_manifest
from scripts.prepare_medium_review import write_manifest as write_review_manifest
from scripts.validate_medium_bundle import validate_bundle


@pytest.fixture(scope='module')
def sealed(tmp_path_factory):
    return build_bundle(tmp_path_factory.mktemp('synthetic-medium') / 'medium')


@pytest.fixture
def candidate(sealed, tmp_path):
    destination = tmp_path / 'medium'
    shutil.copytree(sealed, destination)
    return destination


def test_portable_candidate_is_not_program_replay_or_training(sealed):
    report = validate_bundle(sealed)
    assert (report['objects'], report['cases'], report['native_images']) == (75, 150, 900)
    assert report['human_visibility'] == dict(reviewed=16, medium=11, clean=2, too_obvious=3,
                                             active_pending_visibility=64, full_diagnostic_complete=0)
    assert report['training_eligible_cases'] == 0
    assert report['program_replay_performed'] is False
    assert report['validation_objects'] == 0
    assert not (sealed / 'private/programs').exists()


def test_nested_review_manifest_is_signed_by_candidate_inventory(candidate):
    path = candidate / 'review_bundle/bundle_manifest.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        validate_bundle(candidate)


def test_actual_media_sha_cannot_be_replaced_by_resealing(candidate):
    row = read_rows(candidate / 'review_manifest.jsonl')[0]
    # A review board is valid PNG but no longer the sealed source image.
    (candidate / row['board']).write_bytes(png(110))
    write_manifest(candidate)
    with pytest.raises(ValueError, match='differs from the sealed source'):
        validate_bundle(candidate)


def test_source_alias_duplicate_is_rejected(candidate):
    rows = read_rows(candidate / 'private/gold.jsonl')
    for row in rows[2:4]:
        row['source']['model_id'] = rows[0]['source']['model_id']
    write_rows(candidate / 'private/gold.jsonl', rows)
    write_manifest(candidate)
    with pytest.raises(ValueError, match='Duplicate or missing independent source'):
        validate_bundle(candidate)


def test_checker_flags_cannot_turn_a_clean_case_into_a_defect(candidate):
    rows = read_rows(candidate / 'private/checks.jsonl')
    rows[0]['code_correct'] = False
    write_rows(candidate / 'private/checks.jsonl', rows)
    write_manifest(candidate)
    with pytest.raises(ValueError, match='checker evidence'):
        validate_bundle(candidate)


def test_full_diagnostic_completion_is_not_inferred_from_visibility(candidate):
    rows = read_rows(candidate / 'human_review.jsonl')
    rows[0].update(status='human_review_complete', reviewer='human')
    write_rows(candidate / 'human_review.jsonl', rows)
    write_manifest(candidate)
    with pytest.raises(ValueError, match='unfilled pending template'):
        validate_bundle(candidate)


def test_training_export_is_rejected_even_after_local_reseal(candidate):
    (candidate / 'train_manifest.jsonl').write_text('{"case_id":"not-qualified"}\n')
    write_manifest(candidate)
    with pytest.raises(ValueError, match='training exports'):
        validate_bundle(candidate)


def test_full_source_programs_are_not_distributable_candidate_files(candidate):
    folder = candidate / 'private/programs'
    folder.mkdir()
    (folder / 'arbitrary.json').write_text('{"mesh":"not-distributable"}')
    write_manifest(candidate)
    with pytest.raises(ValueError, match='construction programs remain in local custody'):
        validate_bundle(candidate)


def test_anonymous_frame_must_match_its_coordinator_mapping(candidate):
    folder = candidate / 'review_bundle'
    card = json.loads((folder / 'case_manifest.json').read_text())[0]
    (folder / card['candidate']['limit']).write_bytes(png(170))
    write_review_manifest(folder)
    write_manifest(candidate)
    with pytest.raises(ValueError, match='different source case'):
        validate_bundle(candidate)


def test_mapping_cannot_be_included_in_the_reviewer_package(candidate):
    folder = candidate / 'review_bundle'
    shutil.copyfile(candidate / 'private/review_mapping.jsonl', folder / 'mapping.jsonl')
    write_review_manifest(folder)
    write_manifest(candidate)
    with pytest.raises(ValueError, match='unexpected or missing files'):
        validate_bundle(candidate)


def test_original_human_feedback_is_bound_to_its_recorded_raw_json(candidate):
    folder = candidate / 'private/human_review_history/round-0001'
    raw = json.loads((folder / 'raw.json').read_text())
    raw[0]['notes'] = 'A forged replacement for the real observation'
    write_json(folder / 'raw.json', raw)
    meta = json.loads((folder / 'import.json').read_text())
    meta['raw_sha256'] = sha256(folder / 'raw.json')
    write_json(folder / 'import.json', meta)
    gold = {r['case_id']: r for r in read_rows(candidate / 'private/gold.jsonl')}
    with pytest.raises(ValueError, match='Raw judgments'):
        validate_history(candidate, gold)
