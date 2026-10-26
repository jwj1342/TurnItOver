"""Synthetic tests for the Medium reviewer/coordinator boundary."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.prepare_medium_review import prepare_review, validate_review_bundle, validate_return
from scripts.medium_bundle_inventory import write_manifest


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    rows = []
    for cid in ['source-clean-1', 'source-axis-2']:
        rows.append(dict(case_id=cid, expected_status='secret-answer', defect_id='secret-kind',
                         board='secret-answer-board.png'))
        for role in ['reference', 'candidate']:
            for state in ['rest', 'mid', 'limit']:
                path = root / f'native/{cid}/{role}_{state}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (256, 256), (30, 50, 70)).save(path)
    (root / 'review_manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (root / 'secret-answer-board.png').write_bytes(b'not-for-reviewers')
    return root


def prepared(source, tmp_path):
    output = tmp_path / 'review'
    mapping = prepare_review(source, output, 'synthetic-test-v1')
    return output, mapping


def test_review_contains_only_opaque_original_native_images(source, tmp_path):
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    output, mapping = prepared(source, tmp_path)
    report = validate_review_bundle(output)
    assert report['cases'] == 2 and report['media_files'] == 12 and report['blank_returns'] == 2
    for path in output.rglob('*'):
        if path.is_file() and path.suffix != '.png':
            text = path.read_text()
            assert not any(token in text for token in ['source-clean-1', 'source-axis-2', 'secret-answer', 'secret-kind'])
    for row in mapping:
        for role in ['reference', 'candidate']:
            for state in ['rest', 'mid', 'limit']:
                assert (output / f"native/{row['case_id']}/{role}_{state}.png").read_bytes() == (
                    source / f"native/{row['source_case_id']}/{role}_{state}.png").read_bytes()
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}


def test_extra_answer_file_rejected_even_if_inventory_resealed(source, tmp_path):
    output, _ = prepared(source, tmp_path)
    (output / 'gold.json').write_text('{"answer":"axis"}')
    write_manifest(output)
    with pytest.raises(ValueError, match='unexpected or missing'):
        validate_review_bundle(output)


@pytest.mark.parametrize('reseal', [False, True])
@pytest.mark.parametrize('relative', ['private/bundle_manifest.json', 'native/hidden/bundle_manifest.json'])
def test_nested_manifest_cannot_hide_answers(source, tmp_path, reseal, relative):
    output, _ = prepared(source, tmp_path)
    assert validate_review_bundle(output)['cases'] == 2
    hidden = output / relative
    hidden.parent.mkdir(parents=True, exist_ok=True)
    hidden.write_text('{"secret_gold":"axis"}')
    if reseal:
        files = write_manifest(output)
        assert relative in files
    with pytest.raises(ValueError):
        validate_review_bundle(output)


@pytest.mark.parametrize('attack', ['answer_field', 'filled_blank', 'html_script', 'javascript'])
def test_resealed_answer_and_template_tampering_rejected(source, tmp_path, attack):
    output, _ = prepared(source, tmp_path)
    if attack == 'answer_field':
        path = output / 'case_manifest.json'
        cases = json.loads(path.read_text());cases[0]['expected_status'] = 'fail'
        path.write_text(json.dumps(cases))
    elif attack == 'filled_blank':
        path = output / 'blank_return.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['decision'] = 'medium'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    else:
        path = output / ('review.html' if attack == 'html_script' else 'review.js')
        path.write_text(path.read_text() + '\nsecret-answer')
    write_manifest(output)
    with pytest.raises(ValueError):
        validate_review_bundle(output)


def test_missing_frame_refuses_package_without_partial_output(source, tmp_path):
    next((source / 'native').rglob('candidate_mid.png')).unlink()
    with pytest.raises(ValueError, match='missing, symbolic or invalid'):
        prepare_review(source, tmp_path / 'review', 'synthetic-test')
    assert not (tmp_path / 'review').exists()


def test_tampered_frame_fails_manifest(source, tmp_path):
    output, _ = prepared(source, tmp_path)
    next((output / 'native').rglob('candidate_mid.png')).write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        validate_review_bundle(output)


def test_native_size_change_and_source_path_injection_rejected(source, tmp_path):
    target = next((source / 'native').rglob('candidate_mid.png'))
    Image.new('RGB', (257, 256)).save(target)
    with pytest.raises(ValueError, match='invalid native source'):
        prepare_review(source, tmp_path / 'wrong-size', 'synthetic-test')
    (source / 'review_manifest.jsonl').write_text('{"case_id":"../outside"}\n')
    with pytest.raises(ValueError, match='source review IDs'):
        prepare_review(source, tmp_path / 'injected', 'synthetic-test')


def test_symbolic_source_directory_and_dangling_destination_refused(source, tmp_path):
    linked = tmp_path / 'linked'
    linked.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match='symbolic'):
        prepare_review(linked, tmp_path / 'review', 'synthetic-test')
    dangling = tmp_path / 'dangling'
    dangling.symlink_to(tmp_path / 'new-target', target_is_directory=True)
    with pytest.raises(ValueError, match='symbolic'):
        prepare_review(source, dangling, 'synthetic-test')
    assert not (tmp_path / 'new-target').exists()


def test_blank_response_is_pending_and_mixed_or_unattributed_answers_fail(source, tmp_path):
    output, _ = prepared(source, tmp_path)
    path = tmp_path / 'real-return.jsonl'
    path.write_bytes((output / 'blank_return.jsonl').read_bytes())
    assert validate_return(output, path) == dict(cases=2, completed=0, pending=2)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(decision='medium', reviewed_at='2026-10-25T00:00:00Z')
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='requires a reviewer'):
        validate_return(output, path)
    rows[0]['reviewer'] = 'synthetic-test-reviewer'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert validate_return(output, path) == dict(cases=2, completed=1, pending=1)
    rows[1]['reviewer'] = 'another-test-reviewer'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='one reviewer'):
        validate_return(output, path)


def test_inventory_cannot_carry_extra_answer_fields(source, tmp_path):
    output, _ = prepared(source, tmp_path)
    path = output / 'bundle_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['gold_answers'] = {'example': 'axis'}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    with pytest.raises(ValueError, match='Invalid Medium bundle inventory'):
        validate_review_bundle(output)


def test_inventory_cannot_hide_answers_in_shadowed_duplicate_fields(source, tmp_path):
    output, _ = prepared(source, tmp_path)
    path = output / 'bundle_manifest.json'
    manifest = json.loads(path.read_text())
    # A normal JSON parser discards the first "files" value, but a reviewer can read it.
    path.write_text('{"files":{"secret_gold":"axis"},"version":1,"files":'
                    + json.dumps(manifest['files']) + '}')
    with pytest.raises(ValueError, match='canonical encoding'):
        validate_review_bundle(output)


@pytest.mark.parametrize('file,old,new', [
    ('case_manifest.json', '"case_id":', '"case_id":"secret_gold_axis", "case_id":'),
    ('blank_return.jsonl', '"notes":', '"notes":"secret_gold_axis", "notes":'),
])
def test_review_metadata_cannot_hide_shadowed_answers(source, tmp_path, file, old, new):
    output, _ = prepared(source, tmp_path)
    path = output / file
    path.write_text(path.read_text().replace(old, new, 1))
    write_manifest(output)
    with pytest.raises(ValueError):
        validate_review_bundle(output)


def test_real_return_rejects_shadowed_judgment_fields(source, tmp_path):
    output, _ = prepared(source, tmp_path)
    returned = tmp_path / 'real-return.jsonl'
    returned.write_text((output / 'blank_return.jsonl').read_text().replace(
        '"decision": null', '"decision":"medium", "decision":null', 1))
    with pytest.raises(ValueError, match='duplicate JSON field'):
        validate_return(output, returned)
