import hashlib
import json

import pytest

from turnitover.repair.frozen_manifest import load_frozen_manifest


def make_manifest(tmp_path):
    fields = ('reference_image', 'initial_program', 'reference_program')
    row = {'id': 'case-1', 'source_group': 'source-1', 'sha256': {}}
    for field in fields:
        file = tmp_path / field
        file.write_text(field)
        row[field] = file.name
        row['sha256'][field] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest = {'schema_version': 1, 'status': 'frozen', 'split': 'validation', 'cases': [row]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_load_and_detect_changed_input(tmp_path):
    path, manifest = make_manifest(tmp_path)
    _, cases = load_frozen_manifest(path)
    assert cases[0]['initial_program'] == tmp_path / 'initial_program'
    (tmp_path / 'initial_program').write_text('changed')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_frozen_manifest(path)


def test_reject_split_and_duplicate_ids(tmp_path):
    path, manifest = make_manifest(tmp_path)
    manifest['split'] = 'test'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='validation'):
        load_frozen_manifest(path)
    manifest['split'] = 'validation'
    manifest['cases'].append(manifest['cases'][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unique'):
        load_frozen_manifest(path)


def test_reject_traversal_and_unsafe_id(tmp_path):
    path, manifest = make_manifest(tmp_path)
    manifest['cases'][0]['initial_program'] = '../outside'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Invalid or missing'):
        load_frozen_manifest(path)
    manifest['cases'][0]['id'] = '../case'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='safe ids'):
        load_frozen_manifest(path)
