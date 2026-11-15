from copy import deepcopy
import hashlib
import json

import pytest

from scripts.prepare_rp29_full_repair import refresh_scoring_scopes


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    import scripts.run_rp29_repair_batch as batch
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    report = tmp_path / 'report.json'
    report_sha = save(report, {'defect_id': 'geometry.missing_part',
                              'condition': {'target': 'part'}, 'historical_pass': True})
    label = {'defect_id': 'geometry.missing_part', 'parts': [], 'params': {}}
    annotations = dataset / 'annotations.private.jsonl'
    annotations_sha = save(annotations, {'id': 'case', 'split': 'validation', 'labels': [label],
        'source_roster_row': {'split': 'validation', 'report_path': str(report), 'report_sha256': report_sha},
        'programs': {'case_reference': {'sha256': 'a' * 64}, 'case_candidate': {'sha256': 'b' * 64}}})
    dataset_sha = save(dataset / 'manifest.json', {'files': {'annotations.private.jsonl': annotations_sha}})
    labels_path = tmp_path / 'labels.private.jsonl'
    labels_sha = save(labels_path, {'id': 'case', 'labels': [label], 'scoring_scopes': [None],
        'history': [{'original': 'unchanged'}], 'evidence': {'original': 'unchanged'}})
    metadata = {'private_labels': labels_path.name, 'private_labels_sha256': labels_sha,
                'source_manifest_sha256': dataset_sha, 'cases': [{'id': 'case'}]}
    source = tmp_path / 'manifest.json'
    save(source, metadata)
    case = {'id': 'case', 'sha256': {'reference_program': 'a' * 64, 'initial_program': 'b' * 64}}
    monkeypatch.setattr(batch, 'load_cases', lambda path: (deepcopy(metadata), [(source, deepcopy(case))]))
    return source, dataset, tmp_path / 'reviewed.json', labels_path


def test_refresh_writes_separate_hash_bound_revision_and_preserves_inputs(frozen):
    source, dataset, output, old_labels = frozen
    original = source.read_bytes(), old_labels.read_bytes()
    result = refresh_scoring_scopes(source, dataset, output)
    assert (source.read_bytes(), old_labels.read_bytes()) == original
    row = json.loads((output.parent / result['private_labels']).read_text())
    assert row['labels'][0]['parts'] == ['part']
    assert row['history'] == [{'original': 'unchanged'}]
    assert row['evidence'] == {'original': 'unchanged'}
    assert 'historical_pass' not in json.dumps(row)
    assert result['scoring_revision_of_sha256'] == hashlib.sha256(original[0]).hexdigest()
    with pytest.raises(FileExistsError):
        refresh_scoring_scopes(source, dataset, output)


def test_refresh_rejects_modified_annotations_before_writing(frozen):
    source, dataset, output, _ = frozen
    (dataset / 'annotations.private.jsonl').write_text('{}')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        refresh_scoring_scopes(source, dataset, output)
    assert not output.exists()


def test_refresh_cannot_overwrite_original_manifest(frozen):
    source, dataset, _, _ = frozen
    with pytest.raises(ValueError, match='new sibling'):
        refresh_scoring_scopes(source, dataset, source)
