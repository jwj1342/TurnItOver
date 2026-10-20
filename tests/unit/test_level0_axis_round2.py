"""Round-two source location, integrity and immutable approved-v3 contracts."""
import json
from pathlib import Path

import pytest

from scripts.prepare_level0_axis_v2 import DEFAULT_OUT, read_rows
from scripts.prepare_obvious_defect_review import sha256_path

V3 = DEFAULT_OUT.with_name('partnet-mobility-level0-axis-v3')
V4 = DEFAULT_OUT.with_name('partnet-mobility-level0-axis-v4')


@pytest.fixture
def source_pool(tmp_path, monkeypatch):
    from scripts import level0_expansion_sources as sources
    converted = tmp_path / 'converted'
    downloaded = tmp_path / 'downloaded'
    converted.mkdir()
    downloaded.mkdir()
    clean = next(g for g in read_rows(V3 / 'private/gold.jsonl') if g['kind'] == 'clean')
    spec = json.loads((V3 / 'private' / f"{clean['case_id']}.json").read_text())
    spec['asset_id'] = 'PM_fixture'
    spec_path = converted / 'PM_fixture.json'
    spec_path.write_text(json.dumps(spec))
    archive = downloaded / 'fixture.zip'
    archive.write_bytes(b'SYNTHETIC source archive for SHA-path testing only')
    splits_path = tmp_path / 'splits.json'
    splits_path.write_text(json.dumps({'groups': {'PartNet-Mobility:validation': 'validation'}}))
    monkeypatch.setattr(sources, 'DEFAULT_SPLITS', splits_path)
    candidate = {'asset_id': 'fixture', 'source_identity': 'PartNet-Mobility:fixture',
                 'status': 'structural_candidate', 'frozen_test_overlap': False,
                 'spec_sha256': sha256_path(spec_path), 'archive_sha256': sha256_path(archive),
                 'model_id': 'fixture-model', 'scale_audit': {'geometry_fingerprint': 'fixture-geometry'},
                 'annotation_audit': {}}
    conversion_path = converted / 'conversion.json'
    conversion_path.write_text(json.dumps({'dataset_revision': 'fixture-revision', 'assets': [
        candidate, {**candidate, 'asset_id': 'validation', 'source_identity': 'PartNet-Mobility:validation'},
        {**candidate, 'asset_id': 'test', 'frozen_test_overlap': True}]}))
    catalog_path = converted / 'catalog.json'
    catalog_path.write_text(json.dumps({'assets': [{'spec': 'PM_fixture.json', 'custom_provenance': 'retained'}]}))
    download_path = downloaded / 'download-manifest.json'
    download_path.write_text(json.dumps({'dataset_revision': 'fixture-revision', 'dataset': 'fixture-dataset',
        'license': 'synthetic test only', 'archives': [{'asset_id': 'fixture', 'path': 'fixture.zip'}]}))
    return sources, catalog_path, download_path, spec_path, archive


def test_explicit_pool_paths_keep_source_identity_and_exclude_protected_splits(source_pool):
    sources, catalog, download, spec, archive = source_pool
    pool, splits, pins = sources.load_pool(catalog_path=catalog, download_path=download)
    assert [r['spec'].asset_id for r in pool] == ['PM_fixture']
    meta = pool[0]['metadata']
    assert meta['source_urdf'] == str(download.parent / 'extracted/fixture/mobility.urdf')
    assert meta['archive_sha256'] == sha256_path(archive)
    assert meta['original_spec_sha256'] == sha256_path(spec)
    assert meta['source_dataset'] == 'fixture-dataset'
    assert meta['dataset_revision'] == 'fixture-revision'
    assert meta['source_identity'] == 'PartNet-Mobility:fixture'
    assert meta['custom_provenance'] == 'retained'
    assert pool[0]['original_train'] is False
    assert splits['groups'] == {'PartNet-Mobility:validation': 'validation'}
    assert set(pins) == {str(catalog), str(catalog.parent / 'conversion.json'), str(download), str(sources.DEFAULT_SPLITS)}
    assert all(sha256_path(Path(path)) == digest for path, digest in pins.items())


@pytest.mark.parametrize('tampered', ['spec', 'archive', 'revision'])
def test_explicit_pool_rejects_tampered_inputs(source_pool, tampered):
    sources, catalog, download, spec, archive = source_pool
    if tampered == 'revision':
        manifest = json.loads(download.read_text())
        manifest['dataset_revision'] = 'wrong-revision'
        download.write_text(json.dumps(manifest))
        with pytest.raises(AssertionError):
            sources.load_pool(catalog_path=catalog, download_path=download)
    else:
        (spec if tampered == 'spec' else archive).write_bytes(b'tampered')
        with pytest.raises(ValueError, match='Spec SHA mismatch' if tampered == 'spec' else 'Archive SHA mismatch'):
            sources.load_pool(catalog_path=catalog, download_path=download)


def test_v4_preserves_approved_v3_and_exports_only_real_reviewed_pairs():
    from scripts.finalize_level0_axis_v2 import accepted
    baseline = json.loads((V3 / 'manifest.json').read_text())
    manifest = json.loads((V4 / 'manifest.json').read_text())
    assert manifest['base_manifest_sha256'] == sha256_path(V3 / 'manifest.json')
    for name, digest in baseline['files'].items():
        assert sha256_path(V3 / name) == digest
        if name.startswith(('boards/', 'inputs/', 'frames/', 'private/l0-')):
            assert sha256_path(V4 / name) == digest
    for name, digest in manifest['files'].items():
        assert sha256_path(V4 / name) == digest
    for name, digest in manifest['inputs'].items():
        assert sha256_path(Path(name)) == digest
    for relative in ('private/gold.jsonl', 'human_review.jsonl', 'private/checks.jsonl'):
        inherited = read_rows(V3 / relative)
        current = {r['case_id']: r for r in read_rows(V4 / relative)}
        assert all(current[r['case_id']] == r for r in inherited)
    prior_ids = {r['case_id'] for r in read_rows(V3 / 'human_review.jsonl')}
    gold = read_rows(V4 / 'private/gold.jsonl')
    reviews = {r['case_id']: r for r in read_rows(V4 / 'human_review.jsonl')}
    inherited = json.loads((V4 / 'private/inherited_review_provenance.json').read_text())
    export = read_rows(V4 / 'train_manifest.jsonl')
    expected = set()
    for asset in {g['asset_id'] for g in gold}:
        pair = [g for g in gold if g['asset_id'] == asset]
        assert len(pair) == 2
        assert {g['kind'] for g in pair} == {'clean', 'joint_axis'}
        if all(accepted(reviews[g['case_id']], g, g['case_id'] in inherited) for g in pair):
            expected.update(g['case_id'] for g in pair)
    assert {r['case_id'] for r in export} == expected
    for relative in ('train_manifest.jsonl', 'private/train_targets.jsonl'):
        old = read_rows(V3 / relative)
        current = {r['case_id']: r for r in read_rows(V4 / relative)}
        assert len(old) == 76
        assert all(current[r['case_id']] == r for r in old)
    new = [r for r in reviews.values() if r['case_id'] not in prior_ids]
    assert new
    for review in new:
        if review['status'] == 'pending_human_review':
            assert review['reviewer'] is None
            assert review['reviewed_at'] is None
            assert review['human_judgment']['verdict'] is None
            assert review['case_id'] not in expected
    if all(r['status'] == 'pending_human_review' for r in new):
        assert len(export) == 76
    old_splits = json.loads((V3 / baseline['expanded_splits']).read_text())['groups']
    splits = json.loads((V4 / manifest['expanded_splits']).read_text())['groups']
    assert all(splits[k] == v for k, v in old_splits.items())
    assert all(v == 'train' for k, v in splits.items() if k not in old_splits)
