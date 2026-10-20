"""CPU regression tests for immutable v2 continuation and normal-only cameras."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.prepare_level0_axis_v2 import DEFAULT_OUT, read_rows
from scripts.prepare_obvious_defect_review import sha256_path


def _mock_search(monkeypatch, *, pass_seed=None, outside=False):
    from scripts import level0_expansion_search as search
    from turnitover.core.sample import Framing

    reference = SimpleNamespace(spec=SimpleNamespace(joints=[
        SimpleNamespace(id='hinge', limits=(-2., 1.))]))
    views = [SimpleNamespace(id='front'), SimpleNamespace(id='side')]
    calls = []

    def render(session, program, framing, joint_id, values, view_ids):
        calls.append((program, framing, values.copy(), list(view_ids)))
        return {'program': program, 'framing': framing}

    monkeypatch.setattr(search, 'render_states', render)
    monkeypatch.setattr(search, 'full_normal_framing', lambda rendered: Framing(center=(0., 0., 0.), radius=3.))
    monkeypatch.setattr(search, 'normal_view', lambda rendered: ('front', {'front': .10, 'side': .09}))
    monkeypatch.setattr(search, 'checker_result', lambda *args: {'code_correct': True})
    monkeypatch.setattr(search, 'attempts', lambda *args: [
        {'program': f'axis-{i}', 'label': None, 'seed': i} for i in range(3)])
    monkeypatch.setattr(search, 'screen', lambda ref, candidate, view, check: (
        {'motion': .2}, [] if candidate['program'] == f'axis-{pass_seed}' else ['motion_not_clear']))
    monkeypatch.setattr(search, 'inside_camera', lambda rendered, framing, view: (
        not (outside and rendered['program'] is not reference), 1.1 if outside else .7))
    return search, reference, views, calls


def test_failed_corruptions_never_retry_another_normal_camera(monkeypatch):
    search, reference, views, calls = _mock_search(monkeypatch)
    pair, diagnostics = search.find_pair(None, reference, views, 100)
    assert pair is None
    normal_calls = [c for c in calls if c[0] is reference]
    axis_calls = [c for c in calls if c[0] is not reference]
    # One initial normal sweep and one normal camera selection, even though all
    # three corruptions fail and another view/endpoint/padding remains possible.
    assert len(normal_calls) == 2
    assert len(axis_calls) == 3
    assert all(c[1] == normal_calls[-1][1] for c in axis_calls)
    assert all(c[2] == {'rest': 0., 'mid': -1., 'limit': -2.} for c in calls)
    assert all(c[3] == ['front'] for c in axis_calls)
    assert len(diagnostics) == 1
    assert diagnostics[0]['evaluated_views'] == ['front']
    assert all(a['view'] == 'front' for a in diagnostics[0]['attempts'])


def test_first_passing_seed_stops_without_score_optimization(monkeypatch):
    search, reference, views, calls = _mock_search(monkeypatch, pass_seed=1)
    pair, diagnostics = search.find_pair(None, reference, views, 100)
    assert pair[1]['seed'] == 1
    assert [c[0] for c in calls if c[0] is not reference] == ['axis-0', 'axis-1']
    assert diagnostics[0]['selected'] == {'seed': 1, 'view': 'front'}


def test_outside_frozen_camera_rejects_otherwise_passing_corruption(monkeypatch):
    search, reference, views, calls = _mock_search(monkeypatch, pass_seed=0, outside=True)
    pair, diagnostics = search.find_pair(None, reference, views, 100)
    assert pair is None
    assert 'geometry_outside_frozen_camera' in diagnostics[0]['attempts'][0]['rejection_reasons']
    assert len([c for c in calls if c[0] is reference]) == 2


def test_continuation_rejects_overlapping_output_before_any_render(tmp_path):
    from scripts.extend_level0_axis_v2 import run
    for out, base in ((tmp_path, tmp_path), (tmp_path / 'nested', tmp_path),
                      (tmp_path, tmp_path / 'nested')):
        with pytest.raises(ValueError, match='separate.*immutable baseline'):
            run(out=out, base=base)


def test_reviewed_v2_can_be_continued_without_rendering_or_changing_baseline(tmp_path, monkeypatch):
    from scripts import extend_level0_axis_v2 as expansion

    baseline_manifest = (DEFAULT_OUT / 'manifest.json').read_bytes()
    baseline = json.loads(baseline_manifest)
    source_gold = read_rows(DEFAULT_OUT / 'private/gold.jsonl')
    source_reviews = read_rows(DEFAULT_OUT / 'human_review.jsonl')
    monkeypatch.setattr(expansion, 'load_pool', lambda: ([], {'groups': {}}, {}))
    monkeypatch.setattr(expansion, 'ObservationSession', lambda *args: nullcontext(object()))
    out = tmp_path / 'continuation'
    (out / 'private').mkdir(parents=True)
    (out / 'private/capacity_audit.json').write_text(json.dumps({'inputs': {}, 'queue': []}))
    result = expansion.run(out=out, base=DEFAULT_OUT, target=50)
    assert result['base_objects'] == 28
    assert result['automatic_pass_objects'] == result['training_eligible_objects'] == 28
    assert result['new_objects'] == 0
    assert result['training_eligible_cases'] == 56
    assert result['acceptance'] == 'target_not_met'
    assert read_rows(out / 'private/gold.jsonl') == source_gold
    assert read_rows(out / 'human_review.jsonl') == source_reviews
    assert (DEFAULT_OUT / 'manifest.json').read_bytes() == baseline_manifest
    inherited = json.loads((out / 'private/expansion_base_manifest.json').read_text())
    assert inherited == baseline
    for name, digest in baseline['files'].items():
        assert sha256_path(DEFAULT_OUT / name) == digest
        if name.startswith(('boards/', 'inputs/', 'frames/', 'private/l0-')):
            assert sha256_path(out / name) == digest
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['base_manifest_sha256'] == sha256_path(DEFAULT_OUT / 'manifest.json')
    for name, digest in manifest['files'].items():
        assert sha256_path(out / name) == digest
    for name, digest in manifest['inputs'].items():
        assert sha256_path(Path(name)) == digest


V3 = DEFAULT_OUT.with_name('partnet-mobility-level0-axis-v3')


def test_v3_preserves_all_inherited_artifacts_and_archives_original_source_pins():
    baseline = json.loads((DEFAULT_OUT / 'manifest.json').read_text())
    archived = V3 / 'private/provenance'
    for name, digest in baseline['inputs'].items():
        if Path(name).name in {'extend_level0_axis_v2.py', 'level0_expansion_search.py', 'finalize_level0_axis_v2.py'}:
            assert sha256_path(archived / Path(name).name) == digest
    for name, digest in baseline['files'].items():
        assert sha256_path(DEFAULT_OUT / name) == digest
        if name.startswith(('boards/', 'inputs/', 'frames/', 'private/l0-')):
            assert sha256_path(V3 / name) == digest
    for relative in ('private/gold.jsonl', 'human_review.jsonl'):
        originals = read_rows(DEFAULT_OUT / relative)
        current = {r['case_id']: r for r in read_rows(V3 / relative)}
        assert all(current[r['case_id']] == r for r in originals)


def test_v3_manifest_review_and_pairwise_export_are_consistent():
    from scripts.finalize_level0_axis_v2 import accepted
    manifest = json.loads((V3 / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha256_path(V3 / name) == digest
    for name, digest in manifest['inputs'].items():
        assert sha256_path(Path(name)) == digest
    public = read_rows(V3 / 'review_manifest.jsonl')
    gold = read_rows(V3 / 'private/gold.jsonl')
    reviews = {r['case_id']: r for r in read_rows(V3 / 'human_review.jsonl')}
    prior = {r['case_id'] for r in read_rows(DEFAULT_OUT / 'human_review.jsonl')}
    inherited = json.loads((V3 / 'private/inherited_review_provenance.json').read_text())
    exported = {r['case_id'] for r in read_rows(V3 / 'train_manifest.jsonl')}
    baseline_splits = json.loads((DEFAULT_OUT / 'private/expanded_splits.json').read_text())['groups']
    splits = json.loads((V3 / manifest['expanded_splits']).read_text())['groups']
    assert all(splits[k] == v for k, v in baseline_splits.items())
    assert all(v == 'train' for k, v in splits.items() if k not in baseline_splits)
    expected = set()
    for asset_id in {g['asset_id'] for g in gold}:
        pair = [g for g in gold if g['asset_id'] == asset_id]
        assert len(pair) == 2
        assert {g['kind'] for g in pair} == {'clean', 'joint_axis'}
        if all(accepted(reviews[g['case_id']], g, g['case_id'] in inherited) for g in pair):
            expected.update(g['case_id'] for g in pair)
    assert exported == expected
    confirmation = json.loads((V3 / 'private/user_review_confirmation.json').read_text())
    new_ids = {p['case_id'] for p in public} - prior
    statement = '我已人工审查，这些都没问题；请把这次更改和上次更改合并在一起提交commit'
    assert len(new_ids) == 20
    assert confirmation['case_ids'] == sorted(new_ids)
    assert confirmation['user_statement'] == statement
    assert confirmation['decision_author'] == 'user'
    assert confirmation['recorded_by'] == 'Codex'
    assert confirmation['review_mode'] == 'user_reported_visual_review_aggregate_confirmation'
    gold_by_id = {g['case_id']: g for g in gold}
    for p in public:
        assert not {'kind', 'defect_id', 'expected_status', 'parameters', 'joint_id', 'asset_id'} & p.keys()
        review = reviews[p['case_id']]
        if p['case_id'] not in prior:
            assert review['status'] == 'human_review_complete'
            assert review['reviewer'] == 'user'
            assert review['reviewed_at'] == '2026-10-20'
            assert review['review_source'] == statement
            assert review['motion_description_author'] == 'AI operator; visual summary, not a verbatim human quote'
            assert review['verdict_recording'] == (
                'Mapped from the reviewed clean/axis pair design under explicit aggregate user approval; '
                'not independently supplied per-case labels.')
            assert accepted(review, gold_by_id[p['case_id']])
            assert p['case_id'] in exported
    assert len(exported) == 76


def test_v3_operational_commands_target_continuation_directory():
    manifest = json.loads((V3 / 'manifest.json').read_text())
    readme = (V3 / 'README.md').read_text()
    dataset_directory = manifest['dataset_directory']
    assert Path(dataset_directory).name == V3.name
    commands = [line.strip() for line in readme.splitlines()
                if line.strip().startswith('python -m scripts.')]
    assert any(f'--dataset {dataset_directory}' in command for command in commands)
    assert any(f'--out {dataset_directory}' in command for command in commands)
    assert all('--out docs/evidence/partnet-mobility-level0-axis-v2' not in command
               for command in commands)


@pytest.mark.parametrize('audit_behavior', ['failed_report', 'exception'])
def test_source_authenticity_failure_blocks_pair_before_images_and_review(tmp_path, monkeypatch, audit_behavior):
    from dataclasses import replace
    from scripts import extend_level0_axis_v2 as expansion
    from turnitover.assets.catalog import CatalogSource

    baseline = json.loads((DEFAULT_OUT / 'manifest.json').read_text())
    catalog = CatalogSource(str(DEFAULT_OUT / baseline['source_catalog']))
    source_id = read_rows(DEFAULT_OUT / 'private/gold.jsonl')[0]['asset_id']
    spec = replace(catalog.get(source_id), asset_id='PM_CPU_FIXTURE')
    metadata = {'model_id': 'CPU_FIXTURE', 'source_asset_id': 'CPU_FIXTURE',
                'archive_sha256': 'synthetic-archive', 'original_spec_sha256': 'synthetic-spec',
                'original_geometry_fingerprint': 'synthetic-geometry'}
    queue = {'asset_id': 'CPU_FIXTURE', 'model_id': metadata['model_id'],
             'archive_sha256': metadata['archive_sha256'], 'spec_sha256': metadata['original_spec_sha256'],
             'geometry_fingerprint': metadata['original_geometry_fingerprint']}
    out = tmp_path / 'authenticity-gate'
    (out / 'private').mkdir(parents=True)
    (out / 'private/capacity_audit.json').write_text(json.dumps({'inputs': {}, 'queue': [queue]}))
    monkeypatch.setattr(expansion, 'load_pool', lambda: ([
        {'spec': spec, 'metadata': metadata, 'original_train': False}], {'groups': {}}, {}))
    monkeypatch.setattr(expansion, 'ObservationSession', lambda *args: nullcontext(object()))
    monkeypatch.setattr(expansion, 'find_pair', lambda *args: (('synthetic selected pair',), []))
    monkeypatch.setattr(expansion, 'UrdfModel', lambda path: object())

    def audit(*args):
        if audit_behavior == 'exception':
            raise ValueError('Triangle topology changed for link_0')
        return {'passed': False, 'max_vertex_error_m': .1}

    monkeypatch.setattr(expansion, 'audit_model', audit)
    monkeypatch.setattr(expansion, 'board', lambda *args: pytest.fail('Rejected source must never write review images'))
    result = expansion.run(out=out, base=DEFAULT_OUT, target=50)
    assert result['automatic_pass_objects'] == result['training_eligible_objects'] == 28
    assert read_rows(out / 'private/gold.jsonl') == read_rows(DEFAULT_OUT / 'private/gold.jsonl')
    assert read_rows(out / 'human_review.jsonl') == read_rows(DEFAULT_OUT / 'human_review.jsonl')
    screening = json.loads((out / 'private/expansion_screening.json').read_text())['assets']
    rejected = next(r for r in screening if r['asset_id'] == spec.asset_id)
    assert rejected['status'] == 'rejected'
    assert rejected['rejection_reason_counts'] == {'source_authenticity_failed': 1}
    assert rejected['source_authenticity']['passed'] is False
    if audit_behavior == 'exception':
        assert rejected['source_authenticity']['error_type'] == 'ValueError'
        assert rejected['source_authenticity']['error'] == 'Triangle topology changed for link_0'
