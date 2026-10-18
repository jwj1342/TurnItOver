"""Level 0 v2 isolation, review gates, integrity and export contracts."""
import copy
import json
from collections import Counter
from pathlib import Path
import shutil

import pytest
from PIL import Image

from scripts.finalize_level0_axis_v2 import accepted, finalize, rows
from scripts.prepare_level0_axis_v2 import DEFAULT_OUT, V1
from scripts.prepare_obvious_defect_review import DEFAULT_SPLITS, sha256_path, write_jsonl


def test_v2_checksums_splits_balance_and_review_isolation():
    manifest = json.loads((DEFAULT_OUT / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha256_path(DEFAULT_OUT / name) == digest
    for name, digest in manifest['inputs'].items():
        assert sha256_path(Path(name)) == digest
    public, gold = rows(DEFAULT_OUT / 'review_manifest.jsonl'), rows(DEFAULT_OUT / 'private/gold.jsonl')
    reviews = {r['case_id']: r for r in rows(DEFAULT_OUT / 'human_review.jsonl')}
    prior = {r['case_id']: r for r in rows(V1 / 'human_review.jsonl')}
    exported = rows(DEFAULT_OUT / 'train_manifest.jsonl')
    original_splits = json.loads(DEFAULT_SPLITS.read_text())['groups']
    splits = json.loads((DEFAULT_OUT / manifest['expanded_splits']).read_text())['groups']
    assert all(splits[k] == v for k, v in original_splits.items())
    assert len(public) == len(gold) == len(reviews)
    objects = len({g['asset_id'] for g in gold})
    assert objects >= 28
    assert Counter(g['kind'] for g in gold) == {'clean': objects, 'joint_axis': objects}
    assert all(splits[f"PartNet-Mobility:{g['asset_id'][3:]}"] == 'train' for g in gold)
    assert all(not {'kind', 'defect_id', 'expected_status', 'parameters', 'joint_id', 'asset_id'} & p.keys() for p in public)
    assert all(reviews[k] == v for k, v in prior.items())
    expected_export = set()
    for asset in {g['asset_id'] for g in gold}:
        pair = [g for g in gold if g['asset_id'] == asset]
        if all(accepted(reviews[g['case_id']], g, g['case_id'] in prior) for g in pair):
            expected_export.update(g['case_id'] for g in pair)
    assert {r['case_id'] for r in exported} == expected_export
    for case_id, row in reviews.items():
        if row['status'] == 'pending_human_review':
            assert row['reviewer'] is None
            assert row['human_judgment']['verdict'] is None
            assert case_id not in {r['case_id'] for r in exported}
        elif row['status'] == 'human_review_rejected':
            assert row['human_judgment']['notes']
            assert case_id not in {r['case_id'] for r in exported}
        elif case_id not in prior:
            assert row['reviewer']
            assert row['reviewed_at']
            assert accepted(row, next(g for g in gold if g['case_id'] == case_id))
    for row in public:
        preview = Image.open(DEFAULT_OUT / row['input_preview'])
        for path, (x, y) in zip(row['images'], ((10, 10), (276, 10), (10, 276), (276, 276))):
            tile = Image.open(DEFAULT_OUT / path)
            assert tile.size == (256, 256)
            assert tile.tobytes() == preview.crop((x, y, x + 256, y + 256)).tobytes()


def test_default_approval_is_not_completed_visual_review():
    gold = {'expected_status': 'pass', 'defect_id': None}
    review = {'status': 'human_review_complete', 'human_judgment': {
        'verdict': 'pass', 'defect_id': None, 'one_glance': True,
        'normal_vs_candidate_motion': 'Both follow the same hinge motion.',
        'evidence_states': ['mid', 'limit']}}
    assert not accepted(review, gold)
    review.update(reviewer='human reviewer', reviewed_at='2026-10-18T12:00:00+08:00')
    review['human_judgment'].update(no_clipping_or_occlusion=True, rest_matches_reference=True)
    assert accepted(review, gold)
    occluded = copy.deepcopy(review)
    occluded['human_judgment']['no_clipping_or_occlusion'] = False
    assert not accepted(occluded, gold)
    wrong = copy.deepcopy(review)
    wrong['human_judgment']['verdict'] = 'fail'
    assert not accepted(wrong, gold)


def test_expansion_preserves_reviewed_native_inputs_and_independent_sources():
    manifest = json.loads((DEFAULT_OUT / 'private/expansion_base_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if name.startswith(('boards/', 'inputs/', 'frames/')) or name.startswith('private/l0-'):
            assert sha256_path(DEFAULT_OUT / name) == digest
    catalog = json.loads((DEFAULT_OUT / 'private/source_catalog/catalog.json').read_text())
    entries = catalog['assets']
    assert len(entries) == 28
    assert len({e['model_id'] for e in entries}) == 28
    assert len({e['original_geometry_fingerprint'] for e in entries}) == 28
    assert all(e['upstream_structural_checks']['frozen_test_overlap'] is False for e in entries)


def test_full_geometry_projection_rejects_objects_outside_frozen_camera():
    import numpy as np
    from scripts.level0_expansion_search import inside_camera
    from turnitover.checkers.base import PartMesh
    from turnitover.core.sample import Framing
    from turnitover.render.views import ViewDef
    def rendered(point):
        return {'geometry': {'rest': {'part': PartMesh(np.array([point]), np.empty((0, 3), dtype=int))}}}
    framing = Framing(center=(0., 0., 0.), radius=1.)
    for projection in ('orthographic', 'perspective'):
        view = ViewDef('front', 0., 0., projection)
        assert inside_camera(rendered([0., 0., 0.]), framing, view)[0]
        assert not inside_camera(rendered([10., 0., 0.]), framing, view)[0]
        assert not inside_camera(rendered([0., 0., 10.]), framing, view)[0]


def test_finalize_rejects_tampered_image_before_export(tmp_path):
    bundle = tmp_path / 'bundle'
    shutil.copytree(DEFAULT_OUT, bundle)
    image_path = bundle / rows(bundle / 'review_manifest.jsonl')[0]['input_preview']
    image_path.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Bundle SHA mismatch'):
        finalize(bundle)


def test_finalize_is_idempotent_and_validates_axis_only_specs(tmp_path):
    bundle = tmp_path / 'bundle'
    shutil.copytree(DEFAULT_OUT, bundle)
    before = (bundle / 'manifest.json').read_bytes()
    result = finalize(bundle)
    published = json.loads((DEFAULT_OUT / 'validation.json').read_text())
    assert result['training_eligible_objects'] == published['training_eligible_objects']
    assert result['pending_human_review'] == published['pending_human_review']
    assert result['automatic_pass_objects'] == 28
    assert (bundle / 'manifest.json').read_bytes() == before


def test_pair_requires_both_reviews_and_rejected_partner_blocks_export(tmp_path):
    # Synthetic decisions exist only inside pytest's temporary bundle.
    bundle = tmp_path / 'synthetic-review-fixture'
    shutil.copytree(DEFAULT_OUT, bundle)
    gold = rows(bundle / 'private/gold.jsonl')
    baseline = json.loads((bundle / 'validation.json').read_text())['training_eligible_objects']
    review = rows(bundle / 'human_review.jsonl')
    pair = [g for g in gold if g['asset_id'] == 'PM_102314']
    selected = {g['case_id']: g for g in pair}
    for row in review:
        if row['case_id'] not in selected:
            continue
        g = selected[row['case_id']]
        row.update(status='human_review_complete', reviewer='SYNTHETIC TEST ONLY',
                   reviewed_at='2026-10-18T00:00:00+08:00')
        row['human_judgment'].update(verdict=g['expected_status'], defect_id=g['defect_id'],
            one_glance=True, evidence_states=['mid', 'limit'], no_clipping_or_occlusion=True,
            rest_matches_reference=True, normal_vs_candidate_motion='Synthetic test fixture.')
    partner = next(r for r in review if r['case_id'] == pair[1]['case_id'])
    partner['status'] = 'human_review_rejected'
    write_jsonl(bundle / 'human_review.jsonl', review)
    assert finalize(bundle)['training_eligible_objects'] == baseline - 1
    partner['status'] = 'human_review_complete'
    write_jsonl(bundle / 'human_review.jsonl', review)
    assert finalize(bundle)['training_eligible_objects'] == baseline
    assert len(rows(bundle / 'train_manifest.jsonl')) == 2 * baseline
