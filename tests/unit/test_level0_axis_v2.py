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
    splits = json.loads(DEFAULT_SPLITS.read_text())['groups']
    assert len(public) == len(gold) == len(reviews)
    assert Counter(g['kind'] for g in gold) == {'clean': 7, 'joint_axis': 7}
    assert len({g['asset_id'] for g in gold}) == 7
    assert all(splits[f"PartNet-Mobility:{g['asset_id'][3:]}"] == 'train' for g in gold)
    assert all(not {'kind', 'defect_id', 'expected_status', 'parameters', 'joint_id', 'asset_id'} & p.keys() for p in public)
    assert all(reviews[k] == v for k, v in prior.items())
    assert {r['case_id'] for r in exported} == set(reviews)
    assert all(r['status'] == 'human_review_complete' for r in reviews.values())
    for case_id, row in reviews.items():
        if case_id not in prior:
            assert row['reviewer'] == 'user'
            assert '我已人工审查' in row['review_source']
            assert row['reviewed_at_precision'] == 'date'
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
    assert result['training_eligible_objects'] == 7
    assert result['pending_human_review'] == 0
    assert (bundle / 'manifest.json').read_bytes() == before


def test_pair_requires_both_reviews_and_rejected_partner_blocks_export(tmp_path):
    # Synthetic decisions exist only inside pytest's temporary bundle.
    bundle = tmp_path / 'synthetic-review-fixture'
    shutil.copytree(DEFAULT_OUT, bundle)
    gold = rows(bundle / 'private/gold.jsonl')
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
    assert finalize(bundle)['training_eligible_objects'] == 6
    partner['status'] = 'human_review_complete'
    write_jsonl(bundle / 'human_review.jsonl', review)
    assert finalize(bundle)['training_eligible_objects'] == 7
    assert len(rows(bundle / 'train_manifest.jsonl')) == 14
