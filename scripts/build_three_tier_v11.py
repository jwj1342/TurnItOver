"""Reproducible label-only revision of the pinned private three-tier-900 v1 export."""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turnitover.oracle.clean_pass import POLICY, observation_digest, clean_verdict, validate_clean_pass

PARENT_ARCHIVE = '185d2d6ba7ca2b6d2fb7426d407e50c9682a8632e3a827f18502cab9c309e2e0'
PARENT_MANIFEST = '6d0fc5344810396e9454e83c58572f9a9ecdcae4cb53a477f59e9dd541fc6c25'
VERSION = 'three-tier-900-sft-v1.1'
PREFIX = 'three-tier-900-sft-v1/episode_export/'


def sha(data): return hashlib.sha256(data).hexdigest()
def write_json(path, data): path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf-8')
def write_rows(path, rows): path.write_text(''.join(json.dumps(x, ensure_ascii=False, sort_keys=True)+'\n' for x in rows), encoding='utf-8')
def read_rows(data): return [json.loads(x) for x in data.splitlines() if x.strip()]


def build(args):
    if args.out.exists(): raise ValueError('Output must be fresh')
    assert sha(args.source_archive.read_bytes()) == PARENT_ARCHIVE
    summary = json.loads(args.review_summary.read_text())
    assert summary['cause_adjudication'] == 'pending_for_all_16'
    assert summary['sources']['candidate_zip']['sha256'] == PARENT_ARCHIVE
    assert summary['cases'] == 120 and summary['verified_images'] == 880
    reviews = read_rows(args.reviews.read_text())
    review_map = {x['source_case_id']: x for x in reviews}
    assert len(review_map) == len(reviews) == 120
    args.out.mkdir(parents=True)
    snapshot = json.loads(args.source_snapshot.read_text())
    with zipfile.ZipFile(args.source_archive) as z:
        assert len(z.namelist()) == len(set(z.namelist()))
        assert set(z.namelist()) == set(snapshot['files'])
        for member in z.namelist():
            data = z.read(member)
            assert sha(data) == snapshot['files'][member], member
            if member.startswith(PREFIX):
                relative = Path(member[len(PREFIX):])
                assert not relative.is_absolute() and '..' not in relative.parts
                dest = args.out/relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
    assert sha((args.out/'manifest.json').read_bytes()) == PARENT_MANIFEST
    manifest = json.loads((args.out/'manifest.json').read_text())
    annotations = read_rows((args.out/'annotations.private.jsonl').read_text())
    amap = {x['id']: x for x in annotations}
    assert len(amap) == len(annotations) == 900
    assert {x['id'] for x in annotations if x['split'] == 'validation'} == set(review_map)
    label_changes, quarantined, stats = [], [], {}
    for a in annotations:
        a.update(label_policy=POLICY, is_clean=not a['labels'], admission='eligible')
        review = review_map.get(a['id'])
        if review:
            assert json.loads(review['injected_labels_json']) == a['labels']
            assert review['dataset_difficulty'] == a['difficulty']
            assert (review['preset_status'] == 'clean') == a['is_clean']
            a['human_review'] = copy.deepcopy(review)
            if a['is_clean']:
                assert review['human_decision'] == 'clean'
            elif review['human_decision'] in ('clean', 'invalid'):
                a['admission'] = 'quarantined_pending_adjudication'
                quarantined.append({'id': a['id'], 'split': a['split'], 'difficulty': a['difficulty'],
                                    'human_decision': review['human_decision'], 'reason': 'issue16_cause_adjudication_pending'})
        else:
            a['human_review'] = {'status': 'no_case_level_review_attached_in_this_revision'}
        if a['is_clean']:
            assert a['reference_sha'] == a['candidate_sha'] == a['data_identity']['reference_sha'] == a['data_identity']['candidate_sha']
    assert len(quarantined) == 16
    for split in ('train', 'validation', 'test'):
        before = read_rows((args.out/f'{split}.episodes.jsonl').read_text())
        rows = copy.deepcopy(before)
        stop_rows = read_rows((args.out/f'{split}.stopping.jsonl').read_text())
        stop_by_id = {x['episode_id']: x for x in stop_rows}
        for row, old in zip(rows, before):
            a = amap[row['episode_id']]
            assert a['split'] == split and a['data_identity'] == row['data_identity']
            stop = stop_by_id[row['episode_id']]
            for r in (row, stop):
                r.update(label_policy=POLICY, is_clean=a['is_clean'], admission=a['admission'])
            if a['is_clean']:
                cert = {'policy': POLICY, 'basis': 'published_reference_identity_and_complete_frozen_observation',
                        'scope': 'published_structure_kinematics_fixed_observation_task',
                        'case_id': a['id'], 'reference_sha': a['reference_sha'],
                        'observation_sha256': observation_digest(row), 'parent_manifest_sha256': PARENT_MANIFEST}
                for r in (a, row, stop): r['pass_certificate'] = copy.deepcopy(cert)
                validate_clean_pass(row, a)
                validate_clean_pass(stop, a)
                row['messages'][1]['content'] = json.dumps(clean_verdict())
                stop['messages'][1]['content'] = json.dumps({'stop_decision': {'decision': 'stop', 'outcome': 'pass',
                    'reason': 'The complete frozen reference comparison passes within the observed task scope.'}})
                label_changes.append({'case_id': a['id'], 'split': split, 'difficulty': a['difficulty'],
                                      'old_diagnosis_status': 'uncertain', 'new_diagnosis_status': 'pass',
                                      'old_stopping': 'continue', 'new_stopping': 'stop/pass',
                                      'basis': cert['basis']})
            assert row['messages'][0] == old['messages'][0]
            assert row['images'] == old['images'] and row['image_sha256'] == old['image_sha256']
            if not a['is_clean']: assert row['messages'][1] == old['messages'][1]
        eligible = [r for r in rows if r['admission'] == 'eligible']
        statuses = lambda rs: dict(collections.Counter(json.loads(r['messages'][1]['content'])['verdict']['status'] for r in rs))
        stats[split] = {'raw_cases': len(rows), 'eligible_cases': len(eligible), 'quarantined_cases': len(rows)-len(eligible),
                        'clean': sum(r['is_clean'] for r in rows), 'raw_evidence_status': statuses(rows),
                        'eligible_evidence_status': statuses(eligible),
                        'eligible_ordinary_status': {'pass': sum(r['is_clean'] for r in eligible),
                                                      'fail': sum(not r['is_clean'] for r in eligible)},
                        'eligible_difficulty': dict(collections.Counter(r['difficulty'] for r in eligible)),
                        'raw_objects': len({r['data_identity']['source_identity_sha256'] for r in rows}),
                        'eligible_objects': len({r['data_identity']['source_identity_sha256'] for r in eligible})}
        write_rows(args.out/f'{split}.episodes.jsonl', rows)
        write_rows(args.out/f'{split}.stopping.jsonl', stop_rows)
    assert len(label_changes) == 450
    write_rows(args.out/'annotations.private.jsonl', annotations)
    write_rows(args.out/'label-changes.private.jsonl', label_changes)
    write_rows(args.out/'quarantine.private.jsonl', quarantined)
    shutil.copyfile(args.reviews, args.out/'human-review-comparison.private.jsonl')
    # Avoid publishing machine-local paths while preserving source hashes.
    review_provenance = {k: v for k,v in summary.items() if k != 'sources'}
    review_provenance['source_sha256'] = {k: v['sha256'] for k,v in summary['sources'].items()}
    write_json(args.out/'human-review-provenance.json', review_provenance)
    manifest.update(dataset_version=VERSION, label_policy=POLICY,
        release_status='prerelease', formal_benchmark=False, training_ready=False,
        experimental_use='owner_authorized_label_revision_with_explicit_quarantine',
        admission_policy='eligible_only_by_default; retain quarantined raw rows without adjudicating them',
        parent_dataset_version='three-tier-900-sft-v1', parent_manifest_sha256=PARENT_MANIFEST,
        parent_archive_sha256=PARENT_ARCHIVE, label_revision_counts=stats,
        changed_clean_targets=450, quarantined_validation_cases=16,
        test_access='read_for_mechanical_release_integrity_and_label_conversion_only; no_model_evaluation_or_selection',
        human_review_status='validation_120_reconciled; 16_causes_pending; train_test_not_newly_human_reviewed',
        evidence_support_status='inherited_counterfactual_pixel_proxy_not_independent_human_per_finding_grounding')
    manifest['files'] = {p.relative_to(args.out).as_posix(): sha(p.read_bytes()) for p in sorted(args.out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    write_json(args.out/'manifest.json', manifest)
    print(json.dumps({'dataset_version': VERSION, 'manifest_sha256': sha((args.out/'manifest.json').read_bytes()),
                      'changed_clean_targets': len(label_changes), 'splits': stats}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('source-archive','source-snapshot','reviews','review-summary','out'):
        p.add_argument('--'+name, type=Path, required=True)
    build(p.parse_args())
