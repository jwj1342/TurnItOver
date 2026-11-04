"""Pair immutable full-motion Hard defects with exact normal controls."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_official_archive_verification(path, acquisition_manifests, pins):
    path = Path(path).resolve()
    report = read(path)
    if report.get('status') != 'verified' or report.get('kind') != 'current_archive_verification':
        raise ValueError('Invalid official archive verification report')
    pins[str(path)] = sha(path)
    for row in report.get('archives', []):
        if row.get('status') != 'verified_against_official_lfs':
            raise ValueError('Archive report contains an unverified source')
        asset_id = str(row['asset_id'])
        previous = acquisition_manifests.get(asset_id)
        if previous and previous[1]['sha256'] != row['sha256']:
            raise ValueError('Conflicting official archive verification: ' + asset_id)
        acquisition_manifests[asset_id] = ('official_verification', row)


def pair_control(out, record, source_spec, program_type):
    """Use normal pixels already rendered and the real bound normal checker."""
    original = read(source_spec)
    reference = program_type(original)
    gold, check, public = record['gold'], record['checks'], record['public']
    clean_check = check['clean_check']
    if (reference.sha != gold['reference_sha'] or not clean_check['code_correct']
            or clean_check['candidate_sha'] != reference.sha
            or clean_check['reference_sha'] != reference.sha
            or not check['source_authenticity']['passed']):
        raise ValueError('Normal source/checker binding failed: ' + record['asset_id'])
    cid = 'hp-' + hashlib.sha256(('hard-tier-expansion-clean-v1:' + record['case_id']).encode()).hexdigest()[:14]
    clean_public = {**public, 'case_id': cid, 'native_root': 'native/' + cid,
                    'status': 'pending_human_review'}
    for view in (public['initial_view'], public['supplementary_view']):
        for state in public['states']:
            normal = out / public['native_root'] / view / ('reference_' + state + '.png')
            for role in ('reference', 'candidate'):
                target = out / 'native' / cid / view / (role + '_' + state + '.png')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(normal, target)
                record['files'][target.relative_to(out).as_posix()] = sha(target)
    name = 'private/programs/' + cid + '.json'
    write(out / name, original)
    record['files'][name] = sha(out / name)
    clean_gold = {**gold, 'case_id': cid, 'kind': 'clean', 'parameters': {},
        'defect_id': None, 'parts': [], 'expected_status': 'pass', 'program': name,
        'candidate_sha': reference.sha, 'training_eligibility': 'pending_human_review'}
    review = copy.deepcopy(record['review'])
    review.update(case_id=cid, reviewer=None, reviewed_at=None, status='pending_human_review')
    record['paired_control'] = {'public': clean_public, 'gold': clean_gold,
        'checks': {'case_id': cid, **clean_check}, 'review': review}
    return record


def checkpoint(out, state, target):
    records = [read(p) for p in sorted(out.glob('private/objects/*/record.json'))]
    for filename, key in [('review_manifest.jsonl', 'public'), ('private/gold.jsonl', 'gold'),
                          ('private/checks.jsonl', 'checks'), ('human_review.jsonl', 'review')]:
        rows = [entry[key] for r in records for entry in (r, r['paired_control'])]
        (out / filename).parent.mkdir(parents=True, exist_ok=True)
        (out / filename).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    report = {'status': 'pending_human_review' if len(records) >= target else 'collection_incomplete',
        'objects': len(records), 'cases': len(records) * 2, 'target_objects': target,
        'human_review_complete': False, 'training_eligible_cases': 0,
        'splits': {s: sum(r['gold']['source']['split'] == s for r in records)
                   for s in ('train', 'validation', 'test')}}
    write(out / 'validation.json', report)
    write(out / 'manifest.json', {**state, **report, 'files': {
        p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p != out / 'manifest.json'}})
    return report
