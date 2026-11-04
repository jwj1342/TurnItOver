"""Source bindings and custody boundaries for the portable Medium candidate."""
from __future__ import annotations

import json
import re

if __package__:
    from .easy_release_archive import sha256
    from .validate_easy_bundle import _safe_relative_path
else:
    from easy_release_archive import sha256
    from validate_easy_bundle import _safe_relative_path

SOURCE_FILES = {'source/manifest.json', 'source/summary.json', 'source/automatic-checks.json',
                'source/tightening_policy.json', 'source/metadata.json'}
ROOT_FILES = {'README.md', 'review_manifest.jsonl', 'private/gold.jsonl', 'private/checks.jsonl',
              'human_review.jsonl', 'private/human_spot_review.jsonl',
              'private/human_spot_dispositions.jsonl', 'private/review_mapping.jsonl',
              'construction_registry.json', 'train_manifest.jsonl', 'private/train_targets.jsonl'}


def safe_path(value):
    if not isinstance(value, str) or '\\' in value or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('Unsafe Medium bundle path')
    return _safe_relative_path(value)


def validate_source(root, preserved, gold):
    meta = json.loads((root / 'source/metadata.json').read_text())
    source = json.loads((root / 'source/manifest.json').read_text())
    identity = sha256(root / 'source/manifest.json')
    if (meta.get('version') != 1 or meta.get('source_manifest_sha256') != identity
            or not re.fullmatch(r'[0-9a-f]{40}', meta.get('source_code_commit', ''))
            or meta.get('source_dirty') is not True
            or meta.get('generation_policy') != 'medium-strength-v3'
            or meta.get('selection_policy') != 'medium-selection-tightening-v1'
            or meta.get('archive_availability') != 'local_custody'
            or meta.get('benchmark_frozen') is not False or meta.get('training_ready') is not False
            or source.get('policy', {}).get('version') != meta['generation_policy']):
        raise ValueError('Medium release source identity or candidate boundary is invalid')
    copies = {name: name for name in preserved}
    copies.update({'source/summary.json': 'summary.json',
                   'source/automatic-checks.json': 'private/verification/automatic-checks.json',
                   'source/tightening_policy.json': 'private/tightening_policy.json'})
    for copied, original in copies.items():
        if source['files'].get(original) != sha256(root / copied):
            raise ValueError(f'Portable asset differs from the sealed source: {copied}')
    report = json.loads((root / 'source/automatic-checks.json').read_text())
    if (report.get('status') != 'automatic_checks_passed_pending_human_review'
            or report.get('objects') != 75 or report.get('cases') != 150
            or report.get('selection_replayed_objects') != 75
            or report.get('generation_inputs_verified') is not True
            or report.get('selection_policy') != meta['selection_policy']
            or report.get('human_spot_medium_cases') != 11 or report.get('human_excluded_objects') != 5):
        raise ValueError('Original complete generation/selection evidence is missing')
    registry = json.loads((root / 'construction_registry.json').read_text())
    programs = registry.get('programs', [])
    by_id = {r['case_id']: r for r in programs}
    if (registry.get('availability') != 'local_custody' or registry.get('source_manifest_sha256') != identity
            or len(programs) != len(gold) or len(by_id) != len(gold) or set(by_id) != set(gold)):
        raise ValueError('Construction registry must cover all active programs in local custody')
    for cid, g in gold.items():
        row = by_id[cid]
        safe_path(g['program'])
        if (row.get('source_path') != g['program'] or row.get('sha256') != source['files'].get(g['program'])
                or not re.fullmatch(r'[0-9a-f]{64}', row.get('sha256', ''))
                or type(row.get('size')) is not int or row['size'] <= 0
                or row.get('candidate_sha') != g['candidate_sha']
                or row.get('reference_sha') != g['reference_sha']):
            raise ValueError(f'Construction custody reference changed: {cid}')


def validate_check(gold, check):
    clean = gold['kind'] == 'clean'
    results = check.get('checks')
    if not isinstance(results, list) or len(results) != 2 or any(not isinstance(r, dict) for r in results):
        raise ValueError('Candidate requires stored geometry and runtime checker evidence')
    geometry, runtime = results
    if (check.get('candidate_sha') != gold['candidate_sha'] or check.get('reference_sha') != gold['reference_sha']
            or check.get('code_correct') is not clean or check.get('used_by_student') is not False
            or check.get('privileged') is not True or geometry.get('checker_id') != 'geometry.alignment'
            or runtime.get('checker_id') != 'runtime.budget' or geometry.get('passed') is not clean
            or runtime.get('passed') is not True or not runtime.get('invariants')
            or any(r.get('passed') is not True for r in runtime['invariants'])):
        raise ValueError('Candidate checker evidence or program identity is inconsistent')
    failures = [r for r in geometry.get('invariants', []) if r.get('passed') is False]
    if clean:
        if not geometry.get('invariants') or failures:
            raise ValueError('Clean case fails its stored geometry checks')
    else:
        required = ('rest', 'joint:target:mid', 'joint:target:limit') if gold['kind'] == 'part_offset' else ('joint:target:mid', 'joint:target:limit')
        if (len(gold.get('parts', [])) != 1 or any(not any(
                gold['parts'][0] in r.get('parts', []) and r.get('state') == state for r in failures)
                for state in required)):
            raise ValueError('Stored checker does not flag the declared defect target')
