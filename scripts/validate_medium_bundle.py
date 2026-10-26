"""Validate a portable Medium candidate without local source data or mesh replay."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

if __package__:
    from .easy_release_archive import sha256
    from .medium_bundle_contract import ROOT_FILES, SOURCE_FILES, safe_path, validate_check, validate_source
    from .medium_bundle_history import history_files, validate_history
    from .medium_bundle_inventory import validate_manifest, write_manifest
    from .validate_easy_bundle import _png_size, _read_jsonl
else:
    from easy_release_archive import sha256
    from medium_bundle_contract import ROOT_FILES, SOURCE_FILES, safe_path, validate_check, validate_source
    from medium_bundle_history import history_files, validate_history
    from medium_bundle_inventory import validate_manifest, write_manifest
    from validate_easy_bundle import _png_size, _read_jsonl

EXPECTED_OBJECTS = 75
EXPECTED_CONFIGS = {'multi_joint_axis': 45, 'multi_direction_flip': 22, 'single_part_offset': 8}
KINDS = {'multi_joint_axis': 'joint_axis', 'multi_direction_flip': 'direction_flip',
         'single_part_offset': 'part_offset'}
LEDGERS = {'review_manifest.jsonl', 'private/gold.jsonl', 'private/checks.jsonl',
           'human_review.jsonl', 'private/human_spot_review.jsonl', 'private/human_spot_dispositions.jsonl',
           'train_manifest.jsonl', 'private/train_targets.jsonl'}
SOURCE_KEYS = ('source_identity', 'model_id', 'archive_sha256', 'original_spec_sha256',
               'original_geometry_fingerprint')
STATES = ('rest', 'mid', 'limit')


def _index(path):
    rows = _read_jsonl(path)
    indexed = {r['case_id']: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError(f'Duplicate case IDs: {path.name}')
    return indexed


def _pending(row):
    judgment = row.get('human_judgment')
    if (row.get('status') != 'pending_human_review' or row.get('reviewer') is not None
            or row.get('reviewed_at') is not None or not isinstance(judgment, dict) or not judgment
            or any(value is not None and value != [] for value in judgment.values())):
        raise ValueError('Full diagnostic human review must remain an unfilled pending template')


def _case_media(root, cid, public):
    expected = {'board': f'boards/{cid}.png', 'input_preview': f'inputs/{cid}.png',
                'images': [f'frames/{cid}/{i}.png' for i in range(4)]}
    if (any(public.get(k) != v for k, v in expected.items())
            or public.get('states') != list(STATES) or public.get('input_states') != ['rest', 'limit']
            or public.get('image_size') != [256, 256] or public.get('status') != 'pending_human_review'):
        raise ValueError('Case media paths, states or candidate status are inconsistent')
    media = {expected['board'], expected['input_preview'], *expected['images']}
    native = {f'native/{cid}/{role}_{state}.png' for role in ('reference', 'candidate') for state in STATES}
    for name in media | native:
        safe_path(name)
        size = _png_size(root / name)
        if (name.startswith(('native/', 'frames/')) and size != (256, 256)) or min(size) <= 0:
            raise ValueError('Invalid native or review PNG dimensions')
    endpoints = [('reference', 'rest'), ('reference', 'limit'), ('candidate', 'rest'), ('candidate', 'limit')]
    for name, (role, state) in zip(expected['images'], endpoints):
        if sha256(root / name) != sha256(root / f'native/{cid}/{role}_{state}.png'):
            raise ValueError('Four input frames disagree with the original native states')
    return media | native


def _review_sources(root, gold):
    if __package__:
        from .prepare_medium_review import validate_review_bundle
    else:
        from prepare_medium_review import validate_review_bundle
    review = root / 'review_bundle'
    report = validate_review_bundle(review)
    rows = _read_jsonl(root / 'private/review_mapping.jsonl')
    mapping = {r['case_id']: r['source_case_id'] for r in rows}
    cards = json.loads((review / 'case_manifest.json').read_text())
    if (len(rows) != len(gold) or len(mapping) != len(rows) or len(set(mapping.values())) != len(gold)
            or set(mapping.values()) != set(gold) or {r['case_id'] for r in cards} != set(mapping)
            or any(set(r) != {'case_id', 'source_case_id'} for r in rows)):
        raise ValueError('Blind review mapping must cover every active case exactly once')
    for card in cards:
        for role in ('reference', 'candidate'):
            for state in STATES:
                original = root / f"native/{mapping[card['case_id']]}/{role}_{state}.png"
                if sha256(review / card[role][state]) != sha256(original):
                    raise ValueError('Blind review media changed or belongs to a different source case')
    secrets = {*gold, *(g['asset_id'] for g in gold.values())}
    for path in review.rglob('*'):
        if path.is_file() and path.suffix in {'.json', '.jsonl', '.html', '.js', '.css', '.md'}:
            text = path.read_text()
            if any(value in text for value in secrets):
                raise ValueError('Blind review exposes an original case or source object identity')
    return report


def validate_bundle(root: Path) -> dict:
    if root.is_symlink():
        raise ValueError('Candidate root must not be a symbolic link')
    root = root.resolve()
    inventory = validate_manifest(root)
    if any((root / n).read_text().strip() for n in ('train_manifest.jsonl', 'private/train_targets.jsonl')):
        raise ValueError('Medium candidate must not contain training exports')
    gold, public, human, checks = (_index(root / name) for name in
                                 ('private/gold.jsonl', 'review_manifest.jsonl', 'human_review.jsonl', 'private/checks.jsonl'))
    if not (set(gold) == set(public) == set(human) == set(checks)) or len(gold) != EXPECTED_OBJECTS * 2:
        raise ValueError('Medium Gold, media, checks and diagnostic ledgers must align at 150 cases')
    groups = defaultdict(list)
    media = set()
    for cid, g in gold.items():
        if not re.fullmatch(r'ms-[0-9a-f]{14}', cid) or g.get('config') not in KINDS:
            raise ValueError('Invalid source case identity or Medium configuration')
        if (g.get('training_eligibility') != 'pending_human_review' or g.get('difficulty') != 'medium_provisional'
                or g.get('screening_policy') != 'medium-strength-v3' or g['source'].get('split') != 'train'):
            raise ValueError('Medium candidate eligibility, source grouping or generation policy changed')
        _pending(human[cid])
        validate_check(g, checks[cid])
        groups[g['asset_id']].append(g)
        media.update(_case_media(root, cid, public[cid]))
    if len(groups) != EXPECTED_OBJECTS or any(len(pair) != 2 for pair in groups.values()):
        raise ValueError('Medium requires 75 independent Clean/defect object pairs')
    sources = {key: set() for key in SOURCE_KEYS}
    configs = Counter()
    for pair in groups.values():
        clean = next((g for g in pair if g['kind'] == 'clean'), None)
        if clean is None:
            raise ValueError('Object pair lacks one Clean case')
        defect = next((g for g in pair if g['kind'] != 'clean'), None)
        if (defect is None or defect['config'] != clean['config'] or defect['kind'] != KINDS[clean['config']]
                or clean.get('expected_status') != 'pass' or defect.get('expected_status') != 'fail'
                or clean.get('defect_id') is not None or defect.get('defect_id') !=
                ('structure.part_offset' if defect['kind'] == 'part_offset' else 'kinematics.' + defect['kind'])
                or clean['candidate_sha'] != clean['reference_sha'] or defect['reference_sha'] != clean['reference_sha']
                or clean['source'] != defect['source'] or clean['state_values'] != defect['state_values']
                or public[clean['case_id']]['view'] != public[defect['case_id']]['view']):
            raise ValueError('Clean/defect semantics or frozen source identity changed')
        for key in SOURCE_KEYS:
            identity = clean['source'].get(key)
            if not identity or identity in sources[key]:
                raise ValueError(f'Duplicate or missing independent source identity: {key}')
            sources[key].add(identity)
        for g in pair:
            for state in STATES:
                canonical = root / f"native/{clean['case_id']}/reference_{state}.png"
                if sha256(root / f"native/{g['case_id']}/reference_{state}.png") != sha256(canonical):
                    raise ValueError('Paired cases lost their common frozen reference')
                if g is clean and sha256(root / f"native/{g['case_id']}/candidate_{state}.png") != sha256(canonical):
                    raise ValueError('Clean candidate differs from its reference pixels')
        configs[clean['config']] += 1
    if dict(configs) != EXPECTED_CONFIGS:
        raise ValueError('Medium configuration counts differ from the selected source snapshot')
    originals = LEDGERS | history_files() | media
    validate_source(root, originals, gold)
    visibility = validate_history(root, gold)
    review = _review_sources(root, gold)
    allowed = ROOT_FILES | SOURCE_FILES | history_files() | media
    if {name for name in inventory if not name.startswith('review_bundle/')} != allowed:
        raise ValueError('Unexpected or missing candidate files; construction programs remain in local custody')
    return {'objects': len(groups), 'cases': len(gold), 'clean_cases': len(groups), 'defect_cases': len(groups),
            'config_objects': dict(sorted(configs.items())), 'train_objects': len(groups), 'validation_objects': 0,
            'native_images': len(gold) * 6, 'input_frames': len(gold) * 4, 'media_files': len(media),
            'human_visibility': visibility, 'full_diagnostic_pending': len(human), 'training_eligible_cases': 0,
            'archive_availability': 'local_custody', 'program_replay_performed': False,
            'validation_scope': 'portable files, source bindings, stored evidence and review isolation; no mesh replay',
            'review_bundle': review}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest(args.root)
    print(json.dumps(validate_bundle(args.root), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
