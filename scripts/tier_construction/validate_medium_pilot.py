"""Independent structural and artifact gates for the pending-only Medium pilot."""
from __future__ import annotations

import copy

import numpy as np


def validate_program_pair(reference: dict, candidate: dict, gold: dict) -> None:
    """Allow exactly the declared single corruption, including target identity."""
    kind = gold['kind']
    if kind == 'clean':
        if candidate != reference or gold['defect_id'] is not None:
            raise ValueError('Clean candidate differs from reference')
        return
    expected_ids = {'joint_axis': 'kinematics.joint_axis',
                    'direction_flip': 'kinematics.direction_flip',
                    'part_offset': 'structure.part_offset'}
    if kind not in expected_ids or gold['defect_id'] != expected_ids[kind]:
        raise ValueError('Unsupported or inconsistent corruption kind')
    restored = copy.deepcopy(candidate)
    if kind in ('joint_axis', 'direction_flip'):
        joint_id = gold['joint_id']
        source = next(j for j in reference['joints'] if j['id'] == joint_id)
        actual = next(j for j in restored['joints'] if j['id'] == joint_id)
        if gold['parts'] != [source['part']] or gold['parameters']['joint'] != joint_id:
            raise ValueError('Joint target and affected part disagree')
        before, after = np.asarray(source['axis']), np.asarray(actual['axis'])
        if not np.isfinite(after).all() or not np.isclose(np.linalg.norm(after), 1., atol=1e-6):
            raise ValueError('Invalid candidate joint axis')
        if kind == 'direction_flip':
            if not np.allclose(after, -before, atol=1e-12, rtol=0):
                raise ValueError('Direction flip must negate the declared axis')
        elif not np.isclose(np.dot(before, after), 0., atol=1e-6):
            raise ValueError('Medium joint axis must change by 90 degrees')
        actual['axis'] = source['axis']
    else:
        if len(gold['parts']) != 1:
            raise ValueError('Offset must affect exactly one declared part')
        part_id = gold['parts'][0]
        source = next(p for p in reference['parts'] if p['id'] == part_id)
        actual = next(p for p in restored['parts'] if p['id'] == part_id)
        delta = np.asarray(actual['origin']) - np.asarray(source['origin'])
        vector = np.asarray(gold['parameters']['vector'])
        if (not np.isfinite(delta).all() or np.linalg.norm(delta) <= 0 or
                not np.allclose(delta, vector, atol=1e-12, rtol=0) or
                not np.isclose(np.linalg.norm(delta), gold['parameters']['magnitude_m'], atol=1e-12)):
            raise ValueError('Offset vector or magnitude does not match actual displacement')
        actual['origin'] = source['origin']
    if restored != reference:
        raise ValueError('Modification beyond the declared single target field')


def validate_checker_target(check: dict, gold: dict, reference: dict) -> None:
    """Require actual geometry failure on the direct target, allowing descendants."""
    clean = gold['kind'] == 'clean'
    geometry, runtime = check['checks']
    if (check['code_correct'] is not clean or runtime['passed'] is not True or
            geometry['passed'] is not clean):
        raise ValueError('Independent checker disagrees with case kind')
    failures = [i for i in geometry['invariants'] if i['passed'] is False]
    if clean:
        if failures:
            raise ValueError('Clean geometry has failed invariants')
        return
    target = gold['parts'][0]
    descendants = {target}
    for _ in reference['parts']:
        descendants.update(p['id'] for p in reference['parts'] if p['parent'] in descendants)
    failed_parts = {part for invariant in failures for part in invariant['parts']}
    if target not in failed_parts or not failed_parts <= descendants:
        raise ValueError('Geometry failure does not localize to the declared target subtree')
    if gold['kind'] != 'part_offset' and any(i['state'] == 'rest' for i in failures):
        raise ValueError('Axis corruption changed rest geometry')
    states = {'rest'} if gold['kind'] == 'part_offset' else {'joint:target:mid', 'joint:target:limit'}
    if any(not any(target in i['parts'] and i['state'] == state for i in failures) for state in states):
        raise ValueError('Target has no geometric failure in every required evidence state')


def validate(out):
    """Validate sealed artifacts without rendering, changing reviews or exporting."""
    import json
    from collections import Counter, defaultdict
    from pathlib import Path

    from PIL import Image

    from scripts.tier_construction.prepare_obvious_defect_review import sha256_path
    from turnitover.assets.catalog import decode_spec
    from turnitover.core.program import ObjectProgram

    out = Path(out).resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha256_path(out / name) != digest:
            raise ValueError(f'Bundle SHA mismatch: {name}')
    relocation = {}
    migration = Path('benchmark/easy/migration_manifest.json')
    if migration.exists():
        relocation = json.loads(migration.read_text()).get('input_paths', {})
    for name, digest in manifest['inputs'].items():
        path = Path(relocation.get(name, name))
        if sha256_path(path) != digest:
            raise ValueError(f'Generation input SHA mismatch: {name}')
    actual_files = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
    if actual_files - {'manifest.json'} != set(manifest['files']):
        raise ValueError('Manifest does not cover the complete bundle')

    def rows(name):
        entries = [json.loads(line) for line in (out / name).read_text().splitlines() if line.strip()]
        indexed = {r['case_id']: r for r in entries}
        if len(entries) != len(indexed):
            raise ValueError(f'Duplicate case IDs: {name}')
        return indexed

    public = rows('review_manifest.jsonl')
    gold = rows('private/gold.jsonl')
    checks = rows('private/checks.jsonl')
    reviews = rows('human_review.jsonl')
    if len(public) != 30 or not public.keys() == gold.keys() == checks.keys() == reviews.keys():
        raise ValueError('Pilot requires exactly 30 matching public/private/review/check rows')
    for path in out.rglob('*train*manifest*.jsonl'):
        if path.read_text().strip():
            raise ValueError('Pending pilot must not have a training export')
    config_kind = {'multi_joint_axis': 'joint_axis', 'multi_direction_flip': 'direction_flip',
                   'single_part_offset': 'part_offset'}
    groups = defaultdict(list)
    for g in gold.values():
        if g['config'] not in config_kind:
            raise ValueError('Unknown pilot configuration')
        groups[(g['config'], g['asset_id'])].append(g)
    if Counter(config for config, asset in groups) != {key: 5 for key in config_kind}:
        raise ValueError('Each configuration requires five independent objects')
    if len({asset for config, asset in groups}) != 15:
        raise ValueError('Pilot configurations must use 15 independent objects')
    forbidden = {'asset_id', 'config', 'kind', 'defect_id', 'expected_status', 'parts', 'joint_id',
                 'parameters', 'seed', 'source', 'label'}
    for (config, asset), pair in groups.items():
        if len(pair) != 2 or {g['kind'] for g in pair} != {'clean', config_kind[config]}:
            raise ValueError('Each object must have one clean and one configured defect')
        clean = next(g for g in pair if g['kind'] == 'clean')
        reference = json.loads((out / f"private/programs/{clean['case_id']}.json").read_text())
        joint_count = len(reference['joints'])
        wrong_complexity = joint_count < 2 if config.startswith('multi_') else joint_count != 1
        if wrong_complexity:
            raise ValueError('Object joint complexity disagrees with configuration')
        for g in pair:
            cid = g['case_id']
            p, review = public[cid], reviews[cid]
            candidate = json.loads((out / f'private/programs/{cid}.json').read_text())
            validate_program_pair(reference, candidate, g)
            ref_sha = ObjectProgram.from_spec(decode_spec(reference)).sha
            candidate_sha = ObjectProgram.from_spec(decode_spec(candidate)).sha
            if (g['reference_sha'], g['candidate_sha']) != (ref_sha, candidate_sha):
                raise ValueError('Program SHA mismatch')
            if (checks[cid]['reference_sha'], checks[cid]['candidate_sha']) != (ref_sha, candidate_sha):
                raise ValueError('Checker SHA mismatch')
            validate_checker_target(checks[cid], g, reference)
            if g['kind'] != 'clean':
                from types import SimpleNamespace
                from scripts.tier_construction.construction_pilot import screen_motion, screen_offset
                from scripts.tier_construction.prepare_level0_axis_review import border_subject_fraction
                rendered = {role: {'frames': {state: {p['view']: (out / f'native/{cid}/{role}_{state}.png').read_bytes()}
                    for state in ('rest', 'mid', 'limit')}} for role in ('reference', 'candidate')}
                gate = screen_offset if g['kind'] == 'part_offset' else screen_motion
                _, reasons = gate(rendered['reference'], rendered['candidate'], p['view'],
                                  checks[cid], SimpleNamespace(parts=tuple(g['parts'])))
                if reasons or any(border_subject_fraction(rendered['reference']['frames'][state][p['view']]) > .01
                                  for state in ('rest', 'mid', 'limit')):
                    raise ValueError(f'Existing native-pixel quality gate failed: {cid}: {reasons}')
            if (g['expected_status'] != ('pass' if g['kind'] == 'clean' else 'fail') or
                    g['training_eligibility'] != 'pending_human_review'):
                raise ValueError('Gold or eligibility disagrees with pending pilot scope')
            if (forbidden & p.keys() or p['status'] != 'pending_human_review' or
                    review['status'] != 'pending_human_review' or review['reviewer'] is not None or
                    review['reviewed_at'] is not None or
                    any(v is not None and v != [] for v in review['human_judgment'].values())):
                raise ValueError('Pending human review or public isolation violated')
            if p['image_size'] != [256, 256] or p['states'] != ['rest', 'mid', 'limit']:
                raise ValueError('Unexpected native image/state protocol')
            with Image.open(out / p['input_preview']) as preview, Image.open(out / p['board']) as board:
                if preview.size != (542, 542) or board.size != (912, 630):
                    raise ValueError('Unexpected review image dimensions')
                for i, (role, state) in enumerate((('reference', 'rest'), ('reference', 'limit'),
                                                   ('candidate', 'rest'), ('candidate', 'limit'))):
                    with Image.open(out / p['images'][i]) as frame, Image.open(
                            out / f'native/{cid}/{role}_{state}.png') as native:
                        if frame.size != (256, 256) or frame.convert('RGB').tobytes() != native.convert('RGB').tobytes():
                            raise ValueError('Input frame is not the native rendered image')
                        x, y = (10, 276)[i % 2], (10, 276)[i // 2]
                        if preview.crop((x, y, x + 256, y + 256)).convert('RGB').tobytes() != frame.convert('RGB').tobytes():
                            raise ValueError('Four-tile preview differs from native inputs')
                for row, role in enumerate(('reference', 'candidate')):
                    for col, state in enumerate(('rest', 'mid', 'limit')):
                        with Image.open(out / f'native/{cid}/{role}_{state}.png') as native:
                            x, y = 114 + 266 * col, 98 + 266 * row
                            if board.crop((x, y, x + 256, y + 256)).convert('RGB').tobytes() != native.convert('RGB').tobytes():
                                raise ValueError('Review board differs from native rest/mid/limit frames')
        if public[pair[0]['case_id']]['view'] != public[pair[1]['case_id']]['view']:
            raise ValueError('Clean and defect use different cameras')
    record_paths = list((out / 'private/objects').glob('*/record.json')) + list((out / 'private/rejections').glob('*.json'))
    records = [json.loads(path.read_text()) for path in record_paths]
    accepted_records = [record for record in records if record['status'] == 'automatic_pass']
    if len(accepted_records) != 15:
        raise ValueError('Missing per-object atomic provenance records')
    for record in records:
        if record['status'] == 'automatic_pass':
            if record['source_authenticity']['passed'] is not True:
                raise ValueError('Source authenticity did not pass')
            pair = groups[(record['config'], record['asset_id'])]
            for key, aggregate in (('public', public), ('gold', gold), ('checks', checks), ('reviews', reviews)):
                if len(record[key]) != 2 or any(row != aggregate[row['case_id']] for row in record[key]):
                    raise ValueError('Atomic record differs from aggregate manifests')
            clean = next(g for g in pair if g['kind'] == 'clean')
            if record['source_authenticity']['program_sha'] != clean['reference_sha']:
                raise ValueError('Source authenticity refers to a different reference')
            if record['camera']['policy'] != 'normal-only frozen before corruption; no error-guided view retry':
                raise ValueError('Camera selection does not declare the normal-only freeze')
            for g in pair:
                if (record['camera']['view'] != public[g['case_id']]['view'] or
                        record['camera']['state_values'] != g['state_values']):
                    raise ValueError('Review state/camera differs from frozen normal configuration')
                if g['source']['split'] != 'train':
                    raise ValueError('Non-Train source in pilot')
                source = g['source']
                if (sha256_path(Path(source['source_archive'])) != source['archive_sha256'] or
                        sha256_path(Path(source['source_spec'])) != source['original_spec_sha256']):
                    raise ValueError('Source archive/spec identity mismatch')
                original = json.loads(Path(source['source_spec']).read_text())
                reference = json.loads((out / f"private/programs/{clean['case_id']}.json").read_text())
                if original != reference:
                    raise ValueError('Clean reference differs from original source specification')
        for name, digest in record['files'].items():
            if sha256_path(out / name) != digest:
                raise ValueError('Atomic object artifact SHA mismatch')
    pages = list(out.glob('*.html'))
    if not pages:
        raise ValueError('Missing blind review page')
    for page in pages:
        html = page.read_text()
        if any(token in html for token in (*config_kind, 'kinematics.', 'structure.part_offset', 'href="private/', 'src="private/', *{g['asset_id'] for g in gold.values()})):
            raise ValueError('Blind page exposes private answer or source fields')
        visible_ids = {cid for cid in public if cid in html}
        if page.name == 'review.html' and visible_ids != set(public):
            raise ValueError('Blind page omits review cases')
        if page.name == 'review-spot.html' and len(visible_ids) != 5:
            raise ValueError('Blind spot review must contain five distinct cases')
    if not (out / 'review.html').is_file():
        raise ValueError('Missing full blind review page')
    return {'status': 'automatic_checks_passed_pending_human_review', 'objects': 15,
            'cases': 30, 'pending_human_review': 30, 'training_eligible_cases': 0}


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('benchmark/medium-pilot-v1'))
    print(json.dumps(validate(parser.parse_args().out), indent=2))
