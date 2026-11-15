"""Freeze all 109 RP29 Validation defects and Train-only pixel controls.

Never reads Test rows. Private labels remain in private artifacts, not prompts.
Native recipes are restored from frozen supplementary manifests or authored plans.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from turnitover.repair.pixel_calibration import calibrate_pixel_threshold
from turnitover.repair.native_scopes import construction_scopes

RELEASE_SHA = 'ba59783d06e938b0ae5bc66e3c6b467ecca2a1615e56ea3cb85d69030ed7009a'
STATES = ('zero', 'mid', 'limit')
ALIASES = ({'rest','reset','zero','native_zero','time0','0.0','zero_reset'},
           {'mid','native_mid','time1','0.5'},
           {'endpoint','limit','native_end','native_limit','time2','1.0'})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def copy_checked(source, expected, target):
    if sha(source) != expected:
        raise ValueError(f'Frozen file hash mismatch: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return sha(target)


def phase_command(commands, index, joint=None):
    selected = [value for name, value in commands if name in ALIASES[index]]
    if len(selected) != 1:
        raise ValueError('Exactly one authored phase command required')
    return selected[0] if isinstance(selected[0], dict) else {joint: selected[0]}


def recipe(release, annotation, row, resource):
    """Recover saved phase names and role-specific numeric commands, never infer."""
    native = release / 'supplementary-observations' / row['episode_id'] / 'manifest.private.json'
    if native.exists():
        doc = read(native)
        if doc['status'] != 'complete':
            raise ValueError('Incomplete supplementary recipe')
        return {key: doc.get(key) for key in ('framing','commands','time_sequence')}, [native]
    folder = Path(annotation['source_roster_row']['report_path']).parent
    plan_path = folder / 'observation-plan.private.json'
    provenance = [Path(annotation['source_roster_row']['report_path'])]
    if not plan_path.exists() and (folder / 'public-task.json').exists():
        # The balanced anchor report retained its original program/plan lineage.
        prereg_path = folder.parent / 'preregistration.private.json'
        prereg = read(prereg_path)
        original = next(x for x in prereg['plans'] if x['case_id'] == row['episode_id'])
        plan_path = Path(original['report_path']).parent / 'observation-plan.private.json'
        provenance.append(prereg_path)
    if plan_path.exists():
        plan = read(plan_path)
        commands = plan['commands']
        by_role = {role: [[state, phase_command(commands[role] if isinstance(commands, dict) else commands,
                                               i, plan.get('joint'))]
                          for i, state in enumerate(STATES)] for role in ('reference','candidate')}
        # All source image bytes must bind to the recorded phase/view.
        for i, state in enumerate(STATES):
            candidates = [p for p in resource['image_pairs'] if p.get('state') in ALIASES[i]]
            match = next((p for p in candidates if p['reference_sha256'] == row['image_sha256'][2*i]
                          and p['candidate_sha256'] == row['image_sha256'][2*i+1]), None)
            if match is None:
                raise ValueError('Cannot bind exported images to authored phase')
        return {'framing': plan['framing'], 'commands': by_role['reference'],
                'commands_by_role': by_role, 'time_sequence': None,
                'role_command_policy': 'Replay original authored commands per arm; range cases can differ.'}, provenance + [plan_path]
    # Temporal source explicitly records seconds 0,.5,1; no joint set is applied.
    report = read(provenance[0])
    frame_path = folder / 'offline-framing.private.json'
    contract = report.get('controlled_reference_contract', '')
    if frame_path.exists() and 'seconds0,.5,1' in contract:
        return {'framing': read(frame_path)['framing'], 'commands': [[s, {}] for s in STATES],
                'time_sequence': [0.0, 0.5, 1.0], 'time_policy': contract}, provenance + [frame_path]
    raise ValueError(f'Cannot recover authored recipe: {row["episode_id"]}')


def prepare(release, dataset, output):
    release, dataset, output = [Path(p).resolve() for p in (release, dataset, output)]
    manifest = read(dataset / 'manifest.json')
    if (sha(release / 'export/manifest.json') != RELEASE_SHA
            or manifest['source_manifest_sha256'] != RELEASE_SHA
            or manifest['source_release'] != 'three-tier-900-rp29-v2.1-pre.1'):
        raise ValueError('Expected hash-frozen RP29 v2.1-pre.1 release')
    partitions = {}
    for split, count in [('train', 600), ('validation', 120)]:
        path = dataset / f'{split}.episodes.jsonl'
        if sha(path) != manifest['files'][path.name]:
            raise ValueError('Episode hash mismatch')
        data = rows(path)
        if len(data) != count or len({r['episode_id'] for r in data}) != count:
            raise ValueError('Unique frozen count mismatch')
        for row in data:
            if manifest['groups'][row['data_identity']['split_group']] != split:
                raise ValueError('Source split mismatch')
        partitions[split] = data
    validation = partitions['validation']
    status = lambda r: json.loads(r['messages'][-1]['content'])['verdict']['status']
    if Counter(map(status, validation)) != {'fail': 109, 'pass': 11}:
        raise ValueError('Expected exactly 109 defective and 11 clean Validation cases')
    annotations = {r['id']: r for r in rows(dataset / 'annotations.private.jsonl') if r['split'] == 'validation'}
    resources = {r['case_id']: r for r in rows(release / 'evidence-inventory/cases.private.jsonl')
                 if r['case_id'] in annotations}
    output.mkdir(parents=True, exist_ok=False)
    train_folder = output / 'train-calibration'
    train_folder.mkdir()
    train_cases = []
    for row in partitions['train']:
        pairs = []
        for i, state in enumerate(STATES):
            pair = {'state': state, 'sha256': {}}
            for offset, role in enumerate(('reference','candidate')):
                index = 2*i + offset
                relative = Path('images') / row['episode_id'] / f'{state}-{role}.png'
                pair[role] = str(relative)
                pair['sha256'][role] = copy_checked(dataset / row['images'][index], row['image_sha256'][index], train_folder / relative)
            pairs.append(pair)
        train_cases.append({'id': row['episode_id'], 'source_group': row['data_identity']['split_group'],
                            'defective': status(row) == 'fail', 'pairs': pairs, 'control_kind': 'native_case'})
    train_manifest = {'schema_version': 1, 'status': 'frozen', 'split': 'train',
                      'source_manifest_sha256': sha(dataset/'manifest.json'), 'cases': train_cases}
    write(train_folder / 'native-cases.manifest.json', train_manifest)
    controls = []
    for case in train_cases:
        controls.append({**case, 'id': case['id'] + '-reference-self', 'defective': False,
                         'control_kind': 'synthetic_reference_self', 'source_case_id': case['id'],
                         'pairs': [{**p, 'candidate': p['reference'],
                                    'sha256': {'reference': p['sha256']['reference'],
                                               'candidate': p['sha256']['reference']}} for p in case['pairs']]})
    write(train_folder / 'calibration.manifest.json', {**train_manifest, 'cases': train_cases + controls,
          'native_clean': 0, 'synthetic_clean': 600,
          'limitations': ['Synthetic reference-self controls only; no genuine clean Train examples or clean generalization calibration.']})
    calibration = calibrate_pixel_threshold(train_folder / 'calibration.manifest.json')
    calibration['counts'].update(native_clean=0, synthetic_clean=600)
    calibration['limitations'] = ['All 600 Train cases are defective; clean controls are reference-self synthetic images.',
                                  'This calibration does not establish genuine clean-example generalization.']
    write(train_folder / 'pixel-calibration.json', calibration)
    cases, private = [], []
    for row in validation:
        if status(row) != 'fail':
            continue
        uid = row['episode_id']; annotation = annotations[uid]; resource = resources[uid]
        folder = output / 'cases' / uid; folder.mkdir(parents=True)
        case = {'id': uid, 'source_group': row['data_identity']['split_group'], 'sha256': {}}
        for role, key, name in [('reference','reference_program','reference.private.ts'),('candidate','initial_program','initial.ts')]:
            item = annotation['programs']['case_' + role]
            if item['sha256'][:16] != row['data_identity'][role + '_sha']:
                raise ValueError('Program identity mismatch')
            case[key] = name
            case['sha256'][key] = copy_checked(item['path'], item['sha256'], folder/name)
        for index, (image, digest) in enumerate(zip(row['images'], row['image_sha256'], strict=True)):
            target = folder / f'{STATES[index//2]}-{("reference", "candidate")[index%2]}.png'
            copy_checked(dataset/image, digest, target)
        case['reference_image'] = 'zero-reference.png'
        case['sha256']['reference_image'] = sha(folder/case['reference_image'])
        public = json.loads(row['messages'][0]['content'].rsplit('\n', 1)[-1])
        runtime = [h for h in public['history'] if h['action']['type'] == 'query_runtime']
        case['task_runtime_requirements'] = [h for h in runtime if h.get('payload',{}).get('measurement_kind') == 'public_task_budget']
        case['public_task'] = public['context']['task']
        # Public ABI IDs and budgets only. Old verdicts/runtime never become fresh feedback.
        case['public_context'] = public['context']
        write(folder/'initial-runtime.public.json', runtime)
        case['initial_runtime'] = 'initial-runtime.public.json'
        case['sha256']['initial_runtime'] = sha(folder/case['initial_runtime'])
        recovered, provenance = recipe(release, annotation, row, resource)
        case.update(recovered, view=public['history'][0]['action']['view_id'])
        # Original action order is necessary for view/history-triggered defects.
        # Store only actions and state/time coordinates, never saved payloads,
        # labels or verdicts. The private auditor re-executes these observations.
        case['authored_audit_history'] = [{key: observation[key] for key in ('action', 'state', 'time')
                                           if key in observation} for observation in public['history']]
        by_role = recovered.get('commands_by_role', {})
        if (by_role.get('reference') != by_role.get('candidate')
                and any(label['defect_id'] == 'kinematics.joint_range' for label in annotation['labels'])):
            # The paper's states are semantic detents. Reusing the initial wrong
            # range after a repair would reject even a correctly repaired joint.
            case['replay_command_policy'] = 'current_joint_limits'
        case['recipe_provenance'] = [{'path': str(p), 'sha256': sha(p)} for p in provenance]
        case['runtime_policy'] = 'Fresh native replay after edits; initial runtime is archived only, never reused as candidate feedback.'
        metadata = {'schema_version': 1, 'status': 'frozen', 'split': 'validation',
                    'benchmark': 'RP29 v2.1-pre.1', 'source_manifest_sha256': sha(dataset/'manifest.json'), 'cases': [case]}
        write(folder/'manifest.json', metadata)
        cases.append({'id': uid, 'manifest': str((folder/'manifest.json').relative_to(output)),
                      'manifest_sha256': sha(folder/'manifest.json')})
        scoring = construction_scopes(annotation, case['sha256']['reference_program'])
        private.append({'id': uid, 'labels': scoring['labels'], 'evidence': row['evidence'],
                        'scoring_scopes': scoring['scopes'],
                        'scoring_scope_provenance': scoring['provenance'],
                        'history': public['history'],
                        'recipe_provenance': case['recipe_provenance']})
    (output/'labels.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in private))
    result = {'schema_version': 1, 'status': 'frozen', 'benchmark': 'RP29 v2.1-pre.1', 'split': 'validation',
              'cases': cases, 'counts': {'validation_total':120,'defective':109,'excluded_clean':11},
              'source_manifest_sha256': sha(dataset/'manifest.json'), 'published_manifest_sha256': RELEASE_SHA,
              'pixel_calibration': 'train-calibration/pixel-calibration.json',
              'pixel_calibration_sha256': sha(train_folder/'pixel-calibration.json'),
              'private_labels': 'labels.private.jsonl',
              'private_labels_sha256': sha(output/'labels.private.jsonl'), 'test_opened': False}
    write(output/'manifest.json', result)
    return result


def refresh_scoring_scopes(frozen, dataset, output):
    """Write a sibling manifest revision; preserve frozen observations/results."""
    from scripts.run_rp29_repair_batch import load_cases
    from turnitover.repair.frozen_manifest import _file

    frozen, dataset, output = [Path(p).resolve() for p in (frozen, dataset, output)]
    if output.parent != frozen.parent or output == frozen:
        raise ValueError('Scoring revision must be a new sibling manifest')
    labels_path = output.with_name(output.stem + '.labels.private.jsonl')
    if output.exists() or labels_path.exists():
        raise FileExistsError('Scoring revision already exists')
    metadata, loaded = load_cases(frozen)
    source = read(dataset / 'manifest.json')
    if sha(dataset / 'manifest.json') != metadata['source_manifest_sha256']:
        raise ValueError('Scoring revision dataset identity mismatch')
    annotations_path = _file(dataset, 'annotations.private.jsonl',
                             source['files']['annotations.private.jsonl'], 'annotations')
    ids = {case['id'] for _, case in loaded}
    annotations = {row['id']: row for row in rows(annotations_path)
                   if row['split'] == 'validation' and row['id'] in ids}
    if set(annotations) != ids:
        raise ValueError('Scoring revision annotation roster mismatch')
    old_labels = _file(frozen.parent, metadata['private_labels'],
                       metadata['private_labels_sha256'], 'private labels')
    cases = {case['id']: case for _, case in loaded}
    revised = []
    for row in rows(old_labels):
        case = cases[row['id']]
        annotation = annotations[row['id']]
        for role, key in (('reference', 'reference_program'), ('candidate', 'initial_program')):
            if annotation['programs']['case_' + role]['sha256'] != case['sha256'][key]:
                raise ValueError('Scoring revision program identity mismatch')
        recovered = construction_scopes(annotation, case['sha256']['reference_program'])
        labels = recovered['labels']
        if [r['defect_id'] for r in labels] != [r['defect_id'] for r in row['labels']]:
            raise ValueError('Scoring revision cannot change defect instances')
        for old, new in zip(row['labels'], labels, strict=True):
            for key in ('parts', 'joint_id'):
                if not old.get(key) and new.get(key):
                    old[key] = new[key]
        row['scoring_scopes'] = [new if new is not None else old
            for old, new in zip(row.get('scoring_scopes') or [None] * len(labels),
                                recovered['scopes'], strict=True)]
        row['scoring_scope_provenance'] = recovered['provenance']
        revised.append(row)
    with labels_path.open('x', encoding='utf-8') as handle:
        handle.write(''.join(json.dumps(row) + '\n' for row in revised))
    result = {**metadata, 'private_labels': labels_path.name,
              'private_labels_sha256': sha(labels_path),
              'scoring_revision_of_sha256': sha(frozen)}
    with output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    load_cases(output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--refresh-scoring-scopes-from', type=Path,
                        help='Create a new sibling manifest with refreshed private identifiers only')
    for name in ('dataset','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.refresh_scoring_scopes_from:
        result = refresh_scoring_scopes(args.refresh_scoring_scopes_from, args.dataset, args.output)
    else:
        if args.release is None:
            parser.error('--release is required when preparing a new batch')
        result = prepare(args.release, args.dataset, args.output)
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'cases': len(result['cases'])}))
