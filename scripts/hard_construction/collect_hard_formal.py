"""Construct seventy-five full-motion Hard candidates from the frozen source pool."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

from scripts.hard_construction.build_hard_diverse_calibration import (
    attempts, expected_field, measure)
from scripts.hard_construction.build_hard_full_motion_examples import view_pair
from scripts.hard_construction.hard_artifacts import page
from scripts.hard_construction.hard_full_motion_corruptions import changed_fields
from scripts.hard_construction.hard_geometry import normal_framing, spec_diagonal
from scripts.tier_construction.level0_expansion_search import inside_camera, render_states
from scripts.tier_construction.prepare_obvious_defect_review import DEFAULT_CONFIG, checker_result, sha256_path
from turnitover.assets.audit_urdf import audit_model
from turnitover.assets.catalog import decode_spec
from turnitover.assets.urdf import UrdfModel
from turnitover.config import load_generate_config
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views

ROOT = Path(__file__).resolve().parents[2]
POLICY = 'hard-full-motion-formal-v1'
KINDS = ('axis_direction_3deg', 'joint_anchor_offset', 'moving_part_initial_rotation')
STATES = ('rest', 'mid', 'limit')
TARGET_OBJECTS = 75
PER_KIND_TARGET = 25
SAMPLE_SEED = 2026110301
CATEGORY_CAPS = {
    'StorageFurniture': 50,
    'Table': 12,
    'Toilet': 10,
    'Lamp': 8,
}


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, value)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def category_cap(category: str) -> int:
    return CATEGORY_CAPS.get(category, 5)


def source_record(row: dict) -> dict | None:
    document = json.loads(Path(row['source_record']).read_text())
    return document if 'metadata' in document else None


def load_records(out: Path) -> list[dict]:
    paths = [*out.glob('private/objects/*/record.json'), *out.glob('private/rejections/*/*.json')]
    return [json.loads(path.read_text()) for path in sorted(paths)]


def state(pool_path: Path, target: int) -> dict:
    cfg = load_generate_config(DEFAULT_CONFIG)
    inputs = [Path(__file__), Path(__file__).with_name('hard_full_motion_corruptions.py'),
        Path(__file__).with_name('build_hard_diverse_calibration.py'),
        Path(__file__).with_name('build_hard_full_motion_examples.py'),
        pool_path, pool_path.with_name('manifest.json'), Path(DEFAULT_CONFIG), cfg.views_path]
    return {'version': 1, 'policy': POLICY, 'target_objects': target,
        'per_kind_target': target // len(KINDS), 'sample_seed': SAMPLE_SEED,
        'motion_policy': 'rest_to_half_then_full_native_endpoint',
        'category_caps': CATEGORY_CAPS,
        'inputs': {str(path.resolve()): sha256_path(path) for path in inputs}}


def target_values(target: dict) -> dict:
    limits = [float(value) for value in target['original_limits']]
    endpoint = max(limits, key=abs)
    if abs(endpoint) <= 1e-8:
        raise ValueError('Target has no nonzero native endpoint')
    if target['joint_type'] == 'revolute' and abs(endpoint) > 3.141592653589793 + 1e-8:
        raise ValueError('Revolute native endpoint exceeds 180 degrees')
    return {'rest': 0., 'mid': endpoint / 2, 'limit': endpoint}


def eligible_targets(row: dict, kind: str) -> list[dict]:
    targets = row['eligible_targets']
    if kind == 'joint_anchor_offset':
        targets = [target for target in targets if target['joint_type'] == 'revolute']
    return [target for target in targets if not (target['joint_type'] == 'revolute'
        and max(abs(float(value)) for value in target['original_limits']) > 3.141592653589793 + 1e-8)]


def choose(pool: list[dict], records: list[dict], target: int) -> tuple[str, dict, dict] | None:
    accepted = [row for row in records if row['status'] == 'automatic_pass']
    type_counts = Counter(row['kind'] for row in accepted)
    category_counts = Counter(row['category'] for row in accepted)
    used = {row['asset_id'] for row in accepted}
    attempted = {(row['asset_id'], row['kind'], row['joint_id']) for row in records}
    quota = target // len(KINDS)
    choices = []
    for kind in KINDS:
        if type_counts[kind] >= quota:
            continue
        for row in pool:
            if row['asset_id'] in used or category_counts[row['category']] >= category_cap(row['category']):
                continue
            for target_row in eligible_targets(row, kind):
                key = (row['asset_id'], kind, target_row['joint_id'])
                if key in attempted:
                    continue
                priority = hashlib.sha256(
                    f'{POLICY}:{kind}:{row["asset_id"]}:{target_row["joint_id"]}'.encode()).hexdigest()
                choices.append((type_counts[kind], row['category'] == 'StorageFurniture',
                    category_counts[row['category']], priority, kind, row, target_row))
    if not choices:
        return None
    *_, kind, row, target_row = min(choices)
    return kind, row, target_row


def reject(out: Path, row: dict, target: dict, kind: str, reason: str) -> dict:
    record = {'status': 'rejected', 'asset_id': row['asset_id'], 'category': row['category'],
        'kind': kind, 'joint_id': target['joint_id'], 'part_id': target['part_id'], 'reason': reason}
    write(out / f'private/rejections/{row["asset_id"]}/{kind}-{target["joint_id"]}.json', record)
    return record


def collect_one(session, views_tuple, views, out: Path, row: dict, target: dict, kind: str) -> dict:
    record = source_record(row)
    if record is None:
        return reject(out, row, target, kind, 'historical_source_record_not_atomic')
    spec = decode_spec(json.loads(Path(row['source_spec']).read_text()))
    reference = ObjectProgram.from_spec(spec)
    try:
        values = target_values(target)
    except ValueError as error:
        return reject(out, row, target, kind, str(error))
    seed_render = render_states(session, reference, None, target['joint_id'], values, ['front'])
    framing = normal_framing(seed_render, 1.15)
    normal = render_states(session, reference, framing, target['joint_id'], values, list(views))
    try:
        initial, supplementary = view_pair(normal, framing, views, target['part_id'])
    except ValueError as error:
        return reject(out, row, target, kind, str(error))

    viable = []
    diagonal = spec_diagonal(to_dict(spec))
    for variant, _, _, candidate, parameters in attempts(reference, target, kind, diagonal):
        if changed_fields(reference, candidate) != [expected_field(target, kind)]:
            raise ValueError('Construction changed more than one declared field')
        rendered = render_states(session, candidate, framing, target['joint_id'], values,
                                 [initial, supplementary])
        if any(not inside_camera(rendered, framing, views[view])[0]
               for view in (initial, supplementary)):
            continue
        check = checker_result(candidate, rendered, reference, normal)
        metrics = measure(normal, rendered, framing, [initial, supplementary], views,
                          target['part_id'], kind)
        if check['code_correct'] or not check['checks'][1]['passed'] or not metrics['automatic_visible_not_extreme']:
            continue
        failed = [invariant for invariant in check['checks'][0]['invariants'] if not invariant['passed']]
        if not any(target['part_id'] in invariant['parts'] for invariant in failed):
            continue
        viable.append((abs(metrics['maximum_target_projected_rms_px'] - 4), variant,
                       candidate, rendered, check, metrics, parameters))
    if not viable:
        return reject(out, row, target, kind,
            'no_truthful_visible_not_extreme_candidate_at_frozen_strength')
    _, variant, candidate, rendered, check, metrics, parameters = min(viable, key=lambda item: item[0])
    clean_check = checker_result(reference, normal, reference, normal)
    authenticity = audit_model(session, UrdfModel(Path(row['source_urdf'])), spec)
    if not clean_check['code_correct'] or authenticity['passed'] is not True:
        return reject(out, row, target, kind, 'clean_control_or_source_authenticity_failed')

    case_id = 'hp-' + hashlib.sha256(
        f'{POLICY}:{kind}:{row["asset_id"]}:{target["joint_id"]}:{variant}'.encode()).hexdigest()[:14]
    public = {'case_id': case_id, 'initial_view': initial, 'supplementary_view': supplementary,
        'states': list(STATES), 'native_root': f'native/{case_id}', 'image_size': [256, 256],
        'status': 'pending_human_review'}
    source = {**record['metadata'], 'source_urdf': row['source_urdf'],
              'complexity': {'active_joints': row['active_joints'], 'parts': row['parts'],
                             'category': row['category']}, 'split': 'train'}
    gold = {'case_id': case_id, 'asset_id': row['asset_id'], 'kind': kind,
        'joint_id': target['joint_id'], 'part_id': target['part_id'],
        'semantic_role': target['semantic_role'], 'parameters': parameters,
        'original_limits': target['original_limits'], 'state_values': values,
        'motion_policy': 'rest_to_half_then_full_native_endpoint',
        'reference_sha': reference.sha, 'candidate_sha': candidate.sha,
        'initial_view': initial, 'supplementary_view': supplementary, 'metrics': metrics,
        'source': source, 'training_eligibility': 'pending_human_review'}
    checks = {'case_id': case_id, **check, 'clean_check': clean_check,
              'source_authenticity': authenticity}
    review = {'case_id': case_id, 'status': 'pending_human_review', 'reviewer': None,
        'reviewed_at': None, 'human_judgment': {'initial_verdict': None, 'initial_notes': None,
        'final_decision': None, 'final_notes': None, 'supplementary_decisive': None,
        'wrong_part_description': None, 'observed_views': [], 'view_switches': []}}
    accepted = {'status': 'automatic_pass', 'case_id': case_id, 'asset_id': row['asset_id'],
        'category': row['category'], 'kind': kind, 'joint_id': target['joint_id'],
        'part_id': target['part_id'], 'semantic_role': target['semantic_role'],
        'active_joints': row['active_joints'], 'parts': row['parts'],
        'parameters': parameters, 'views': [initial, supplementary], 'metrics': metrics,
        'public': public, 'gold': gold, 'checks': checks, 'review': review, 'files': {}}

    with tempfile.TemporaryDirectory(prefix='.hard-formal-', dir=out.parent) as temporary:
        stage = Path(temporary)
        for view in (initial, supplementary):
            for role, frames in (('reference', normal), ('candidate', rendered)):
                for state_name in STATES:
                    path = stage / f'native/{case_id}/{view}/{role}_{state_name}.png'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(frames['frames'][state_name][view])
        program_path = stage / f'private/programs/{case_id}.json'
        write(program_path, to_dict(candidate.spec))
        gold['program'] = f'private/programs/{case_id}.json'
        for path in sorted(stage.rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(stage)
            destination = out / relative
            if destination.exists():
                raise ValueError('Formal Hard artifact collision: ' + str(relative))
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
            accepted['files'][relative.as_posix()] = sha256_path(destination)
    write(out / f'private/objects/{row["asset_id"]}/record.json', accepted)
    return accepted


def finalize(out: Path, run_state: dict, target: int) -> dict:
    records = load_records(out)
    accepted = sorted([row for row in records if row['status'] == 'automatic_pass'],
                      key=lambda row: row['case_id'])
    public = [row['public'] for row in accepted]
    gold = [row['gold'] for row in accepted]
    checks = [row['checks'] for row in accepted]
    reviews = [row['review'] for row in accepted]
    rng = __import__('numpy').random.default_rng(SAMPLE_SEED)
    sample_indices = sorted(int(index) for index in rng.choice(len(accepted), size=min(10, len(accepted)), replace=False))
    sample = [public[index] for index in sample_indices]
    write_rows(out / 'pool_manifest.jsonl', public)
    write_rows(out / 'review_manifest.jsonl', sample)
    write_rows(out / 'private/gold.jsonl', gold)
    write_rows(out / 'private/checks.jsonl', checks)
    write_rows(out / 'human_review.jsonl', [reviews[index] for index in sample_indices])
    write_rows(out / 'train_manifest.jsonl', [])
    page(out, public, 'review-all.html')
    page(out, sample, 'review.html')
    write(out / 'sample.json', {'seed': SAMPLE_SEED, 'case_ids': [row['case_id'] for row in sample],
        'method': 'uniform without replacement over independent accepted objects',
        'human_review_status': 'pending'})
    counts = Counter(row['kind'] for row in accepted)
    categories = Counter(row['category'] for row in accepted)
    complete = len(accepted) == target and counts == Counter({kind: target // len(KINDS) for kind in KINDS})
    report = {'status': 'ready_pending_human_review' if complete else 'collection_incomplete',
        'objects': len(accepted), 'target_objects': target, 'cases': len(public),
        'corruption_counts': dict(counts), 'category_counts': dict(categories),
        'attempts': len(records), 'rejections': len(records) - len(accepted),
        'human_review_sample': len(sample), 'human_review_complete': False,
        'training_eligible_cases': 0}
    write(out / 'validation.json', report)
    files = {path.relative_to(out).as_posix(): sha256_path(path) for path in sorted(out.rglob('*'))
             if path.is_file() and path.name != 'manifest.json'}
    write(out / 'manifest.json', {**run_state, **report, 'files': files})
    return report


def run(pool_path: Path, out: Path, target: int = TARGET_OBJECTS, limit: int = 1000) -> dict:
    pool_path, out = pool_path.resolve(), out.resolve()
    if target <= 0 or target % len(KINDS) or limit < 0:
        raise ValueError('Target must be positive and divisible across corruption types')
    run_state = state(pool_path, target)
    state_path = out / 'private/run_state.json'
    if state_path.exists() and json.loads(state_path.read_text()) != run_state:
        raise ValueError('Formal Hard run inputs changed; use a new output directory')
    if out.exists() and any(out.iterdir()) and not state_path.exists():
        raise ValueError('Existing Formal Hard output has no resumable state')
    out.mkdir(parents=True, exist_ok=True)
    write(state_path, run_state)
    pool = json.loads(pool_path.read_text())
    attempts_made = 0
    cfg = load_generate_config(DEFAULT_CONFIG)
    views_tuple = load_views(cfg.views_path)
    views = {str(view.id): view for view in views_tuple}
    with ObservationSession(cfg.render, views_tuple) as session:
        while attempts_made < limit:
            records = load_records(out)
            if len([row for row in records if row['status'] == 'automatic_pass']) >= target:
                break
            choice = choose(pool, records, target)
            if choice is None:
                break
            kind, row, target_row = choice
            print(json.dumps({'event': 'start', 'asset_id': row['asset_id'], 'category': row['category'],
                'kind': kind, 'joint_id': target_row['joint_id']}), flush=True)
            result = collect_one(session, views_tuple, views, out, row, target_row, kind)
            attempts_made += 1
            current = len([item for item in load_records(out) if item['status'] == 'automatic_pass'])
            print(json.dumps({'event': 'result', 'asset_id': row['asset_id'], 'kind': kind,
                'status': result['status'], 'objects': current}), flush=True)
    for path, digest in run_state['inputs'].items():
        if sha256_path(Path(path)) != digest:
            raise ValueError('Formal Hard input changed during collection: ' + path)
    return finalize(out, run_state, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path,
        default=ROOT / 'output/hard-candidate-pool-v2/final/pool.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'output/hard-formal-v1')
    parser.add_argument('--target', type=int, default=TARGET_OBJECTS)
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args()
    result = run(args.pool, args.out, args.target, args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['objects'] == args.target else 2)
