"""Build a complex-object, full-motion calibration pool and sample five cases."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.hard_construction.build_hard_full_motion_examples import pixel_fraction, view_pair
from scripts.hard_construction.hard_artifacts import page
from scripts.hard_construction.hard_full_motion_corruptions import (
    changed_fields, cone_directions, joint_anchor, part_rotation,
    part_translation, unit_directions)
from scripts.hard_construction.hard_geometry import normal_framing, project_vertices, spec_diagonal
from scripts.tier_construction.level0_expansion_search import inside_camera, render_states
from scripts.tier_construction.prepare_obvious_defect_review import DEFAULT_CONFIG, checker_result, sha256_path
from turnitover.assets.audit_urdf import audit_model
from turnitover.assets.catalog import decode_spec
from turnitover.assets.urdf import UrdfModel
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views

ROOT = Path(__file__).resolve().parents[2]
POLICY = 'hard-diverse-full-motion-v3'
STATES = ('rest', 'mid', 'limit')
KINDS = ('axis_direction_3deg', 'joint_anchor_offset',
         'moving_part_initial_translation', 'moving_part_initial_rotation')
POOL_PER_KIND = 3
SAMPLE_SEED = 2026102901
SEARCH_EXCLUSIONS = {
    ('joint_anchor_offset', 'PM_7179'): 'no visible but non-extreme 0.5% anchor-offset variant',
    ('moving_part_initial_rotation', 'PM_49188'):
        'selected readable views collapse the multi-joint object to a simple silhouette',
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def attempts(reference, target, kind, diagonal):
    if kind == 'axis_direction_3deg':
        for index, axis in enumerate(cone_directions(target['original_axis'], 3)):
            yield index, reference.spec.joint(target['joint_id']).axis, axis, joint_axis_program(reference, target, axis), {
                'axis_angle_deg': 3., 'original_axis': target['original_axis'], 'candidate_axis': list(axis)}
    elif kind in ('joint_anchor_offset', 'moving_part_initial_translation'):
        fraction = .005 if kind == 'joint_anchor_offset' else .01
        magnitude = diagonal * fraction
        for index, direction in enumerate(unit_directions()):
            vector = tuple(float(magnitude * v) for v in direction)
            if kind == 'joint_anchor_offset':
                candidate = joint_anchor(reference, target['joint_id'], vector)
                params = {'anchor_offset_fraction': .005, 'anchor_offset_m': magnitude, 'vector': list(vector)}
            else:
                candidate = part_translation(reference, target['part_id'], vector)
                params = {'initial_translation_fraction': fraction, 'translation_m': magnitude, 'vector': list(vector)}
            yield index, None, None, candidate, params
    else:
        for index, axis in enumerate(unit_directions()):
            candidate = part_rotation(reference, target['part_id'], axis, 2)
            yield index, None, None, candidate, {
                'initial_rotation_deg': 2., 'local_rotation_axis': list(axis)}


def joint_axis_program(reference, target, axis):
    from scripts.hard_construction.hard_full_motion_corruptions import joint_axis
    return joint_axis(reference, target['joint_id'], axis)


def expected_field(target, kind):
    return {
        'axis_direction_3deg': f"joints.{target['joint_id']}.axis",
        'joint_anchor_offset': f"joints.{target['joint_id']}.anchor",
        'moving_part_initial_translation': f"parts.{target['part_id']}.origin",
        'moving_part_initial_rotation': f"parts.{target['part_id']}.rotation",
    }[kind]


def measure(normal, rendered, framing, view_ids, view_map, part, kind):
    result = {'per_view': {}}
    relevant = ('mid', 'limit') if kind in ('axis_direction_3deg', 'joint_anchor_offset') else STATES
    for view in view_ids:
        states = {}
        for state in STATES:
            before, after = normal['geometry'][state][part].vertices, rendered['geometry'][state][part].vertices
            delta = project_vertices(before, framing, view_map[view]) - project_vertices(after, framing, view_map[view])
            states[state] = {
                'target_projected_rms_px': float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))),
                'changed_fraction': pixel_fraction(normal['frames'][state][view], rendered['frames'][state][view]),
            }
        result['per_view'][view] = states
    supported = [state for state in relevant if
                 result['per_view'][view_ids[-1]][state]['target_projected_rms_px'] >= 1.25
                 and result['per_view'][view_ids[-1]][state]['changed_fraction'] >= .0003]
    maximum = max(result['per_view'][view][state]['target_projected_rms_px']
                  for view in view_ids for state in relevant)
    max_fraction = max(result['per_view'][view][state]['changed_fraction']
                       for view in view_ids for state in relevant)
    rest_max = max(result['per_view'][view]['rest']['target_projected_rms_px'] for view in view_ids)
    result.update(relevant_states=list(relevant), supported_states=supported,
                  maximum_target_projected_rms_px=maximum,
                  maximum_changed_fraction=max_fraction, rest_error_px=rest_max)
    valid_rest = rest_max <= 1e-5 if kind in ('axis_direction_3deg', 'joint_anchor_offset') else rest_max >= 1.25
    result['automatic_visible_not_extreme'] = bool(supported and valid_rest and maximum <= 10 and max_fraction <= .08)
    return result


def select_targets(document):
    by_id = {row['asset_id']: row for row in document['targets']}
    return [(kind, by_id[asset_id]) for kind in KINDS
            for asset_id in document['fallback_by_corruption'][kind]
            if (kind, asset_id) not in SEARCH_EXCLUSIONS]


def sample_cases(accepted):
    rng = np.random.default_rng(SAMPLE_SEED)
    chosen = []
    for kind in KINDS:
        group = [row for row in accepted if row['kind'] == kind]
        chosen.append(group[int(rng.integers(len(group)))])
    remaining = [row for row in accepted if row not in chosen]
    chosen.append(remaining[int(rng.integers(len(remaining)))])
    rng.shuffle(chosen)
    return chosen


def build(target_pool, out):
    target_pool, out = target_pool.resolve(), out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Calibration bundle output must be new and empty')
    targets_document = json.loads(target_pool.read_text())
    selections = select_targets(targets_document)
    cfg = load_generate_config(DEFAULT_CONFIG)
    views_tuple = load_views(cfg.views_path)
    views = {str(v.id): v for v in views_tuple}
    accepted, public, gold, checks, reviews, rejections = [], [], [], [], [], []
    accepted_counts, used_assets = Counter(), set()
    with ObservationSession(cfg.render, views_tuple) as session:
        for kind, target in selections:
            if accepted_counts[kind] >= POOL_PER_KIND or target['asset_id'] in used_assets:
                continue
            if target['complexity']['active_joints'] < 3 or target['complexity']['parts'] < 5:
                raise ValueError('Target is not a complex multi-joint object')
            spec = decode_spec(json.loads(Path(target['source_spec']).read_text()))
            reference = ObjectProgram.from_spec(spec)
            endpoint = target['native_endpoint_rad']
            values = {'rest': 0., 'mid': float(endpoint / 2), 'limit': float(endpoint)}
            initial_render = render_states(session, reference, None, target['joint_id'], values, ['front'])
            framing = normal_framing(initial_render, 1.15)
            normal = render_states(session, reference, framing, target['joint_id'], values, list(views))
            try:
                initial, supplementary = view_pair(normal, framing, views, target['part_id'])
            except ValueError as exc:
                rejections.append({'kind': kind, 'asset_id': target['asset_id'],
                    'reason': str(exc)})
                continue
            viable = []
            diagonal = spec_diagonal(to_dict(spec))
            for variant, _, _, candidate, params in attempts(reference, target, kind, diagonal):
                if changed_fields(reference, candidate) != [expected_field(target, kind)]:
                    raise ValueError('Construction changed more than its single declared field')
                rendered = render_states(session, candidate, framing, target['joint_id'], values,
                                         [initial, supplementary])
                if any(not inside_camera(rendered, framing, views[v])[0] for v in (initial, supplementary)):
                    continue
                check = checker_result(candidate, rendered, reference, normal)
                metrics = measure(normal, rendered, framing, [initial, supplementary], views,
                                  target['part_id'], kind)
                if check['code_correct'] or not check['checks'][1]['passed'] or not metrics['automatic_visible_not_extreme']:
                    continue
                failed = [inv for inv in check['checks'][0]['invariants'] if not inv['passed']]
                if not any(target['part_id'] in inv['parts'] for inv in failed):
                    continue
                score = abs(metrics['maximum_target_projected_rms_px'] - 4)
                viable.append((score, variant, candidate, rendered, check, metrics, params))
                if score <= .5:
                    break
            if not viable:
                rejections.append({'kind': kind, 'asset_id': target['asset_id'],
                    'reason': 'no_truthful_visible_not_extreme_candidate_at_frozen_strength'})
                continue
            _, variant, candidate, rendered, check, metrics, params = min(viable, key=lambda row: row[0])
            clean_check = checker_result(reference, normal, reference, normal)
            source_record = json.loads(Path(target['source_record']).read_text())
            authenticity = audit_model(session, UrdfModel(Path(target['source_urdf'])), spec)
            if not clean_check['code_correct'] or authenticity['passed'] is not True:
                raise ValueError('Clean control or source authenticity failed')
            cid = 'hp-' + hashlib.sha256(
                f'{POLICY}:{kind}:{target["asset_id"]}:{target["joint_id"]}:{variant}'.encode()).hexdigest()[:14]
            for view in (initial, supplementary):
                for role, frames in (('reference', normal), ('candidate', rendered)):
                    for state in STATES:
                        path = out / f'native/{cid}/{view}/{role}_{state}.png'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(frames['frames'][state][view])
            program_path = f'private/programs/{cid}.json'
            write_json(out / program_path, to_dict(candidate.spec))
            public_row = {'case_id': cid, 'initial_view': initial, 'supplementary_view': supplementary,
                'states': list(STATES), 'native_root': f'native/{cid}', 'image_size': [256, 256],
                'status': 'pending_human_review'}
            gold_row = {'case_id': cid, 'asset_id': target['asset_id'], 'kind': kind,
                'joint_id': target['joint_id'], 'part_id': target['part_id'],
                'semantic_role': target['semantic_role'], 'parameters': params,
                'original_limits': target['original_limits'], 'state_values': values,
                'motion_policy': 'rest_to_half_then_full_native_endpoint',
                'program': program_path, 'reference_sha': reference.sha, 'candidate_sha': candidate.sha,
                'initial_view': initial, 'supplementary_view': supplementary, 'metrics': metrics,
                'complexity': target['complexity'], 'source': source_record['metadata'],
                'training_eligibility': 'calibration_only_pending_human_review'}
            check_row = {'case_id': cid, **check, 'clean_check': clean_check,
                         'source_authenticity': authenticity}
            review_row = {'case_id': cid, 'status': 'pending_human_review', 'reviewer': None,
                'reviewed_at': None, 'human_judgment': {'initial_verdict': None, 'initial_notes': None,
                'final_decision': None, 'final_notes': None, 'supplementary_decisive': None,
                'wrong_part_description': None, 'observed_views': [], 'view_switches': []}}
            public.append(public_row); gold.append(gold_row); checks.append(check_row); reviews.append(review_row)
            accepted.append({'case_id': cid, 'kind': kind, 'asset_id': target['asset_id'],
                'category': target['category'], 'semantic_role': target['semantic_role'],
                'active_joints': target['complexity']['active_joints'], 'parts': target['complexity']['parts'],
                'native_travel_deg': target['native_endpoint_deg'], 'parameters': params,
                'views': [initial, supplementary], 'metrics': metrics})
            accepted_counts[kind] += 1
            used_assets.add(target['asset_id'])
    if accepted_counts != Counter({kind: POOL_PER_KIND for kind in KINDS}):
        raise ValueError(f'Complex-object calibration pool incomplete: {dict(accepted_counts)}')
    sampled = sample_cases(accepted)
    sample_ids = {row['case_id'] for row in sampled}
    page(out, [row for row in public if row['case_id'] in sample_ids])
    write_rows(out / 'pool_manifest.jsonl', public)
    write_rows(out / 'review_manifest.jsonl', [row for row in public if row['case_id'] in sample_ids])
    write_rows(out / 'private/gold.jsonl', gold)
    write_rows(out / 'private/checks.jsonl', checks)
    write_rows(out / 'human_review.jsonl', [row for row in reviews if row['case_id'] in sample_ids])
    write_rows(out / 'train_manifest.jsonl', [])
    write_json(out / 'private/search_rejections.json', rejections)
    write_json(out / 'pool.json', accepted)
    write_json(out / 'sample.json', {'seed': SAMPLE_SEED,
        'method': 'one uniform sample per corruption, then one uniform sample from remaining objects; final order shuffled',
        'case_ids': [row['case_id'] for row in sampled], 'human_review_status': 'pending'})
    inputs = [Path(__file__), Path(__file__).with_name('hard_full_motion_corruptions.py'),
              target_pool, Path(DEFAULT_CONFIG), cfg.views_path,
              *[Path(target['source_spec']) for _, target in selections],
              *[Path(target['source_urdf']) for _, target in selections]]
    write_json(out / 'manifest.json', {'version': 1, 'policy': POLICY, 'pool_objects': len(accepted),
        'review_sample_objects': len(sampled), 'corruption_pool_counts': dict(Counter(r['kind'] for r in accepted)),
        'sample_seed': SAMPLE_SEED, 'complex_object_min_active_joints': 3,
        'search_exclusions': [{'kind': kind, 'asset_id': asset, 'reason': reason}
                              for (kind, asset), reason in sorted(SEARCH_EXCLUSIONS.items())],
        'motion_policy': 'full_native_legal_travel', 'human_review_complete': False,
        'training_eligible_cases': 0, 'inputs': {str(p.resolve()): sha256_path(p) for p in inputs},
        'files': {p.relative_to(out).as_posix(): sha256_path(p) for p in out.rglob('*')
                  if p.is_file() and p.name != 'manifest.json'}})
    return {'pool_objects': len(accepted), 'sample': sampled}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-pool', type=Path,
                        default=ROOT / 'output/hard-diverse-calibration-v1/target-pool.json')
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'output/hard-diverse-calibration-v1/review-bundle')
    args = parser.parse_args()
    print(json.dumps(build(args.target_pool, args.out), ensure_ascii=False, indent=2))
