"""Truthful Hard mutations and provisional two-view evidence admission."""
from __future__ import annotations

import copy
import io
import numpy as np
from PIL import Image
from scipy import ndimage

from scripts.hard_construction.hard_geometry import project_vertices, spec_diagonal
from scripts.hard_construction.hard_observability import target_readability
from scripts.tier_construction.prepare_obvious_defect_review import STATES

CONFIGS = {'multi_joint_axis': 'kinematics.joint_axis',
           'multi_direction_flip': 'kinematics.direction_flip',
           'multi_part_offset': 'structure.part_offset'}
POLICY = {
    'version': 'hard-pilot-v2', 'native_image_size': [256, 256],
    'axis_revolute_degrees': [1., 3., 5., 8., 10.],
    'axis_prismatic_degrees': [1., 5., 10., 20., 30.],
    'offset_diagonal_fractions': [.0025, .005, .0075, .01],
    'flip_revolute_degrees': [3., 8.], 'flip_prismatic_diagonal_fractions': [.01, .03],
    'flip_nominal_revolute_degrees': 5., 'flip_nominal_prismatic_diagonal_fraction': .02,
    'normal_revolute_max_degrees': 90., 'normal_prismatic_max_diagonal_fraction': .20,
    'max_target_joints': 3, 'attempts_per_strength': 3,
    'supplementary_changed_fraction_bounds': [.0003, .08],
    'supplementary_changed_region_min_px': 16,
    'target_projected_error_px_bounds': [1.25, 14.],
    'motion_target_min_projected_error_px': 2.,
    'normal_framing_padding': 1.1,
    'normal_target_min_long_edge_px': 24., 'normal_target_min_short_edge_px': 6.,
    'normal_target_min_solid_long_edge_px': 18., 'normal_target_min_solid_short_edge_px': 6.,
    'normal_target_min_solid_area_px': 80,
    'subject_difference_max': .45, 'world_target_error_diagonal_max': .06,
    'camera_policy': 'Two views and framing frozen from normal only, before corruption',
    'difficulty_status': 'pending_human_review',
    'qualification': 'Automatic evidence proxies do not establish Hard difficulty or human approval',
}


def validate_pair(reference, candidate, gold):
    restored = copy.deepcopy(candidate)
    kind, params = gold['kind'], gold.get('parameters', {})
    if kind == 'clean':
        if candidate != reference or gold.get('defect_id') is not None:
            raise ValueError('Clean candidate differs from reference')
        return
    expected = {'joint_axis': CONFIGS['multi_joint_axis'], 'direction_flip': CONFIGS['multi_direction_flip'],
                'part_offset': CONFIGS['multi_part_offset']}
    if kind not in expected or gold['defect_id'] != expected[kind]:
        raise ValueError('Inconsistent single corruption kind')
    joint = next(j for j in reference['joints'] if j['id'] == gold['joint_id'])
    if gold['parts'] != [joint['part']]:
        raise ValueError('Declared target disagrees with the selected joint')
    if kind in ('joint_axis', 'direction_flip'):
        actual = next(j for j in restored['joints'] if j['id'] == joint['id'])
        a, b = np.asarray(joint['axis']), np.asarray(actual['axis'])
        if (params.get('joint') != joint['id'] or not np.isfinite(b).all()
                or not np.isclose(np.linalg.norm(b), 1., atol=1e-6)
                or not np.allclose(params.get('original_axis'), a, atol=1e-12, rtol=0)):
            raise ValueError('Invalid target or candidate axis')
        if kind == 'direction_flip':
            if params.get('mode') != 'axis_negation' or not np.allclose(b, -a, atol=1e-12, rtol=0):
                raise ValueError('Direction flip must truly negate one axis')
        else:
            angle = float(np.rad2deg(np.arccos(np.clip(a @ b / np.linalg.norm(a), -1., 1.))))
            maximum = 10. if joint['type'] == 'revolute' else 30.
            if (params.get('mode') != 'axis_random_angle' or not 1. - 1e-7 <= angle <= maximum + 1e-7
                    or not np.isclose(angle, params['angle_deg'], atol=1e-6)):
                raise ValueError('Actual axis deflection is outside Hard bounds or mislabeled')
        actual['axis'] = joint['axis']
    else:
        actual = next(p for p in restored['parts'] if p['id'] == joint['part'])
        before = next(p for p in reference['parts'] if p['id'] == joint['part'])
        delta = np.asarray(actual['origin']) - before['origin']
        diagonal, magnitude = spec_diagonal(reference), float(np.linalg.norm(delta))
        if (not np.isfinite(delta).all() or not 0 < magnitude / diagonal <= .01 + 1e-8
                or not np.allclose(delta, params['vector'], atol=1e-12, rtol=0)
                or not np.isclose(magnitude, params['magnitude_m'], atol=1e-12)
                or not np.isclose(diagonal, params['bbox_diagonal_m'], rtol=1e-6)
                or not np.isclose(magnitude / diagonal, params['offset_fraction'], atol=1e-8)):
            raise ValueError('Offset must be positive and no more than one percent of the actual bbox')
        actual['origin'] = before['origin']
    if restored != reference:
        raise ValueError('Changes beyond the single declared target field')


def state_values(joint, kind, diagonal):
    if not joint.limits[0] <= 0 <= joint.limits[1]:
        return None
    endpoint = max(joint.limits, key=lambda value: (abs(value), value))
    revolute = str(joint.type) == 'revolute'
    if kind == 'direction_flip':
        minimum, maximum = (np.deg2rad([3., 8.]) if revolute else np.array([.01, .03]) * diagonal)
        nominal = (np.deg2rad(POLICY['flip_nominal_revolute_degrees']) if revolute else
                   diagonal * POLICY['flip_nominal_prismatic_diagonal_fraction'])
        endpoint = np.sign(endpoint) * min(abs(endpoint), nominal, maximum)
        if abs(endpoint) < minimum:
            return None
    else:
        cap = np.deg2rad(POLICY['normal_revolute_max_degrees']) if revolute else diagonal * .20
        endpoint = np.sign(endpoint) * min(abs(endpoint), cap)
    if endpoint == 0:
        return None
    return {'rest': 0., 'mid': float(endpoint / 2), 'limit': float(endpoint)}


def validate_states(reference, gold):
    values = gold['state_values']
    joint = next(j for j in reference['joints'] if j['id'] == gold['joint_id'])
    if (set(values) != set(STATES) or not all(np.isfinite(v) for v in values.values())
            or values['rest'] != 0 or values['limit'] == 0 or not np.isclose(values['mid'], values['limit'] / 2)
            or any(not joint['limits'][0] <= v <= joint['limits'][1] for v in values.values())):
        raise ValueError('Observation states are not finite, legal rest/half/full source states')
    if gold['kind'] == 'direction_flip' or gold.get('config', gold.get('configuration')) == 'multi_direction_flip':
        bounds = np.deg2rad([3., 8.]) if joint['type'] == 'revolute' else spec_diagonal(reference) * np.array([.01, .03])
        if not bounds[0] - 1e-10 <= abs(values['limit']) <= bounds[1] + 1e-10:
            raise ValueError('Flip observation excursion outside confirmed Hard limits')


def _pixel_metrics(a, b):
    a, b = [np.asarray(Image.open(io.BytesIO(data)).convert('RGB'), dtype=np.int16) for data in (a, b)]
    if a.shape != (256, 256, 3) or b.shape != a.shape:
        raise ValueError('Hard evidence must use native 256 images')
    changed = np.max(np.abs(a - b), axis=2) > 16
    subject = (np.max(np.abs(a - a[0, 0]), axis=2) > 16) | (np.max(np.abs(b - b[0, 0]), axis=2) > 16)
    labels, _ = ndimage.label(changed)
    sizes = np.bincount(labels.ravel())[1:]
    return {'changed_fraction': float(changed.mean()), 'largest_region_px': int(sizes.max()) if sizes.size else 0,
            'subject_changed_fraction': float(changed.sum() / max(subject.sum(), 1))}


def screen_pair(reference, candidate, initial, supplementary, framing, views, part, kind, complexity):
    metrics = {'per_view': {}, 'target_world_error_diagonal': {}, 'route': None}
    reasons = []
    required = STATES if kind == 'part_offset' else ('mid', 'limit')
    points = np.concatenate([m.vertices for m in reference['geometry']['rest'].values()])
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    for state in required:
        a, b = [v['geometry'][state][part].vertices for v in (reference, candidate)]
        displacement = float(np.sqrt(np.mean(np.sum((a - b)**2, axis=1))) / diagonal)
        metrics['target_world_error_diagonal'][state] = displacement
        if displacement <= 0 or displacement > POLICY['world_target_error_diagonal_max']:
            reasons.append(f'{state}_target_geometry_too_small_or_large')
    for view in (initial, supplementary):
        states = {}
        for state in STATES:
            row = _pixel_metrics(reference['frames'][state][view], candidate['frames'][state][view])
            a, b = [v['geometry'][state][part].vertices for v in (reference, candidate)]
            delta = project_vertices(a, framing, views[view]) - project_vertices(b, framing, views[view])
            row['target_projected_rms_px'] = float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))
            row['normal_target_readability'] = target_readability(reference, part, framing, views[view], state, POLICY)
            states[state] = row
            if state in required and (row['subject_changed_fraction'] > POLICY['subject_difference_max']
                    or row['changed_fraction'] > POLICY['supplementary_changed_fraction_bounds'][1]
                    or row['target_projected_rms_px'] > POLICY['target_projected_error_px_bounds'][1]):
                reasons.append(f'{view}_{state}_difference_too_obvious')
            if state == 'rest' and kind != 'part_offset' and row['changed_fraction'] > .0001:
                reasons.append('motion_corruption_changed_rest')
        metrics['per_view'][view] = states
    supported = [s for s in required if
        metrics['per_view'][supplementary][s]['changed_fraction'] >= POLICY['supplementary_changed_fraction_bounds'][0]
        and metrics['per_view'][supplementary][s]['largest_region_px'] >= POLICY['supplementary_changed_region_min_px']
        and metrics['per_view'][supplementary][s]['normal_target_readability']['readable']
        and metrics['per_view'][supplementary][s]['target_projected_rms_px'] >= (
            POLICY['target_projected_error_px_bounds'][0] if kind == 'part_offset'
            else POLICY['motion_target_min_projected_error_px'])]
    metrics['supported_states'] = supported
    if not supported:
        reasons.append('supplementary_visual_evidence_insufficient')
    else:
        ambiguity = any(metrics['per_view'][initial][s]['changed_fraction'] <= .6 * metrics['per_view'][supplementary][s]['changed_fraction']
                        and (metrics['per_view'][initial][s]['target_projected_rms_px'] <= .7 * metrics['per_view'][supplementary][s]['target_projected_rms_px']
                             or metrics['per_view'][initial][s]['largest_region_px'] <= .35 * metrics['per_view'][supplementary][s]['largest_region_px']) for s in supported)
        if ambiguity:
            metrics['route'] = 'one_view_switch_candidate'
        elif complexity >= 3:
            metrics['route'] = 'structural_localization_candidate'
        else:
            reasons.append('no_hard_route_evidence')
    return metrics, sorted(set(reasons))


def validate_exact_check(check, part, supported_states):
    geometry, runtime = check['checks']
    if check['code_correct'] or geometry['passed'] or not runtime['passed']:
        return ['exact_checker_disagreement']
    flags = [i for i in geometry['invariants'] if i['passed'] is False and part in i['parts']]
    states = {'rest' if s == 'rest' else f'joint:target:{s}' for s in supported_states}
    return [] if any(i['state'] in states for i in flags) else ['exact_checker_does_not_flag_visible_target']
