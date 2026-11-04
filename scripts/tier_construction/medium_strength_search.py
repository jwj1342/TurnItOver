"""Conservative native-pixel Medium search; automatic pass remains provisional."""
from __future__ import annotations

import copy
import io
from itertools import product

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from scripts.tier_construction.level0_expansion_search import inside_camera, render_states
from scripts.tier_construction.prepare_level0_axis_review import border_subject_fraction, full_normal_framing, normal_view
from scripts.tier_construction.prepare_obvious_defect_review import STATES, checker_result, image_metrics
from scripts.tier_construction.validate_medium_pilot import validate_checker_target
from turnitover.checkers.base import PartMesh
from turnitover.core.sample import Framing
from turnitover.core.serde import to_dict
from turnitover.corruptions import get_corruption

CONFIGS = {
    'multi_joint_axis': {'defect': 'kinematics.joint_axis', 'joints': 'multi'},
    'multi_direction_flip': {'defect': 'kinematics.direction_flip', 'joints': 'multi'},
    'single_part_offset': {'defect': 'structure.part_offset', 'joints': 'single'},
}
POLICY = {
    'version': 'medium-strength-v3', 'native_image_size': [256, 256],
    'axis_degrees': [20., 30., 40.], 'axis_allowed_degrees': [15., 45.],
    'offset_diagonal_fractions': [.01, .02, .03],
    'flip_max_revolute_degrees': 20., 'flip_min_revolute_degrees': 8.,
    'flip_max_prismatic_diagonal_fraction': .06,
    'flip_min_prismatic_diagonal_fraction': .02,
    'core_fraction_bounds': [.0008, .065], 'changed_fraction_bounds': [.004, .18],
    'subject_changed_fraction_bounds': [.015, .55], 'minimum_core_region_px': 35,
    'target_displacement_diagonal_bounds': [.006, .22],
    'normal_motion_core_min': .0008, 'normal_motion_changed_max': .18,
    'review_status': 'pending_human_review',
    'limitations': 'Pixel and geometry screening are proxies, not a human difficulty judgment.',
    'camera_policy': 'Freeze from normal only; never retry camera after seeing corruption.',
}


def spec_geometry(spec, joint_values=None):
    """World vertices reconstructed from the emitted ABI (boxes use eight corners).

    Intended for bounds/displacement audit, not the sampled surface checker.
    """
    joint_values = joint_values or {}
    parts = {p['id']: p for p in spec['parts']}
    joints = {j['part']: j for j in spec['joints']}
    transforms = {}
    def transform(pid):
        if pid in transforms:
            return transforms[pid]
        p, local = parts[pid], np.eye(4)
        local[:3, :3] = Rotation.from_euler('XYZ', p['rotation']).as_matrix()
        local[:3, 3] = p['origin']
        if pid in joints:
            j, pivot = joints[pid], np.eye(4)
            pivot[:3, :3] = Rotation.from_euler('XYZ', j.get('frame_rotation', [0., 0., 0.])).as_matrix()
            pivot[:3, 3] = j['anchor']
            value = np.clip(joint_values.get(j['id'], 0.), *j['limits']) * j.get('sign', 1.)
            axis = np.asarray(j['axis']) / np.linalg.norm(j['axis'])
            if j['type'] == 'revolute':
                pivot[:3, :3] = pivot[:3, :3] @ Rotation.from_rotvec(axis * value).as_matrix()
            else:
                pivot[:3, 3] += pivot[:3, :3] @ (axis * value)
            local[:3, 3] -= j['anchor']
            local = pivot @ local
        transforms[pid] = (transform(p['parent']) @ local if p['parent'] else local)
        return transforms[pid]
    geometry = {}
    for pid, p in parts.items():
        if p.get('mesh') is not None:
            vertices = np.asarray(p['mesh']['vertices'], dtype=float)
            if not len(vertices):
                continue
        else:
            vertices = np.asarray(list(product((-.5, .5), repeat=3))) * p['size']
        matrix = transform(pid)
        geometry[pid] = PartMesh(vertices @ matrix[:3, :3].T + matrix[:3, 3], np.empty((0, 3), dtype=int))
    return geometry


def spec_diagonal(spec):
    """Rest world bounding-box diagonal, independent of supplied corruption labels."""
    points = np.concatenate([m.vertices for m in spec_geometry(spec).values()])
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError('Invalid reference bounding box')
    return diagonal


def validate_pair(reference_dict, candidate_dict, gold):
    """Require one truthful production corruption within the revised strength range."""
    kind, parameters = gold['kind'], gold.get('parameters', {})
    restored = copy.deepcopy(candidate_dict)
    if kind == 'clean':
        if restored != reference_dict or gold.get('defect_id') is not None:
            raise ValueError('Clean candidate differs from reference')
        return
    expected = {'joint_axis': 'kinematics.joint_axis', 'direction_flip': 'kinematics.direction_flip',
                'part_offset': 'structure.part_offset'}
    if kind not in expected or gold['defect_id'] != expected[kind]:
        raise ValueError('Inconsistent corruption kind')
    if kind != 'part_offset':
        source = next(j for j in reference_dict['joints'] if j['id'] == gold['joint_id'])
        actual = next(j for j in restored['joints'] if j['id'] == gold['joint_id'])
        if gold['parts'] != [source['part']] or parameters.get('joint') != source['id']:
            raise ValueError('Inconsistent joint target')
        before, after = np.asarray(source['axis']), np.asarray(actual['axis'])
        if not np.isfinite(after).all() or not np.isclose(np.linalg.norm(after), 1., atol=1e-6):
            raise ValueError('Invalid joint axis')
        if kind == 'direction_flip':
            if parameters.get('mode') != 'axis_negation' or not np.allclose(after, -before, atol=1e-12, rtol=0):
                raise ValueError('Flip must negate only the target axis')
        else:
            angle = float(np.rad2deg(np.arccos(np.clip(before @ after / np.linalg.norm(before), -1, 1))))
            if not 15. - 1e-8 <= angle <= 45. + 1e-8 or not np.isclose(angle, parameters['angle_deg'], atol=1e-6):
                raise ValueError('Axis deflection outside 15-45 degrees or mislabeled')
        actual['axis'] = source['axis']
    else:
        if len(gold['parts']) != 1:
            raise ValueError('Offset requires exactly one target')
        pid = gold['parts'][0]
        source = next(p for p in reference_dict['parts'] if p['id'] == pid)
        actual = next(p for p in restored['parts'] if p['id'] == pid)
        joint = next(j for j in reference_dict['joints'] if j['id'] == gold['joint_id'])
        delta = np.asarray(actual['origin']) - source['origin']
        diagonal, magnitude = spec_diagonal(reference_dict), float(np.linalg.norm(delta))
        fraction = magnitude / diagonal
        if (joint['part'] != pid or not np.isfinite(delta).all() or not .01 - 1e-7 <= fraction <= .03 + 1e-7
                or not np.allclose(delta, parameters['vector'], atol=1e-12, rtol=0)
                or not np.isclose(magnitude, parameters['magnitude_m'], atol=1e-12)
                or not np.isclose(diagonal, parameters['bbox_diagonal_m'], rtol=1e-5)
                or not np.isclose(fraction, parameters['offset_fraction'], atol=1e-7)):
            raise ValueError('Offset outside 1-3 percent of actual rest bbox or mislabeled')
        actual['origin'] = source['origin']
    if restored != reference_dict:
        raise ValueError('Changes beyond the declared target field')


def _mask(png):
    image = np.asarray(Image.open(io.BytesIO(png)).convert('RGB'), dtype=np.int16)
    if image.shape != (256, 256, 3):
        raise ValueError('Medium screening requires native 256px images')
    return np.max(np.abs(image - image[0, 0]), axis=2) > 16


def screen_pair(ref, rendered, view, check, label):
    """Cheap pixel/vertex screen; ``check=None`` deliberately omits exact checks."""
    metrics = {'reference_candidate': {}, 'reference_motion': {}, 'candidate_motion': {},
               'subject_normalized': {}, 'target_displacement_diagonal': {}}
    reasons, kind = [], label.defect_id.rsplit('.', 1)[1]
    required = STATES if kind == 'part_offset' else ('mid', 'limit')
    for state in STATES:
        a, b = ref['frames'][state][view], rendered['frames'][state][view]
        difference = image_metrics(b, a)
        subject = _mask(a) | _mask(b)
        normalized = difference['changed_fraction'] / max(float(subject.mean()), 1 / 65536)
        metrics['reference_candidate'][state] = difference
        metrics['subject_normalized'][state] = normalized
        if state in required:
            for key, bounds in [('core_changed_fraction', POLICY['core_fraction_bounds']),
                                ('changed_fraction', POLICY['changed_fraction_bounds'])]:
                if not bounds[0] <= difference[key] <= bounds[1]:
                    reasons.append(f'{state}_{key}_outside_medium_bounds')
            if not POLICY['subject_changed_fraction_bounds'][0] <= normalized <= POLICY['subject_changed_fraction_bounds'][1]:
                reasons.append(f'{state}_subject_difference_outside_medium_bounds')
            if difference['largest_core_region_px'] < POLICY['minimum_core_region_px']:
                reasons.append(f'{state}_core_region_too_small')
        elif difference['changed_fraction'] > .001:
            reasons.append('motion_defect_changes_rest')
        for role, png in [('reference', a), ('candidate', b)]:
            if border_subject_fraction(png) > .01:
                reasons.append(f'{role}_{state}_clipped')
        if state != 'rest':
            for role, data in [('reference', ref), ('candidate', rendered)]:
                motion = image_metrics(data['frames'][state][view], data['frames']['rest'][view])
                metrics[f'{role}_motion'][state] = motion
                if motion['core_changed_fraction'] < POLICY['normal_motion_core_min']:
                    reasons.append(f'{role}_{state}_motion_too_small')
                if kind == 'direction_flip' and motion['changed_fraction'] > POLICY['normal_motion_changed_max']:
                    reasons.append(f'{role}_{state}_flip_motion_exaggerated')
    if 'geometry' in ref and 'geometry' in rendered:
        all_points = np.concatenate([m.vertices for m in ref['geometry']['rest'].values()])
        diagonal = float(np.linalg.norm(np.ptp(all_points, axis=0)))
        for state in required:
            a, b = ref['geometry'][state][label.parts[0]].vertices, rendered['geometry'][state][label.parts[0]].vertices
            displacement = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))) / diagonal)
            metrics['target_displacement_diagonal'][state] = displacement
            if not POLICY['target_displacement_diagonal_bounds'][0] <= displacement <= POLICY['target_displacement_diagonal_bounds'][1]:
                reasons.append(f'{state}_target_displacement_outside_medium_bounds')
    if check is None:
        return metrics, reasons
    geometry, runtime = check['checks']
    failures = [i for i in geometry['invariants'] if i['passed'] is False]
    if check['code_correct'] or geometry['passed'] or not runtime['passed']:
        reasons.append('independent_checker_disagreement')
    for state in required:
        name = 'rest' if state == 'rest' else f'joint:target:{state}'
        if not any(label.parts[0] in i['parts'] and i['state'] == name for i in failures):
            reasons.append(f'{state}_checker_does_not_flag_target')
    return metrics, reasons


def _attempts(reference, joint, kind, diagonal, seed_base):
    strengths = POLICY['axis_degrees'] if kind == 'joint_axis' else POLICY['offset_diagonal_fractions'] if kind == 'part_offset' else [None]
    for index, strength in enumerate(strengths):
        kwargs = ({'min_deg': strength, 'max_deg': strength} if kind == 'joint_axis' else
                  {'min_m': diagonal * strength, 'max_m': diagonal * strength} if kind == 'part_offset' else {})
        corruption, found = get_corruption(CONFIGS_KIND[kind], **kwargs), 0
        for seed in range(seed_base + index * 10000, seed_base + index * 10000 + 8192):
            program, label = corruption.apply(reference, np.random.default_rng(seed))
            if (label.params.get('joint') != joint.id if kind != 'part_offset' else label.parts != (joint.part,)):
                continue
            if kind == 'part_offset':
                label.params.update(bbox_diagonal_m=diagonal, offset_fraction=strength)
            yield {'program': program, 'label': label, 'seed': seed}
            found += 1
            if found == (1 if kind == 'direction_flip' else 8):
                break


CONFIGS_KIND = {c['defect'].rsplit('.', 1)[1]: c['defect'] for c in CONFIGS.values()}


def _values(joint, kind, diagonal, preferred=None):
    endpoint = max(joint.limits, key=lambda x: (abs(x), x))
    if preferred:
        endpoint = float(preferred['limit'])
    if kind == 'direction_flip':
        cap = np.deg2rad(POLICY['flip_max_revolute_degrees']) if str(joint.type) == 'revolute' else diagonal * .06
        minimum = np.deg2rad(8.) if str(joint.type) == 'revolute' else diagonal * .02
        endpoint = np.sign(endpoint) * min(abs(endpoint), cap)
        if abs(endpoint) < minimum:
            return None
    if not joint.limits[0] <= endpoint <= joint.limits[1] or not joint.limits[0] <= 0 <= joint.limits[1]:
        return None
    return {'rest': 0., 'mid': endpoint / 2., 'limit': endpoint}


def find_pair(session, reference, views, configuration, seed_base, preferred=None):
    config, diagnostics = CONFIGS[configuration], []
    count, kind = len(reference.spec.joints), config['defect'].rsplit('.', 1)[1]
    if (count != 1 if config['joints'] == 'single' else count < 2):
        raise ValueError('Source joint complexity does not match configuration')
    diagonal, before = spec_diagonal(to_dict(reference.spec)), to_dict(reference.spec)
    preferred_gold = next((g for g in (preferred or {}).get('gold', []) if g['kind'] != 'clean'), {})
    preferred_joint = preferred_gold.get('joint_id')
    joints = sorted(reference.spec.joints, key=lambda j: j.id != preferred_joint)
    view_by_id = {str(v.id): v for v in views}
    for index, joint in enumerate(joints):
        camera = (preferred or {}).get('camera', {}) if joint.id == preferred_joint else {}
        values = _values(joint, kind, diagonal, camera.get('state_values'))
        record = {'joint_id': joint.id, 'state_values': values, 'attempts': [],
                  'selection_policy': 'first_normal_configuration_then_first_passing_corruption;fixed_camera'}
        diagnostics.append(record)
        if values is None:
            record['rejection_reasons'] = ['no_legal_modest_motion']
            continue
        view_ids = [camera['view']] if camera else list(view_by_id)
        framing = (Framing(center=tuple(camera['framing']['center']), radius=camera['framing']['radius'])
                   if camera else full_normal_framing(render_states(session, reference, None, joint.id, values, view_ids[:1])))
        ref = render_states(session, reference, framing, joint.id, values, view_ids)
        _, scores = normal_view(ref)
        eligible = [v for v in sorted(scores, key=lambda v: (-scores[v], v))
                    if scores[v] >= POLICY['normal_motion_core_min'] and inside_camera(ref, framing, view_by_id[v])[0]]
        record.update(normal_framing=to_dict(framing), normal_view_scores=scores)
        if not eligible:
            record['rejection_reasons'] = ['normal_motion_or_framing_failed']
            continue
        view = eligible[0]
        record['evaluated_views'] = [view]
        clean_checked = False
        for attempt in _attempts(reference, joint, kind, diagonal, seed_base + index * 100000):
            label = attempt['label']
            gold = {'kind': kind, 'defect_id': label.defect_id, 'joint_id': joint.id,
                    'parts': list(label.parts), 'parameters': label.params}
            validate_pair(before, to_dict(attempt['program'].spec), gold)
            rendered = render_states(session, attempt['program'], framing, joint.id, values, [view])
            metrics, reasons = screen_pair(ref, rendered, view, None, label)
            in_frame, extent = inside_camera(rendered, framing, view_by_id[view])
            if not in_frame:
                reasons.append('geometry_outside_frozen_camera')
            exact_checker_run, stage, check = False, 'visual', None
            if not reasons:
                if not clean_checked:
                    clean_check = checker_result(reference, ref, reference, ref)
                    record['clean_exact_checker_run'] = True
                    if not clean_check['code_correct']:
                        record['rejection_reasons'] = ['clean_checker_failed']
                        break
                    clean_checked = True
                check = checker_result(attempt['program'], rendered, reference, ref)
                exact_checker_run, stage = True, 'exact_checker'
                metrics, reasons = screen_pair(ref, rendered, view, check, label)
                try:
                    validate_checker_target(check, gold, before)
                except ValueError:
                    reasons.append('checker_target_subtree_gate_failed')
            record['attempts'].append({'seed': attempt['seed'], 'view': view, 'metrics': metrics,
                'parameters': label.params, 'automatic_pass': not reasons, 'rejection_reasons': reasons,
                'projected_extent_ndc': extent, 'exact_checker_run': exact_checker_run, 'stage': stage})
            if not reasons:
                record['selected'] = {'seed': attempt['seed'], 'view': view}
                return (ref, {**attempt, 'render': rendered, 'check': check, 'view': view,
                              'framing': framing, 'metrics': metrics}, joint.id, values), diagnostics
    return None, diagnostics
