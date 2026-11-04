"""Reuse the existing frozen-normal-camera search and audit one defect only."""
from __future__ import annotations

import numpy as np

from scripts.tier_construction.construction_pilot import find_case
from scripts.tier_construction.prepare_level0_axis_review import border_subject_fraction
from scripts.tier_construction.prepare_obvious_defect_review import STATES, checker_result
from scripts.tier_construction.validate_medium_pilot import validate_checker_target, validate_program_pair
from turnitover.core.serde import to_dict

CONFIGS = {
    'multi_joint_axis': {'defect': 'kinematics.joint_axis', 'joints': 'multi'},
    'multi_direction_flip': {'defect': 'kinematics.direction_flip', 'joints': 'multi'},
    'single_part_offset': {'defect': 'structure.part_offset', 'joints': 'single'},
}


def validate_change(reference, candidate, label, joint_id):
    """Require the existing corruption's exact single-field structural edit."""
    before, after = to_dict(reference.spec), to_dict(candidate.spec)
    if label.defect_id.startswith('kinematics.'):
        old = next(j for j in before['joints'] if j['id'] == joint_id)
        new = next(j for j in after['joints'] if j['id'] == joint_id)
        if label.params['joint'] != joint_id or list(label.parts) != [old['part']]:
            raise ValueError('Target joint/part label disagrees with program')
        a, b = np.asarray(old['axis']), np.asarray(new['axis'])
        if label.defect_id == 'kinematics.direction_flip':
            if label.params.get('mode') != 'axis_negation' or not np.allclose(b, -a, atol=1e-12, rtol=0):
                raise ValueError('Direction flip must negate only the target axis')
        elif label.defect_id == 'kinematics.joint_axis':
            if abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))) > 1e-6:
                raise ValueError('Joint axis must change by exactly 90 degrees')
        else:
            raise ValueError('Unexpected motion defect')
        new['axis'] = old['axis']
    elif label.defect_id == 'structure.part_offset':
        if len(label.parts) != 1:
            raise ValueError('Offset must affect exactly one directly modified part')
        part_id = label.parts[0]
        joint = next(j for j in before['joints'] if j['id'] == joint_id)
        if joint['part'] != part_id:
            raise ValueError('Offset must target the visible moving joint part')
        old = next(p for p in before['parts'] if p['id'] == part_id)
        new = next(p for p in after['parts'] if p['id'] == part_id)
        delta = np.asarray(new['origin']) - old['origin']
        if not np.allclose(delta, label.params['vector'], atol=1e-12, rtol=0) or np.linalg.norm(delta) <= 0:
            raise ValueError('Offset vector disagrees with the actual changed origin')
        new['origin'] = old['origin']
    else:
        raise ValueError('Unexpected pilot defect')
    if before != after:
        raise ValueError('Changes beyond the permitted single target field')


def find_pair(session, reference, views, configuration, seed_base):
    config = CONFIGS[configuration]
    count = len(reference.spec.joints)
    if (config['joints'] == 'single' and count != 1) or (config['joints'] == 'multi' and count < 2):
        raise ValueError('Source joint complexity does not match the configuration')
    pair, diagnostics = find_case(session, reference, views, config, seed_base)
    if pair is not None:
        ref, chosen, joint_id, _ = pair
        label = chosen['label']
        gold = {'kind': config['defect'].split('.')[1], 'defect_id': label.defect_id,
                'joint_id': joint_id, 'parts': list(label.parts), 'parameters': label.params}
        try:
            validate_change(reference, chosen['program'], label, joint_id)
            validate_program_pair(to_dict(reference.spec), to_dict(chosen['program'].spec), gold)
            validate_checker_target(checker_result(chosen['program'], chosen['render'], reference, ref),
                                    gold, to_dict(reference.spec))
        except ValueError as error:
            diagnostics[-1].setdefault('rejection_reasons', []).append('strict_target_or_structure_gate_failed')
            diagnostics[-1]['strict_gate_error'] = str(error)
            return None, diagnostics
        # The reused offset gate checks the defective image; also require the
        # frozen normal trajectory to remain clear of all four image borders.
        if any(border_subject_fraction(ref['frames'][state][chosen['view']]) > .01 for state in STATES):
            diagnostics[-1].setdefault('rejection_reasons', []).append('normal_reference_clipped')
            return None, diagnostics
    return pair, diagnostics
