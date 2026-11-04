"""Feedback-calibrated Medium selection overlay; never a human-review verdict."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from scripts.tier_construction.medium_strength_search import spec_geometry

POLICY = {
    'version': 'medium-selection-tightening-v1',
    'native_image_size': [256, 256],
    'revolute_axis_max_degrees': 20.,
    'prismatic_axis_max_degrees': 30.,
    'revolute_axis_silhouette_max_distance_px': 16.,
    'revolute_axis_mid_error_over_normal_motion_max': .30,
    'calibration': 'Finite human-feedback calibration; not independent validation.',
    'review_status': 'pending_human_review',
}


def _mask(png):
    pixels = np.asarray(Image.open(io.BytesIO(png)).convert('RGB'), dtype=np.int16)
    if pixels.shape != (256, 256, 3):
        raise ValueError('Selection overlay requires native 256px PNGs')
    return np.max(np.abs(pixels - pixels[0, 0]), axis=2) > 16


def silhouette_distance(reference_png, candidate_png):
    """Symmetric foreground Hausdorff distance in native pixels.

    Each foreground pixel measures distance to the nearest foreground pixel in
    the other mask. Internal texture differences do not affect this metric.
    """
    reference, candidate = _mask(reference_png), _mask(candidate_png)
    if not reference.any() or not candidate.any():
        raise ValueError('Cannot compare an empty foreground silhouette')
    return float(max(distance_transform_edt(~reference)[candidate].max(),
                     distance_transform_edt(~candidate)[reference].max()))


def evaluate(reference_spec, candidate_spec, gold, reference_pngs, candidate_pngs):
    """Return metrics/reasons for an immutable candidate, without regenerating it.

    Specs are emitted ABI dictionaries; PNG mappings are state -> bytes. Gold
    supplies kind, joint_id, parts and state_values. Geometry uses every vertex
    of the directly targeted part, following the original screening contract.
    The caller must still run the original truthfulness/checker gates.
    """
    metrics, reasons = {}, []
    if gold['kind'] != 'joint_axis':
        return {'metrics': metrics, 'reasons': reasons}
    joint = next(j for j in reference_spec['joints'] if j['id'] == gold['joint_id'])
    candidate_joint = next(j for j in candidate_spec['joints'] if j['id'] == gold['joint_id'])
    before, after = np.asarray(joint['axis'], dtype=float), np.asarray(candidate_joint['axis'], dtype=float)
    if (not np.isfinite(before).all() or not np.isfinite(after).all()
            or np.linalg.norm(before) <= 0 or np.linalg.norm(after) <= 0):
        raise ValueError('Invalid joint axes')
    angle = float(np.rad2deg(np.arccos(np.clip(
        before @ after / (np.linalg.norm(before) * np.linalg.norm(after)), -1., 1.))))
    metrics['axis_degrees'] = angle
    joint_type = joint['type']
    if joint_type not in ('revolute', 'prismatic'):
        raise ValueError('Unsupported joint type')
    if angle > POLICY[f'{joint_type}_axis_max_degrees'] + 1e-7:
        reasons.append(f'{joint_type}_axis_angle_above_selection_max')
    if joint_type == 'prismatic':
        return {'metrics': metrics, 'reasons': reasons}

    distances = {state: silhouette_distance(reference_pngs[state], candidate_pngs[state])
                 for state in ('mid', 'limit')}
    metrics['silhouette_symmetric_max_distance_px'] = distances
    for state, distance in distances.items():
        if distance > POLICY['revolute_axis_silhouette_max_distance_px']:
            reasons.append(f'{state}_revolute_axis_silhouette_above_selection_max')
    pid = gold['parts'][0]
    rest = spec_geometry(reference_spec)[pid].vertices
    values = {gold['joint_id']: gold['state_values']['mid']}
    reference_mid = spec_geometry(reference_spec, values)[pid].vertices
    candidate_mid = spec_geometry(candidate_spec, values)[pid].vertices
    normal_motion = float(np.sqrt(np.mean(np.sum((reference_mid - rest) ** 2, axis=1))))
    error = float(np.sqrt(np.mean(np.sum((candidate_mid - reference_mid) ** 2, axis=1))))
    metrics['mid_target_error_rms'] = error
    metrics['mid_reference_motion_rms'] = normal_motion
    if normal_motion <= 1e-12:
        metrics['mid_target_error_over_reference_motion'] = None
        reasons.append('mid_reference_target_motion_too_small_for_selection_ratio')
    else:
        ratio = error / normal_motion
        metrics['mid_target_error_over_reference_motion'] = ratio
        if ratio > POLICY['revolute_axis_mid_error_over_normal_motion_max']:
            reasons.append('mid_revolute_axis_relative_motion_error_above_selection_max')
    return {'metrics': metrics, 'reasons': reasons}
