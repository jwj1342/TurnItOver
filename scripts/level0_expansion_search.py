"""Normal-only state/camera search with the unchanged Level 0 pixel gates."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from scripts.prepare_level0_axis_review import full_normal_framing, normal_view, screen, score
from scripts.prepare_obvious_defect_review import STATES, checker_result
from turnitover.core.actions import RuntimeProperty
from turnitover.corruptions import get_corruption


def render_states(session, program, framing, joint_id, values, views):
    info = session.load(program, framing=framing)
    frames, geometry = {}, {}
    try:
        for state in STATES:
            session.reset_joints()
            session.actuate_joint(joint_id, values[state])
            geometry[state] = session.export_geometry()
            frames[state] = {v: session.request_view(v).png for v in views}
        return {'frames': frames, 'geometry': geometry, 'framing': info.framing,
                'joint': info.joints[joint_id], 'stats': session.query_runtime(RuntimeProperty.STATS),
                'hierarchy': session.query_runtime(RuntimeProperty.HIERARCHY), 'parts': info.parts}
    finally:
        session.dispose()


def inside_camera(rendered, framing, view):
    """Project all exported vertices with the actual fixed harness camera formula."""
    az, el = np.deg2rad([view.azimuth_deg, view.elevation_deg])
    direction = np.array([np.sin(az)*np.cos(el), np.sin(el), np.cos(az)*np.cos(el)])
    up = np.array([0., 0., -1.]) if abs(view.elevation_deg) > 89 else np.array([0., 1., 0.])
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    maximum = 0.
    for geometry in rendered['geometry'].values():
        points = np.concatenate([m.vertices for m in geometry.values()]) - framing.center
        if view.projection == 'orthographic':
            depth = 3 * framing.radius - points @ direction
            xy = np.column_stack((points @ right, points @ up)) / (1.1 * framing.radius)
            far = 10 * framing.radius
        else:
            distance = framing.radius * (view.distance_factor or 1.) / np.sin(np.deg2rad(20)) * 1.05
            depth = distance - points @ direction
            xy = np.column_stack((points @ right, points @ up)) / (depth[:, None] * np.tan(np.deg2rad(20)))
            far = 10 * distance
        maximum = max(maximum, float(np.max(np.abs(xy))))
        if np.any(depth <= .01) or np.any(depth >= far):
            return False, maximum
    return maximum <= .98, maximum


def attempts(reference, joint_id, seed_base):
    corruption = get_corruption('kinematics.joint_axis', min_deg=90., max_deg=90.)
    found = []
    for seed in range(seed_base, seed_base + 4096):
        program, label = corruption.apply(reference, np.random.default_rng(seed))
        if label.params['joint'] == joint_id:
            found.append({'program': program, 'label': label, 'seed': seed})
        if len(found) == 24:
            return found
    raise ValueError('No 24 target-joint attempts within deterministic seed bound')


def find_pair(session, reference, views, seed_base):
    view_ids = [str(v.id) for v in views]
    view_by_id = {str(v.id): v for v in views}
    diagnostics = []
    for index, joint in enumerate(reference.spec.joints):
        # Values are legal original-source joint values, chosen before corruption.
        endpoints = sorted(set(joint.limits) - {0.}, key=lambda x: (-abs(x), -x))
        candidates = None
        for endpoint in endpoints:
            values = {'rest': 0., 'mid': endpoint / 2., 'limit': endpoint}
            initial = render_states(session, reference, None, joint.id, values, [view_ids[0]])
            wide = full_normal_framing(initial)
            for padding in (1.25, 1.0, 1.5):
                framing = replace(wide, radius=wide.radius * padding / 1.5)
                ref = render_states(session, reference, framing, joint.id, values, view_ids)
                _, scores = normal_view(ref)
                eligible = [v for v in sorted(scores, key=lambda v: (-scores[v], v))
                            if scores[v] >= .020 and inside_camera(ref, framing, view_by_id[v])[0]][:4]
                record = {'joint_id': joint.id, 'state_values': values,
                    'state_policy': 'rest_to_native_endpoint_half_then_endpoint',
                    'normal_padding': padding, 'normal_framing': {'center': list(framing.center), 'radius': framing.radius},
                    'normal_view_scores': scores, 'evaluated_views': eligible, 'attempts': []}
                diagnostics.append(record)
                if not eligible:
                    record['rejection_reasons'] = ['normal_motion_or_framing_gate_failed']
                    continue
                clean_check = checker_result(reference, ref, reference, ref)
                if not clean_check['code_correct']:
                    record['rejection_reasons'] = ['clean_checker_failed']
                    continue
                if candidates is None:
                    candidates = attempts(reference, joint.id, seed_base + index * 5000)
                best = None
                for attempt in candidates:
                    rendered = render_states(session, attempt['program'], framing, joint.id, values, eligible)
                    check = checker_result(attempt['program'], rendered, reference, ref)
                    for view in eligible:
                        metrics, reasons = screen(ref, rendered, view, check)
                        in_frame, extent = inside_camera(rendered, framing, view_by_id[view])
                        if not in_frame:
                            reasons.append('geometry_outside_frozen_camera')
                        record['attempts'].append({'seed': attempt['seed'], 'view': view,
                            'automatic_pass': not reasons, 'rejection_reasons': reasons,
                            'projected_extent_ndc': extent, 'metrics': metrics})
                        if not reasons and (best is None or score(metrics) > score(best['metrics'])):
                            best = {**attempt, 'render': rendered, 'check': check, 'view': view, 'metrics': metrics}
                    if best:
                        record['selection_policy'] = 'first_passing_seed; best_eligible_view_within_seed'
                        record['selected'] = {'seed': best['seed'], 'view': best['view']}
                        return (ref, best, joint.id, values), diagnostics
                if best:
                    record['selected'] = {'seed': best['seed'], 'view': best['view']}
                    return (ref, best, joint.id, values), diagnostics
    return None, diagnostics
