"""Build five calibration examples: a 3 degree axis error over full native travel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.hard_construction.hard_artifacts import page
from scripts.hard_construction.hard_geometry import normal_framing, project_vertices
from scripts.hard_construction.hard_observability import target_readability
from scripts.tier_construction.level0_expansion_search import inside_camera, render_states
from scripts.tier_construction.prepare_level0_axis_review import border_subject_fraction
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
STATES = ('rest', 'mid', 'limit')
POLICY = 'hard-full-motion-axis-3deg-v1'
TARGETS = (
    ('PM_45661', 'joint_0', 'cabinet door'),
    ('PM_101971', 'joint_0', 'oven door'),
    ('PM_7332', 'joint_3', 'dishwasher door'),
    ('PM_101668', 'joint_2', 'suitcase lid'),
    ('PM_22301', 'joint_1', 'table board'),
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def rotated_axes(axis):
    """Eight deterministic points on the exact 3-degree cone around an axis."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
    perpendicular = basis - axis * (basis @ axis)
    perpendicular /= np.linalg.norm(perpendicular)
    for angle in np.arange(0, 360, 45):
        direction = Rotation.from_rotvec(axis * np.deg2rad(angle)).apply(perpendicular)
        changed = axis * np.cos(np.deg2rad(3)) + direction * np.sin(np.deg2rad(3))
        yield tuple(float(v) for v in changed / np.linalg.norm(changed))


def view_pair(normal, framing, views, part):
    preference = ('front', 'oblique_fr', 'right', 'left', 'oblique_fl')
    available = []
    for view in preference:
        if view not in views or not inside_camera(normal, framing, views[view])[0]:
            continue
        if any(border_subject_fraction(normal['frames'][state][view]) > .01 for state in STATES):
            continue
        motion = max(pixel_fraction(normal['frames']['rest'][view], normal['frames'][state][view])
                     for state in ('mid', 'limit'))
        readable = target_readability(normal, part, framing, views[view], 'limit', {
            'normal_target_min_long_edge_px': 24, 'normal_target_min_short_edge_px': 6,
            'normal_target_min_solid_long_edge_px': 18, 'normal_target_min_solid_short_edge_px': 6,
            'normal_target_min_solid_area_px': 80})['readable']
        if readable and motion >= .005:
            available.append((view, motion))
    if len(available) < 2:
        raise ValueError('Normal full motion lacks two readable views')
    initial = available[0][0]
    initial_def = views[initial]
    separated = [(v, score) for v, score in available[1:] if
        abs((views[v].azimuth_deg - initial_def.azimuth_deg + 180) % 360 - 180) >= 30
        or abs(views[v].elevation_deg - initial_def.elevation_deg) >= 30]
    if not separated:
        raise ValueError('Normal full motion lacks a separated supplementary view')
    return initial, max(separated, key=lambda row: row[1])[0]


def pixel_fraction(a, b):
    from io import BytesIO
    from PIL import Image
    x, y = [np.asarray(Image.open(BytesIO(v)).convert('RGB'), dtype=np.int16) for v in (a, b)]
    return float((np.max(np.abs(x - y), axis=2) > 16).mean())


def build(source, out):
    source, out = source.resolve(), out.resolve()
    cfg = load_generate_config(DEFAULT_CONFIG)
    views_tuple = load_views(cfg.views_path)
    views = {str(v.id): v for v in views_tuple}
    records = {json.loads(p.read_text())['asset_id']: json.loads(p.read_text())
               for p in (source / 'qualified').glob('*.json')}
    public, gold, checks, reviews, examples = [], [], [], [], []
    with ObservationSession(cfg.render, views_tuple) as session:
        for asset_id, joint_id, role in TARGETS:
            record = records[asset_id]
            spec = decode_spec(json.loads(Path(record['metadata']['source_spec']).read_text()))
            reference = ObjectProgram.from_spec(spec)
            joint = spec.joint(joint_id)
            endpoint = max((v for v in joint.limits if abs(v) > 1e-8), key=abs)
            if str(joint.type) != 'revolute' or abs(endpoint) < np.deg2rad(60):
                raise ValueError(f'{asset_id} target lacks large native revolute travel')
            values = {'rest': 0., 'mid': float(endpoint / 2), 'limit': float(endpoint)}
            seed_render = render_states(session, reference, None, joint_id, values, ['front'])
            framing = normal_framing(seed_render, 1.15)
            normal = render_states(session, reference, framing, joint_id, values, list(views))
            initial, supplementary = view_pair(normal, framing, views, joint.part)
            candidates = []
            for index, axis in enumerate(rotated_axes(joint.axis)):
                candidate = ObjectProgram.from_spec(spec.replace_joint(joint_id, axis=axis))
                rendered = render_states(session, candidate, framing, joint_id, values,
                                         [initial, supplementary])
                if any(not inside_camera(rendered, framing, views[v])[0] for v in (initial, supplementary)):
                    continue
                check = checker_result(candidate, rendered, reference, normal)
                if check['code_correct'] or not check['checks'][1]['passed']:
                    continue
                errors = {}
                for view in (initial, supplementary):
                    a = normal['geometry']['limit'][joint.part].vertices
                    b = rendered['geometry']['limit'][joint.part].vertices
                    delta = project_vertices(a, framing, views[view]) - project_vertices(b, framing, views[view])
                    errors[view] = float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1))))
                if min(errors.values()) < .75 or max(errors.values()) > 12:
                    continue
                candidates.append((abs(max(errors.values()) - 4), index, candidate, rendered, check, errors))
            if not candidates:
                raise ValueError(f'{asset_id} has no truthful visible 3-degree candidate')
            _, axis_variant, candidate, rendered, check, errors = min(candidates)
            clean_check = checker_result(reference, normal, reference, normal)
            authenticity = audit_model(session, UrdfModel(Path(record['source_urdf'])), spec)
            if not clean_check['code_correct'] or authenticity['passed'] is not True:
                raise ValueError(f'{asset_id} clean or source authenticity failed')
            cid = 'hp-' + hashlib.sha256(f'{POLICY}:{asset_id}:{joint_id}:{axis_variant}'.encode()).hexdigest()[:14]
            for view in (initial, supplementary):
                for name, frames in (('reference', normal), ('candidate', rendered)):
                    for state in STATES:
                        path = out / f'native/{cid}/{view}/{name}_{state}.png'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(frames['frames'][state][view])
            program = f'private/programs/{cid}.json'
            write_json(out / program, to_dict(candidate.spec))
            public.append({'case_id': cid, 'initial_view': initial, 'supplementary_view': supplementary,
                'states': list(STATES), 'native_root': f'native/{cid}', 'image_size': [256, 256],
                'status': 'pending_human_review'})
            gold.append({'case_id': cid, 'asset_id': asset_id, 'kind': 'joint_axis', 'joint_id': joint_id,
                'part_id': joint.part, 'semantic_role': role, 'axis_angle_deg': 3.,
                'original_axis': list(joint.axis), 'candidate_axis': list(candidate.spec.joint(joint_id).axis),
                'original_limits': list(joint.limits), 'state_values': values,
                'motion_policy': 'rest_to_half_then_full_native_endpoint', 'program': program,
                'reference_sha': reference.sha, 'candidate_sha': candidate.sha,
                'initial_view': initial, 'supplementary_view': supplementary,
                'projected_limit_error_px': errors, 'source_record': record['metadata'],
                'training_eligibility': 'calibration_only_pending_human_review'})
            checks.append({'case_id': cid, **check, 'source_authenticity': authenticity})
            reviews.append({'case_id': cid, 'status': 'pending_human_review', 'reviewer': None,
                'reviewed_at': None, 'human_judgment': {'initial_verdict': None, 'initial_notes': None,
                'final_decision': None, 'final_notes': None, 'supplementary_decisive': None,
                'wrong_part_description': None, 'observed_views': [], 'view_switches': []}})
            examples.append({'case_id': cid, 'asset_id': asset_id, 'category': spec.category,
                'semantic_role': role, 'joint_id': joint_id, 'native_travel_deg': abs(np.rad2deg(endpoint)),
                'axis_error_deg': 3., 'axis_variant': axis_variant, 'views': [initial, supplementary],
                'projected_limit_error_px': errors})
    page(out, public)
    for name, rows in (('review_manifest.jsonl', public), ('private/gold.jsonl', gold),
                       ('private/checks.jsonl', checks), ('human_review.jsonl', reviews)):
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    write_json(out / 'examples.json', examples)
    inputs = [Path(__file__), Path(DEFAULT_CONFIG), cfg.views_path,
              *[Path(records[a]['metadata']['source_spec']) for a, _, _ in TARGETS],
              *[Path(records[a]['source_urdf']) for a, _, _ in TARGETS]]
    write_json(out / 'manifest.json', {'version': 1, 'policy': POLICY, 'objects': 5, 'cases': 5,
        'axis_error_deg': 3., 'motion_policy': 'full_native_legal_travel',
        'human_review_complete': False, 'training_eligible_cases': 0,
        'inputs': {str(p.resolve()): sha256_path(p) for p in inputs},
        'files': {p.relative_to(out).as_posix(): sha256_path(p) for p in out.rglob('*')
                  if p.is_file() and p.name != 'manifest.json'}})
    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'output/hard-new-download-v2')
    parser.add_argument('--out', type=Path, default=ROOT / 'output/hard-full-motion-calibration-v1')
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.out), ensure_ascii=False, indent=2))
