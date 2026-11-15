"""Independent native replay scoring using the repository's existing checkers.

This is evaluator glue, not a coding-agent or browser harness. The development
contract deliberately lists unsupported semantic domains and never turns their
missing predicates into passing checks. No validation-fitted thresholds.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from turnitover.checkers.base import Evidence, PartMesh
from turnitover.checkers.geometry import GeometryAlignmentChecker
from turnitover.checkers.native_mesh import topology, self_crossings
from turnitover.checkers.mesh_heuristics import HEURISTICS, component_count, surface_match
from turnitover.repair.native_properties import (
    ATTRIBUTE_PROFILE_TOLERANCE, MATERIAL_FIELDS, hierarchy, lifecycle, materials, snapshots,
)
from turnitover.repair.native_geometry import (
    FEATURE_DEFECTS, CONTACT_DEFECTS, contact_regressions, feature_regressions, measure_geometry,
)
from turnitover.core.serde import to_dict
from turnitover.render.protocol import decode_indices, decode_positions
from turnitover.repair.table1_scoring import (
    classify_exposure, make_repair_audit, runtime_budget_predicate,
)
from turnitover.verifier.gt_validator import _check_contract

STATES = ('zero', 'mid', 'limit')
ALIGNMENT_DEFECTS = (
    'geometry.global_scale', 'geometry.anisotropic_scale',
    'structure.part_offset', 'structure.part_rotation', 'structure.symmetry_break',
    'structure.pivot_offset', 'kinematics.joint_axis',
    'kinematics.joint_anchor', 'kinematics.direction_flip',
)
CONTRACT_DEFECTS = ('kinematics.joint_type', 'kinematics.joint_range')
BUDGET_DEFECTS = ('runtime.triangle_budget', 'runtime.draw_calls')
TOPOLOGY_DEFECTS = ('geometry.open_volume', 'geometry.self_intersection', 'geometry.duplicate_part')
SUPPORTED = (*ALIGNMENT_DEFECTS, *CONTRACT_DEFECTS, *BUDGET_DEFECTS,
             'geometry.missing_part', *TOPOLOGY_DEFECTS, *MATERIAL_FIELDS,
             'structure.wrong_parent', 'runtime.missing_dispose', 'runtime.tick_missing_or_phase',
             *FEATURE_DEFECTS, *CONTACT_DEFECTS)
SCORING_CONTRACT = {
    'status': 'development', 'version': 4, 'exposure_status': 'unreviewed',
    'heuristics': HEURISTICS,
    'material_attribute_profiles': ATTRIBUTE_PROFILE_TOLERANCE,
    'supported_defects': list(SUPPORTED),
    'required_checks': ['compile_render', 'program.contract', 'geometry.alignment',
                        'runtime.budget', 'geometry.topology', 'structure.hierarchy',
                        'material.properties', 'runtime.lifecycle', 'runtime.temporal',
                        'geometry.feature_shapes', 'structure.contacts'],
    'geometry': {'tol_chamfer_m': .005, 'tol_iou': .9,
                 'samples_per_part': 2000, 'seed': 0},
    'threshold_source': 'Existing GeometryAlignmentChecker defaults; public task runtime caps',
    'limitations': [
        'Candidate reference-alignment predicates, not validated full29 construction predicates.',
        'Same-part multiple defects can couple reference-alignment resolved outcomes.',
        'Native topology scope is proper nonadjacent noncoplanar crossings, not general solid certification.',
        'Native material slot and temporal vertex correspondence must remain identifiable.',
        'Thin-sheet, bevel and contact predicates require hash-bound construction selectors.',
        'Reference-relative thickness/surface/normal and nonconvex occupancy checks are heuristic.',
        'Defaults are versioned engineering tolerances, not Validation-fitted or Train-calibrated.',
    ],
}


def _join(values):
    values = list(values)
    return False if False in values else True if values and all(v is True for v in values) else None


def _private_role(evidence, role):
    """Load ONLY hash-bound, role/source-bound private native evidence."""
    item = evidence.get('private_evidence', {}).get(role)
    if item is None:
        return None
    data = Path(item['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != item['sha256']:
        raise ValueError('Private native evidence hash mismatch')
    doc = json.loads(data)
    expected_sha = evidence.get('program_sha256', {}).get(role)
    if (doc.get('schema_version') != 1 or doc.get('role') != role
            or not isinstance(expected_sha, str) or len(expected_sha) != 64
            or doc.get('program_sha256') != expected_sha):
        raise ValueError('Private native evidence role/source mismatch')
    return doc


def _contract(doc):
    if doc is None:
        return None
    value = doc.get('contract')
    if (not isinstance(value, dict) or not isinstance(value.get('parts'), list)
            or not all(isinstance(p, str) and p for p in value['parts'])
            or len(set(value['parts'])) != len(value['parts'])
            or not isinstance(value.get('joints'), dict)):
        return None
    for joint in value['joints'].values():
        if (not isinstance(joint, dict) or not isinstance(joint.get('type'), str)
                or not isinstance(joint.get('part'), str)
                or not isinstance(joint.get('limits'), list) or len(joint['limits']) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in joint['limits'])
                or joint['limits'][0] > joint['limits'][1]):
            return None
    return {**value, 'parts': sorted(value['parts'])}


def _meshes(doc):
    """None is a missing/invalid export; {} is a measured empty scene."""
    if doc is None:
        return None
    states = doc.get('states', {})
    if not isinstance(states, dict) or set(states) != set(STATES):
        return None
    geometry = {state: _mesh_export(states[state].get('geometry'))
                if isinstance(states[state], dict) else None for state in STATES}
    return None if any(value is None for value in geometry.values()) else geometry


def _mesh_export(export):
    """Decode one actual snapshot, including additional private history probes."""
    try:
        rows = export['parts']
        if not isinstance(rows, list):
            return None
        parts = {}
        for row in rows:
            vertices = decode_positions(row['positions_b64'])
            faces = decode_indices(row['indices_b64'])
            if (not isinstance(row['name'], str) or not row['name'] or row['name'] in parts
                    or row['vertex_count'] != len(vertices) or row['triangle_count'] != len(faces)
                    or not len(vertices) or not len(faces) or not np.isfinite(vertices).all()
                    or faces.max() >= len(vertices)):
                return None
            parts[row['name']] = PartMesh(vertices, faces)
    except (KeyError, ValueError, TypeError):
        return None
    return parts


def _topology(docs):
    histories = {role: snapshots(doc) for role, doc in docs.items()}
    if (any(value is None for value in histories.values())
            or set(histories['reference']) != set(histories['candidate'])):
        return None
    result = {}
    for state in histories['reference']:
        roles = {role: _mesh_export(history[state].get('geometry')) for role, history in histories.items()}
        if any(parts is None for parts in roles.values()) or not roles['reference']:
            return None
        result[state] = {role: {part: {'topology': topology(mesh), 'crossings': self_crossings(mesh),
                                      'components': component_count(mesh)}
                                for part, mesh in parts.items()} for role, parts in roles.items()}
    return result


def _topology_predicate(measured, parts, kind):
    result = {'checker_id': 'native.mesh_topology', 'passed': None}
    if measured is None or not parts:
        return result
    values = []
    for roles in measured.values():
        for part in parts:
            a, b = roles['reference'].get(part), roles['candidate'].get(part)
            if a is None:
                values.append(None)
                continue
            if b is None:
                values.append(False)
                continue
            if kind == 'geometry.open_volume':
                # A nonclosed reference needs the original closed-component
                # selector; a whole-part edge count cannot substitute for it.
                keys = ('boundary_edges', 'nonmanifold_edges', 'degenerate_faces')
                values.append(all(b['topology'][k] == 0 for k in keys)
                              if all(a['topology'][k] == 0 for k in keys) else None)
            elif kind == 'geometry.duplicate_part':
                values.append(b['components'] <= a['components'] and
                              b['topology']['duplicate_unoriented_faces'] <= a['topology']['duplicate_unoriented_faces'])
            else:
                x, y = a['crossings'], b['crossings']
                values.append(y['count'] == 0 if x['complete'] and x['count'] == 0 and y['complete'] else None)
    return {**result, 'passed': _join(values),
            'scope': 'exact_position_edge_incidence_or_proper_nonadjacent_noncoplanar_crossings'}


def _native_pose(a, b, parts):
    if a is None or b is None or not parts or not set(parts) <= set(a):
        return None
    values = []
    for part in parts:
        if part not in b:
            values.append(False)
        elif a[part].vertices.shape != b[part].vertices.shape or not np.array_equal(a[part].faces, b[part].faces):
            values.append(surface_match(a[part], b[part], 2e-6)['passed'])
        else:
            values.append(bool(np.max(abs(a[part].vertices - b[part].vertices)) <= 2e-6))
    return _join(values)


def _temporal(docs, parts, *, require_motion=False):
    result = {'checker_id': 'native.absolute_time_history', 'passed': None}
    readbacks = {role: (doc or {}).get('initial', {}) for role, doc in docs.items()}
    callbacks = [readbacks[role].get('tick_callback_present') for role in ('reference', 'candidate')]
    if any(type(value) is not bool for value in callbacks):
        return result
    if callbacks[0] is False:
        return {**result, 'passed': None if require_motion else not callbacks[1]}
    if callbacks[1] is False:
        return {**result, 'passed': False}
    ref, cand = docs['reference'], docs['candidate']
    times = ref.get('time_sequence')
    if (not isinstance(times, list) or len(times) != 3 or cand.get('time_sequence') != times
            or not all(type(t) in (int, float) and math.isfinite(t) for t in times)
            or not 0 <= times[0] < times[1] < times[2]):
        return {**result, 'reason': 'Missing authored absolute-time history'}
    primary = {role: _meshes(doc) for role, doc in docs.items()}
    if any(value is None for value in primary.values()):
        return result
    if require_motion and not any(_native_pose(primary['reference']['zero'],
            primary['reference'][state], parts) is False for state in ('mid', 'limit')):
        return {**result, 'reason': 'Reference motion not established at authored timestamps'}
    direct = {}
    for role, doc in docs.items():
        endpoint = doc.get('direct_time_endpoint', {})
        if endpoint.get('time_seconds') != times[-1]:
            return result
        direct[role] = _mesh_export(endpoint.get('geometry'))
    reference_history_valid = _native_pose(primary['reference']['limit'], direct['reference'], parts)
    if reference_history_valid is not True:
        return {**result, 'reason': 'Reference absolute-time consistency not established'}
    values = [_native_pose(primary['reference'][state], primary['candidate'][state], parts) for state in STATES]
    values += [_native_pose(primary['candidate']['limit'], direct['candidate'], parts)]
    return {**result, 'passed': _join(values), 'tolerance_m': 2e-6,
            'scope': 'authored_three_times_plus_direct_same_final_absolute_time'}


def _alignment(evidence, docs):
    geometries = {role: _meshes(doc) for role, doc in docs.items()}
    reference, candidate = geometries['reference'], geometries['candidate']
    if (reference is None or candidate is None or any(not reference[s] for s in STATES)):
        return None
    checker = GeometryAlignmentChecker(**SCORING_CONTRACT['geometry'])
    values = {role: Evidence(evidence['program_sha256'][role], True, None,
                            geometry, {}, {}) for role, geometry in geometries.items()}
    return checker.check(values['candidate'], values['reference'])


def _part_alignment(label, alignment, reference):
    result = {'passed': None, 'checker_id': 'geometry.alignment'}
    if alignment is None or reference is None:
        return {**result, 'reason': 'Incomplete three-state reference/candidate geometry'}
    parts = set(label.get('parts') or [])
    if label['defect_id'] == 'geometry.global_scale' or (
            not parts and label['defect_id'] == 'geometry.anisotropic_scale'):
        parts = set(reference['parts'])
    if not parts or not parts <= set(reference['parts']):
        return {**result, 'reason': 'Missing or unknown target part IDs'}
    invariants = [inv for inv in alignment.invariants if set(inv.parts) & parts]
    if not invariants or any(not any(inv.state == state and part in inv.parts for inv in invariants)
                             for state in STATES for part in parts):
        return {**result, 'reason': 'Incomplete target-part sweep'}
    return {**result, 'passed': all(inv.passed for inv in invariants),
            'invariants': [to_dict(inv) for inv in invariants]}


def measure_replay(evidence, labels, context, scopes=None):
    """Compute geometry once, preserve each label instance and regression scope."""
    docs = {role: _private_role(evidence, role) for role in ('reference', 'candidate')}
    scopes = [None] * len(labels) if scopes is None else scopes
    if not isinstance(scopes, list) or len(scopes) != len(labels):
        raise ValueError('Private checker scope/label instance roster mismatch')
    histories = {role: snapshots(doc) for role, doc in docs.items()}
    histories = {role: None if history is None else {
        state: _mesh_export(row.get('geometry')) if isinstance(row, dict) else None
        for state, row in history.items()} for role, history in histories.items()}
    contracts = {role: _contract(doc) for role, doc in docs.items()}
    reference, candidate = contracts['reference'], contracts['candidate']
    contract_check = _check_contract(candidate, reference) if reference and candidate else None
    alignment = _alignment(evidence, docs)
    topology_evidence = _topology(docs)
    all_parts = (reference or {}).get('parts', [])
    topology_checks = [_topology_predicate(topology_evidence, all_parts, kind)['passed'] for kind in TOPOLOGY_DEFECTS]
    property_checks = {'structure.hierarchy': hierarchy(docs['reference'], docs['candidate'], all_parts),
        'material.properties': materials(docs['reference'], docs['candidate']),
        'runtime.lifecycle': lifecycle(docs['candidate']),
        'runtime.temporal': _temporal(docs, all_parts),
        'geometry.feature_shapes': feature_regressions(histories),
        'structure.contacts': contact_regressions(histories)}
    caps = (context.get('max_triangles'), context.get('max_draw_calls'))
    budgets = {kind: runtime_budget_predicate({'defect_id': kind}, evidence,
                    max_triangles=caps[0], max_draw_calls=caps[1])
               if all(type(v) is int and v >= 0 for v in caps)
               else {'passed': None, 'checker_id': 'runtime.budget',
                     'reason': 'Missing explicit public task caps'} for kind in BUDGET_DEFECTS}
    checks = {name: None for name in SCORING_CONTRACT['required_checks']}
    checks.update({'compile_render': (evidence.get('gate') or {}).get('passed'),
        'program.contract': contract_check['passed'] if contract_check else None,
        'geometry.alignment': alignment.passed if alignment else None,
        'runtime.budget': _join(row['passed'] for row in budgets.values())})
    checks['geometry.topology'] = _join(topology_checks)
    checks.update({key: value['passed'] for key, value in property_checks.items()})
    measurements = []
    for label, scope in zip(labels, scopes, strict=True):
        kind = label['defect_id']
        if kind in (*FEATURE_DEFECTS, *CONTACT_DEFECTS):
            item = measure_geometry(kind, scope, histories, evidence['program_sha256'])
        elif kind in BUDGET_DEFECTS:
            item = budgets[kind]
        elif kind in MATERIAL_FIELDS:
            item = materials(docs['reference'], docs['candidate'], label)
        elif kind in TOPOLOGY_DEFECTS:
            item = _topology_predicate(topology_evidence, label.get('parts') or [], kind)
        elif kind == 'structure.wrong_parent':
            item = hierarchy(docs['reference'], docs['candidate'], label.get('parts') or [])
        elif kind == 'runtime.missing_dispose':
            item = property_checks['runtime.lifecycle']
        elif kind == 'runtime.tick_missing_or_phase':
            item = _temporal(docs, label.get('parts') or [], require_motion=True)
        elif kind == 'geometry.missing_part':
            parts = set(label.get('parts') or [])
            geometry = _meshes(docs['candidate'])
            valid = bool(parts) and reference is not None and parts <= set(reference['parts'])
            passed = None
            if valid and candidate is not None:
                if not parts <= set(candidate['parts']):
                    passed = False
                elif geometry is not None:
                    passed = all(parts <= set(geometry[state]) for state in STATES)
            item = {'checker_id': 'program.part_presence', 'passed': passed}
        elif kind in CONTRACT_DEFECTS:
            parts = set(label.get('parts') or [])
            joint_id = label.get('joint_id') or label.get('params', {}).get('joint_id')
            joints = ([joint_id] if joint_id and reference and joint_id in reference['joints']
                      else [key for key, joint in (reference or {}).get('joints', {}).items()
                            if joint['part'] in parts] if not joint_id else [])
            field = 'type' if kind == 'kinematics.joint_type' else 'limits'
            matched = None
            if joints and contract_check:
                matched = all(key in candidate['joints'] and
                              f'joint:{key}:{field}' not in contract_check['differences']
                              and f'joint:{key}:part' not in contract_check['differences']
                              for key in joints)
            # Per-label metadata resolution is independent of another geometry
            # defect on that part. Global alignment still gates final Success.
            item = {'checker_id': 'program.joint_' + field, 'passed': matched, 'joints': joints}
        elif kind in ALIGNMENT_DEFECTS:
            item = _part_alignment(label, alignment, reference)
        else:
            item = {'passed': None, 'checker_id': None, 'reason': 'Native predicate not implemented'}
        measurements.append(item)
    return {'measurements': measurements, 'checks': checks,
            'diagnostics': {'contract': contract_check,
                            'alignment': to_dict(alignment) if alignment else None,
                            'native_topology': topology_evidence, 'native_properties': property_checks,
                            'budgets': budgets}}


def make_audit(case, replay):
    labels = case.get('private_labels')
    if not labels:
        raise ValueError('Native evaluator requires hash-bound private label instances')
    last = {}

    def measured(source):
        evidence = replay(source)
        sha = hashlib.sha256(source.encode()).hexdigest()
        if evidence.get('program_sha256', {}).get('candidate') != sha or evidence.get('status') != 'complete':
            raise ValueError('Native scoring requires complete current-source replay')
        # A failed compile/render is handled by make_repair_audit without mesh IO.
        if (evidence.get('gate') or {}).get('passed') is not False:
            last['scores'] = measure_replay(evidence, labels, case.get('public_context') or {},
                                            case.get('private_scoring_scopes'))
        else:
            last['scores'] = None
        return evidence

    def predicate(label, evidence):
        # Labels are the original list objects passed by make_repair_audit;
        # identity retains distinct, even byte-identical, injected instances.
        index = next(i for i, row in enumerate(labels) if row is label)
        return last['scores']['measurements'][index]

    callbacks = {kind: predicate for kind in SUPPORTED}
    program = {name: lambda evidence, key=name: last['scores']['checks'][key]
               for name in SCORING_CONTRACT['required_checks']}
    exposure = {i: classify_exposure(i, case.get('private_exposure') or {},
                                    case.get('private_exposure_history') or []) for i in range(len(labels))}
    base = make_repair_audit(measured, labels, callbacks, program,
                            SCORING_CONTRACT['required_checks'], exposure=exposure)

    def audit(source):
        result = base(source)
        result['scoring_contract'] = SCORING_CONTRACT
        result['diagnostics'] = last['scores']['diagnostics'] if last['scores'] else {}
        result['measurements'] = last['scores']['measurements'] if last['scores'] else []
        result['table1_eligible'] = False  # This module's contract is development.
        return result
    return audit


def make_gold(case, replay):
    """Current injected-defect oracle; private sources/measurements stay private.

    Gold citations name checked current states, not a new exposure annotation.
    Regressions without an attributable original label remain uncertain.
    """
    audit = make_audit(case, replay)

    def gold(source, public_evidence):
        from turnitover.oracle.supervision_validation import public_payload
        payload = public_payload(public_evidence['public_record'])
        report = audit(source)
        context, history = payload['context'], payload['history']
        findings = []
        for defect, measurement in zip(report['defects'], report.get('measurements', [])):
            if defect['resolved'] is not False:
                continue
            label = case['private_labels'][defect['label_index']]
            joint = label.get('joint_id') or label.get('params', {}).get('joint_id')
            if defect['defect_id'] == 'kinematics.joint_axis' and joint not in context['joints']:
                continue
            states = {row['state'] for key in ('details', 'invariants', 'mismatches')
                      for row in measurement.get(key, []) if isinstance(row, dict) and 'state' in row
                      and row.get('passed') is not True}
            steps = [row['step'] for row in history if not states or row.get('state') in states
                     or row.get('state') == 'midpoint' and 'mid' in states]
            if not steps:
                continue
            findings.append({'defect_id': defect['defect_id'],
                'parts': [p for p in defect['parts'] if p in context['parts']],
                'joint_id': joint if defect['defect_id'] == 'kinematics.joint_axis' else None,
                'severity': 1., 'confidence': 1., 'evidence_steps': steps,
                'description': 'Independent current-program check finds ' + defect['defect_id'] + ' unresolved.',
                'suggested_fix': 'Restore the affected ' + defect['defect_id'].split('.')[-1] +
                    ' property while preserving other parts and behavior.'})
        status = 'fail' if findings else 'pass' if report['passed'] is True else 'uncertain'
        return {'status': status, 'confidence': 1. if status != 'uncertain' else 0.,
                'findings': findings, 'summary': 'Current-program privileged checker diagnosis.',
                'limitations': ['Privileged diagnosis; cited steps identify replay context, not visual exposure.'] +
                    (['Unresolved or unsupported program checks have no attributable original finding.']
                     if status == 'uncertain' else [])}
    return gold
