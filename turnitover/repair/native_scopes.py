"""Recover private checker selectors from hash-checked construction reports.

Reports provide identifiers and scope only. Historical pass/fail fields never
become measurements of an edited program. Test records are not accepted here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from turnitover.repair.native_geometry import FEATURE_DEFECTS, CONTACT_DEFECTS


def _feature_tolerance(proof):
    if not isinstance(proof, dict):
        return None
    for key in ('tolerance_m', 'tolerance'):
        value = proof.get(key)
        if type(value) in (int, float) and math.isfinite(value) and value > 0:
            return value
    for key in ('native_profile', 'strict_local_native_edge_proof'):
        value = _feature_tolerance(proof.get(key))
        if value is not None:
            return value
    return None


def _feature_depth(proof):
    if isinstance(proof, dict):
        for key in ('depth', 'depth_m', 'bevel_depth', 'chamfer_depth'):
            value = proof.get(key)
            if type(value) in (int, float) and math.isfinite(value) and value > 0:
                return value
        for key in ('native_profile', 'strict_local_native_edge_proof'):
            value = _feature_depth(proof.get(key))
            if value is not None:
                return value
    return None


def construction_scopes(annotation, reference_sha256, *, resolve=Path):
    source = annotation.get('source_roster_row', {})
    split = source.get('split', annotation.get('split'))
    if split not in {'train', 'validation'}:
        raise ValueError('Private scoring scope recovery accepts Train/Validation only')
    labels = deepcopy(annotation['labels'])
    provenance = []

    def read(path, expected=None):
        raw = resolve(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError('Construction scope report hash mismatch')
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('Construction scope report must be an object')
        provenance.append({'path': str(path), 'sha256': digest})
        return value

    if not source.get('report_path') or not source.get('report_sha256'):
        return {'labels': labels, 'scopes': [None] * len(labels), 'provenance': [],
                'missing_reason': 'No hash-bound original construction report'}
    report = read(source['report_path'], source['report_sha256'])
    native = report
    companion = report.get('native_collection_report_path')
    if companion:
        native = read(companion, report.get('native_collection_report_sha256'))
    elif not native.get('target'):
        # Public-view wrappers retain the original programs in another folder.
        # Follow that collector ONLY when the wrapper preregistration binds the
        # exact report and both original program hashes. No target is guessed.
        programs = annotation.get('programs', {})
        candidate = programs.get('case_candidate', {})
        reference = programs.get('case_reference', {})
        if candidate.get('path') and reference.get('path'):
            collector = Path(candidate['path']).with_name('report.private.json').as_posix()
            registration = Path(source['report_path']).with_name('preregistration.private.json').as_posix()
            if resolve(registration).is_file() and resolve(collector).is_file():
                prereg = read(registration)
                hashes = prereg.get('inputs_sha256', {})
                if (hashes.get(candidate['path']) == candidate.get('sha256')
                        and hashes.get(reference['path']) == reference_sha256
                        and hashes.get(collector)):
                    native = read(collector, hashes[collector])
    scopes = []
    for label in labels:
        kind = label['defect_id']
        target = native.get('target')
        # Collectors use several explicit selector spellings. Recover only
        # identifiers for this defect, never infer them from saved outcomes.
        native_kind = native.get('defect_id', native.get('primary_defect'))
        matching = native_kind == kind or (native_kind is None and
                    {item['defect_id'] for item in labels} == {kind})
        if matching:
            condition = native.get('condition') or {}
            if target is None:
                target = condition.get('target') or condition.get('part')
            if target is None and kind == 'geometry.duplicate_part':
                target = native.get('duplicate_target')
            if kind.startswith('kinematics.'):
                joint = native.get('joint')
                params = label.get('params') or {}
                if isinstance(joint, str) and joint and not (
                        label.get('joint_id') or params.get('joint_id') or params.get('joint')):
                    label['joint_id'] = joint
                subtree = native.get('target_subtree')
                if (not label.get('parts') and isinstance(subtree, list) and subtree
                        and all(isinstance(part, str) and part for part in subtree)):
                    label['parts'] = list(subtree)
        if isinstance(target, str) and target and not label.get('parts'):
            if matching:
                label['parts'] = [target]
        scope = {'defect_id': kind, 'reference_program_sha256': reference_sha256,
                 'provenance': list(provenance), 'status': 'development'}
        if kind == 'geometry.thin_sheet' and isinstance(target, str):
            # Exact native restoration is sufficient. Remeshed repairs use the
            # evaluator's versioned reference-relative thickness/surface defaults.
            scope.update(target={'part': target}, tolerance_m=0,
                         native_mutation=native.get('mutation_witness', {}),
                         acceptance='native_restoration_or_reference_relative_thickness_and_surface')
        elif kind == 'geometry.missing_bevel' and isinstance(target, str):
            proof = native.get('native_component_chamfer_proof', {})
            raw = proof.get('raw_component_chamfer_proof', {})
            lift = proof.get('lift', [])
            tolerance = _feature_tolerance(raw)
            if not lift or tolerance is None:
                scopes.append(None)
                continue
            scope.update(target={'part': target, 'vertex_ranges': [
                [r['native_vertex_start'], r['native_vertex_count']] for r in lift]},
                tolerance_m=tolerance, source_component=native.get('source_component'),
                feature_depth_m=_feature_depth(raw),
                acceptance='source_component_surface_and_bevel_normal_profile')
        elif kind in CONTACT_DEFECTS:
            condition = native.get('condition', {})
            support = native.get('support')
            if isinstance(target, dict) and isinstance(support, dict):
                # Native collector partitions the selected original component
                # into this explicitly named mesh in BOTH reference/candidate.
                # Remove the selected original offsets from the support map.
                removed = target.get('source_face_ids', [])
                support_ids = support.get('source_face_ids', [])
                if not removed or not support_ids or target.get('part') != support.get('part'):
                    scopes.append(None)
                    continue
                scope.update(target={'part': 'native-contact-surface'},
                    support={'part': support['part'], 'face_ids': [
                        i - sum(j < i for j in removed) for i in support_ids]},
                    original_component_binding={'target': target, 'support': support})
                if not label.get('parts'):
                    label['parts'] = ['native-contact-surface']
            elif isinstance(target, str) and condition.get('target') == target and condition.get('support'):
                scope.update(target={'part': target, 'face_ids': condition.get('target_source_face_ids')},
                    support={'part': condition['support'], 'face_ids': condition.get('support_source_face_ids')})
            elif kind == 'structure.floating' and isinstance(target, str):
                contacts = native.get('reference_actual_contact', [])
                if len(contacts) != 1:
                    scopes.append(None)
                    continue
                contact = contacts[0]
                if contact.get('kind') == 'exact_positive_area_coplanar_assembly_contact':
                    proofs = [p for p in contact.get('proofs', []) if p.get('coordinate_mode') == 'runtime_float32']
                    if len(proofs) != 1:
                        scopes.append(None)
                        continue
                    left, right = proofs[0]['target_raw_face'], proofs[0]['other_raw_face']
                elif contact.get('kind') == 'exact_strict_native_assembly_crossing':
                    proof = contact['original_world_triangles']
                    left, right = proof['left_raw_face'], proof['right_raw_face']
                else:
                    scopes.append(None)
                    continue
                scope.update(target={'part': target, 'face_ids': [left]},
                    support={'part': contact['other_part'], 'face_ids': [right]},
                    acceptance='original_geometric_assembly_connection_not_gravity_support')
            else:
                scopes.append(None)
                continue
        else:
            scopes.append(None)
            continue
        scopes.append(scope)
    return {'labels': labels, 'scopes': scopes, 'provenance': provenance}
