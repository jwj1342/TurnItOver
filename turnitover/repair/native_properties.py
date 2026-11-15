"""Score actual native material, hierarchy and owned-resource readbacks.

These predicates use the original construction checks' observed fields, not
source markers, callback presence alone or saved pre-repair verdicts.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


MATERIAL_FIELDS = {
    'material.basecolor_as_rough': (('roughness',), ('roughnessMap',), ('roughness', 'uses_roughnessmap')),
    'material.metalness': (('metalness',), ('metalnessMap',), ('metalness', 'uses_metalnessmap')),
    'material.tiling_scale': ((), (), ()),
    'material.emissive': (('emissive', 'emissiveIntensity'), ('emissiveMap',), ('emissive', 'uses_emissivemap')),
    'material.normal_map': (('normalScale', 'normalMapType'), ('normalMap',), ('normalScale', 'uses_normalmap')),
}
PROPERTIES = ('color', 'specular', 'shininess', 'roughness', 'metalness', 'ior',
    'specularColor', 'specularIntensity', 'opacity', 'transparent', 'depthWrite', 'vertexColors',
    'emissive', 'emissiveIntensity', 'normalScale', 'normalMapType', 'side', 'flatShading')
TEXTURES = ('map', 'roughnessMap', 'metalnessMap', 'normalMap', 'emissiveMap',
            'aoMap', 'alphaMap', 'bumpMap', 'displacementMap', 'lightMap', 'envMap')
UV_ATTRIBUTES = ('uv', 'uv1', 'uv2')
TEXTURE_TRANSFORM = ('repeat', 'offset', 'center', 'rotation', 'matrix',
                     'matrixAutoUpdate', 'wrapS', 'wrapT', 'channel')
ATTRIBUTE_PROFILE_TOLERANCE = {'rtol': .02, 'atol': 1e-6}


def join(values):
    values = list(values)
    return False if False in values else True if values and all(v is True for v in values) else None


def snapshots(doc):
    if not isinstance(doc, dict):
        return None
    values = doc.get('states', {})
    if not isinstance(values, dict) or set(values) != {'zero', 'mid', 'limit'}:
        return None
    values = dict(values)
    if 'return_to_zero' in doc:
        values['return_to_zero'] = doc['return_to_zero']
    if 'direct_time_endpoint' in doc:
        values['direct_time_endpoint'] = doc['direct_time_endpoint']
    if 'authored_history' in doc:
        history = doc['authored_history']
        if (not isinstance(history, dict) or history.get('complete') is not True
                or not isinstance(history.get('snapshots'), list) or not history['snapshots']):
            return None
        values.update({f'authored:{i}': row for i, row in enumerate(history['snapshots'])})
    return values


def material_slots(snapshot):
    readback = snapshot.get('readback') if isinstance(snapshot, dict) else None
    if not isinstance(readback, dict) or readback.get('schema_version') != 1:
        return None
    meshes = readback.get('meshes')
    if not isinstance(meshes, list):
        return None
    slots = {}
    for mesh in meshes:
        if (not isinstance(mesh, dict) or not isinstance(mesh.get('part'), str)
                or not isinstance(mesh.get('mesh'), str) or mesh.get('supported_geometry') is not True
                or not isinstance(mesh.get('materials'), list) or not isinstance(mesh.get('attributes'), dict)):
            return None
        for index, material in enumerate(mesh['materials']):
            key = (mesh['part'], mesh['mesh'], index)
            if (key in slots or not isinstance(material, dict) or not isinstance(material.get('type'), str)
                    or not isinstance(material.get('properties'), dict)
                    or not isinstance(material.get('textures'), dict) or not isinstance(material.get('gpu'), dict)):
                return None
            slots[key] = {**material, 'attributes': mesh['attributes']}
    return slots


def _texture(a, b, *, transform_only=False):
    if a is None or b is None:
        return a is b
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    if transform_only:
        if any(k not in value for value in (a, b) for k in TEXTURE_TRANSFORM):
            return None
        return all(a[k] == b[k] for k in TEXTURE_TRANSFORM)
    if a.get('complete') is not True or b.get('complete') is not True:
        return None
    return a == b


def _attribute(a, b):
    if a == b:
        return True
    if a is None or b is None:
        return False
    if a.get('complete') is False or b.get('complete') is False:
        return None
    if a.get('itemSize') != b.get('itemSize'):
        return False
    left, right = a.get('profile'), b.get('profile')
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False  # Older readbacks retain their original strict comparison.
    for key in ('mean', 'second', 'position_cross'):
        try:
            x, y = (np.asarray(profile.get(key, []), dtype=float) for profile in (left, right))
        except (TypeError, ValueError):
            return None
        if not x.size or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            return None
        if not np.allclose(x, y, **ATTRIBUTE_PROFILE_TOLERANCE):
            return False
    return True


def _slot(a, b, kind):
    checks = [a['type'] == b['type']]
    if kind != 'material.tiling_scale':
        checks.append(True if all(v['gpu'].get('program_present') is True for v in (a, b)) else None)
    if kind is None:
        properties, textures = PROPERTIES, TEXTURES
        gpu = ('program_present', 'roughness', 'metalness', 'emissive', 'normalScale',
               'uses_roughnessmap', 'uses_metalnessmap', 'uses_normalmap', 'uses_emissivemap')
    else:
        properties, textures, gpu = MATERIAL_FIELDS[kind]
        if properties and all(a['properties'].get(key) is None for key in properties):
            checks.append(None)  # Material type has no meaningful target property.
        for key in gpu:
            if not key.startswith('uses_') and a['gpu'].get(key) is None:
                checks.append(None)
    if kind == 'material.tiling_scale':
        usable = any(isinstance(a['textures'].get(key), dict)
                     and a['textures'][key].get('complete') is True for key in TEXTURES)
        checks.append(True if usable and any(a['attributes'].get(k) is not None for k in UV_ATTRIBUTES) else None)
    if kind == 'material.normal_map' and not isinstance(a['textures'].get('normalMap'), dict):
        checks.append(None)  # A missing native normal map cannot establish its restoration.
    for group, keys in (('properties', properties), ('gpu', gpu)):
        checks.extend(a[group][key] == b[group][key] if key in a[group] and key in b[group] else None
                      for key in keys)
    for key in (TEXTURES if kind == 'material.tiling_scale' else textures):
        checks.append(_texture(a['textures'][key], b['textures'][key],
                               transform_only=kind == 'material.tiling_scale')
                      if key in a['textures'] and key in b['textures'] else None)
    if kind in (None, 'material.tiling_scale', 'material.normal_map'):
        attributes = UV_ATTRIBUTES + (('normal', 'tangent') if kind != 'material.tiling_scale' else ())
        for key in attributes:
            values = [v['attributes'].get(key) for v in (a, b)]
            unsupported = any(isinstance(v, dict) and v.get('complete') is False for v in values)
            checks.append(_attribute(*values) if not unsupported
                          and all(key in v['attributes'] for v in (a, b)) else None)
    if kind in (None, 'material.basecolor_as_rough'):
        key = 'roughness_and_basecolor_share_texture'
        checks.append(a[key] == b[key] if key in a and key in b else None)
    return join(checks)


def materials(reference, candidate, label=None):
    kind = label['defect_id'] if label else None
    result = {'checker_id': 'native.material_slots', 'passed': None,
              'scope': 'native_mesh_slots_CPU_texture_bindings_and_last_render_GPU_uniforms'}
    ref, cand = snapshots(reference), snapshots(candidate)
    if ref is None or cand is None or set(ref) != set(cand):
        return {**result, 'reason': 'Incomplete matching native material histories'}
    parts = set(label.get('parts') or []) if label else None
    if label and not parts:
        return {**result, 'reason': 'Missing target part IDs'}
    outcomes, mismatches = [], []
    for state in ref:
        a, b = material_slots(ref[state]), material_slots(cand[state])
        if a is None or b is None:
            outcomes.append(None)
            continue
        keys = {key for key in a if parts is None or key[0] in parts}
        if not keys or (parts is not None and {key[0] for key in keys} != parts):
            outcomes.append(None)
            continue
        if not label:
            outcomes.append(set(a) == set(b))
        for key in sorted(keys):
            value = _slot(a[key], b[key], kind) if key in b else False
            outcomes.append(value)
            if value is False:
                mismatches.append({'state': state, 'part': key[0], 'mesh': key[1], 'slot': key[2]})
    return {**result, 'passed': join(outcomes), 'mismatches': mismatches,
            'limitations': ['Slot correspondence assumes stable native mesh names and material slots.',
                'No arbitrary custom shader equivalence or hidden texture-closure certification.']}


def hierarchy_parents(snapshot):
    readback = snapshot.get('readback') if isinstance(snapshot, dict) else None
    payload = readback.get('hierarchy') if isinstance(readback, dict) else None
    if not isinstance(payload, dict) or payload.get('property') != 'hierarchy':
        return None
    result = {}
    def walk(node, parent, root=False):
        if not isinstance(node, dict) or not isinstance(node.get('name'), str) or not isinstance(node.get('children'), list):
            raise ValueError('Invalid hierarchy node')
        name = '@root' if root else node['name']
        if name:
            if name in result:
                raise ValueError('Ambiguous named hierarchy')
            result[name] = parent
        for child in node['children']:
            walk(child, name or parent)
    try:
        walk(payload['root'], None, True)
    except (KeyError, ValueError):
        return None
    return result


def hierarchy(reference, candidate, parts):
    ref, cand = snapshots(reference), snapshots(candidate)
    result = {'checker_id': 'native.hierarchy_parents', 'passed': None}
    if not parts or ref is None or cand is None or set(ref) != set(cand):
        return result
    outcomes = []
    for state in ref:
        a, b = hierarchy_parents(ref[state]), hierarchy_parents(cand[state])
        if a is None or b is None or not set(parts) <= set(a):
            outcomes.append(None)
        else:
            outcomes.append(all(part in b and a[part] == b[part] for part in parts))
    return {**result, 'passed': join(outcomes)}


def lifecycle(doc):
    result = {'checker_id': 'native.owned_resource_disposal', 'passed': None}
    value = doc.get('lifecycle') if isinstance(doc, dict) else None
    required = ('resources', 'events', 'active_object_null', 'root_detached',
                'result', 'before_load', 'before_dispose', 'after_dispose')
    if not isinstance(value, dict) or any(k not in value for k in required):
        return {**result, 'reason': 'Missing actual owned-resource disposal trace'}
    response = value['result']
    if (not isinstance(response, dict) or not isinstance(value['resources'], list)
            or not value['resources'] or not isinstance(value['events'], list)):
        return result
    try:
        resources = [row['uuid'] for row in value['resources']]
        events = Counter(row['uuid'] for row in value['events'])
        if any(not isinstance(key, str) or not key for key in resources + list(events)):
            return result
        memory = [value[k] for k in ('before_load', 'before_dispose', 'after_dispose')]
        if any(not isinstance(row, dict) or any(type(row.get(k)) is not int or row[k] < 0
                                                for k in ('geometries', 'textures')) for row in memory):
            return result
        passed = (response.get('ok') is True and response.get('callback_present') is True
            and 'callback_error' in response and response['callback_error'] is None
            and value['active_object_null'] is True and value['root_detached'] is True
            and len(set(resources)) == len(resources)
            and all(events[key] == 1 for key in resources)
            and all(count == 1 for count in events.values())
            and all(row['active_object_null'] is True and row['root_detached'] is True for row in value['events'])
            and all(memory[0][k] == memory[2][k] for k in ('geometries', 'textures')))
    except (KeyError, TypeError):
        return result
    authored = doc.get('authored_history')
    if authored is not None:
        extra = lifecycle({'lifecycle': authored.get('lifecycle')})
        passed = join([bool(passed), extra['passed']])
    if 'direct_time_endpoint' in doc:
        direct = lifecycle({'lifecycle': doc.get('direct_time_lifecycle')})
        passed = join([passed, direct['passed']])
    return {**result, 'passed': passed, 'observed_resources': len(resources),
            'limitations': ['Visible owned resources plus observed detached-resource events and renderer counters.',
                            'Not a JavaScript heap, hidden CPU allocation or physical GPU memory proof.']}
