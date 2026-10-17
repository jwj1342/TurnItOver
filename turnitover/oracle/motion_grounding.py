"""Deployable, label-blind comparison of matched RGB observations.

Temporal residuals describe image changes, not 3D rotations: offsets, occlusion,
lighting and wrong axes may all produce a residual. No geometry enters this API.
"""
from __future__ import annotations

import numpy as np
from turnitover.teacher.witnesses import core_changed_fraction

THRESHOLD = 0.001


def compare_observations(pairs, states, rest_value):
    if len(pairs) != len(states) or not pairs:
        raise ValueError("One RGB pair per public state required")
    steps = [s['step'] for s in states]
    if len(set(steps)) != len(steps):
        raise ValueError("Duplicate step IDs")
    static, temporal = [], []
    for (reference, candidate), state in zip(pairs, states, strict=True):
        r, c = np.asarray(reference).astype(np.float32), np.asarray(candidate).astype(np.float32)
        if r.shape != c.shape or r.ndim != 3 or r.shape[2] != 3:
            raise ValueError("Matched RGB dimensions required")
        mask = np.max(np.abs(r-c), axis=2) > 255 * .05
        ys, xs = np.nonzero(mask)
        bbox = None if not len(xs) else [round(float(v), 3) for v in
            (xs.min()/r.shape[1], ys.min()/r.shape[0], (xs.max()+1)/r.shape[1], (ys.max()+1)/r.shape[0])]
        static.append({'step': state['step'], 'core_fraction': round(core_changed_fraction(r,c),6), 'bbox_xyxy':bbox})
    for i, state in enumerate(states):
        if abs(state['actual_value'] - rest_value) < 1e-6:
            continue
        starts = [j for j,s in enumerate(states) if s['view']==state['view'] and abs(s['actual_value']-rest_value)<1e-6]
        if not starts:
            continue
        j=starts[0]
        r0,c0=(np.asarray(x).astype(np.float32) for x in pairs[j])
        r1,c1=(np.asarray(x).astype(np.float32) for x in pairs[i])
        temporal.append({'steps':[states[j]['step'],state['step']],
            'reference_change':round(core_changed_fraction(r0,r1),6),
            'candidate_change':round(core_changed_fraction(c0,c1),6),
            'change_residual':round(core_changed_fraction(r1-r0,c1-c0),6)})
    return {'static':static,'temporal':temporal,'threshold':THRESHOLD}


def visible_comparison(features, states, rest_value):
    rest = {s['step'] for s in states if abs(s['actual_value']-rest_value)<1e-6}
    changed = [s['step'] for s in features['static'] if s['core_fraction']>=THRESHOLD]
    motion = features['temporal']
    moving = any(max(t['reference_change'],t['candidate_change'])>=THRESHOLD for t in motion)
    return {'static_rest':'different' if rest.intersection(changed) else 'matching',
            'static_steps':changed,
            'motion_change':('different' if any(t['change_residual']>=THRESHOLD for t in motion) else 'matching') if moving else 'unobserved',
            'motion_pairs':[t['steps'] for t in motion if t['change_residual']>=THRESHOLD]}


def observable_signature(comparison):
    """Restricted single-corruption probe; NOT a general defect classifier."""
    if comparison['static_rest']=='different':
        return 'offset'
    if comparison['motion_change']=='unobserved':
        return 'uncertain'
    if comparison['motion_change']=='different':
        return 'axis'
    return 'clean'
