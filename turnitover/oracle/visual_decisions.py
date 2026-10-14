"""Conservative stopping labels computed only from public, aligned rest images.

This is a static shape-consistency subtask, not articulation diagnosis. No gold,
corruption, reference program, unobserved candidate image or future label is input.
"""
from __future__ import annotations

import json
import numpy as np
from scipy.ndimage import label

from turnitover.oracle.evidence import pixels

VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
ORDERS = (VIEWS, VIEWS[2:] + VIEWS[:2], VIEWS[4:] + VIEWS[:4])
BUDGET = 6
INSTRUCTION = '''Inspect static shape consistency against the six supplied reference images.
All references and observations use identical framing, lighting and rest joint state.
Reference images are ordered front, back, left, right, top, bottom. An observation
is comparable only to its named reference view. Stop/fail when the observed shape
visibly differs. Otherwise continue inspecting unseen views while budget remains;
at budget exhaustion stop/uncertain, never infer pass or correct articulation.
Return only a stop_decision JSON object. This first_failure objective concerns
visible static shape differences, not motion, material or hidden defect labels.'''


def image_evidence(reference: bytes, observed: bytes) -> dict:
    a, b = pixels(reference), pixels(observed)
    if a.shape != b.shape:
        raise ValueError('Aligned images must have equal dimensions')
    if np.array_equal(a, b):
        return {'state': 'identical', 'bbox': None, 'changed_pixels': 0}
    # Background is the corner colour. Silhouette disagreement excludes pure
    # shading differences; connected support excludes isolated raster noise.
    bg = a[0, 0].astype(float)
    if not np.array_equal(a[0, 0], b[0, 0]):
        return {'state': 'ambiguous', 'bbox': None, 'changed_pixels': 0}
    ma = np.max(np.abs(a.astype(float) - bg), axis=2) > 20
    mb = np.max(np.abs(b.astype(float) - bg), axis=2) > 20
    diff = ma ^ mb
    components, _ = label(diff)
    sizes = np.bincount(components.ravel())[1:]
    count = int(diff.sum())
    sufficient = count >= 64 and count / max(1, int((ma | mb).sum())) >= .01 and max(sizes, default=0) >= 16
    y, x = np.where(diff)
    return {'state': 'sufficient' if sufficient else 'ambiguous',
            'bbox': [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1] if len(x) else None,
            'changed_pixels': count, 'largest_component': int(max(sizes, default=0))}


def decide(evidence, remaining):
    sufficient = [i for i, e in enumerate(evidence) if e['state'] == 'sufficient']
    if sufficient:
        return 'stop', 'fail', 'evidence_sufficient', sufficient
    if any(e['state'] != 'identical' for e in evidence):
        return None  # Do not manufacture continue from an uncertain difference.
    if remaining == 0:
        return 'stop', 'uncertain', 'budget_exhausted', []
    return 'continue', None, 'further_evidence_required', []


def action_candidates(context, history, remaining):
    """Ex ante coverage gain, never future-image gain or a privileged action target."""
    seen = {h['action'].get('view_id') for h in history}
    return [{'action': {'type': 'request_view', 'view_id': view},
             'legal': remaining > 0 and view in context['views'],
             'unseen_view_gain': int(view not in seen),
             'information_gain': 'unknown_without_execution',
             'eligible': remaining > 0 and view in context['views'] and view not in seen}
            for view in VIEWS]


def make_row(uid, order_id, context, refs, history, evidence, identity):
    from turnitover.oracle.motion_decisions import MOTION_INSTRUCTION, candidates
    motion = 'reference_states' in context
    step = len(history)
    target = decide(evidence, BUDGET-step)
    if target is None:
        return None
    decision, outcome, reason, steps = target
    images = refs + [h['image_ref'] for h in history]
    # Public paths are positional aliases, avoiding variant/asset ID shortcuts.
    public_history = [{**h, 'image_ref': f'observation_{i}.png'} for i, h in enumerate(history)]
    payload = {'context': context, 'remaining': BUDGET-step, 'history': public_history,
               'image_order': ([f'reference_{i}.png' for i in range(len(refs))] if motion else [f'reference_{v}.png' for v in VIEWS]) + [h['image_ref'] for h in public_history],
               'training_objective': 'first_failure'}
    reasons = {'evidence_sufficient': 'The acquired image contains a visible shape mismatch against its aligned reference.',
               'further_evidence_required': 'The acquired views match their references; unseen views still require inspection.',
               'budget_exhausted': 'The observation budget is exhausted without established shape mismatch; articulation remains unverified.'}
    answer = {'stop_decision': {'decision': decision, 'outcome': outcome, 'reason': reasons[reason]}}
    return {'id': f'{uid}:order{order_id}:stop:{step}', 'episode_id': f'{uid}:order{order_id}',
            'source_sample_id': uid, 'decision_index': step, 'training_objective': 'first_failure',
            'condition': 'visible_articulated_shape_v5' if motion else 'visible_static_shape_v5', 'terminal': decision == 'stop',
            'stop_target': decision, 'stop_outcome': outcome, 'stop_reason': reason,
            'termination_kind': 'forced' if step == BUDGET else 'active' if outcome == 'fail' else None,
            'evidence_steps': steps, 'evidence': evidence, 'images': images, 'data_identity': identity,
            'action_audit': (candidates if motion else action_candidates)(context, history, BUDGET-step),
            'messages': [{'role': 'user', 'content': (MOTION_INSTRUCTION if motion else INSTRUCTION) + '\n' + json.dumps(payload)},
                         {'role': 'assistant', 'content': json.dumps(answer)}]}


def deduplicate(rows):
    """Keep one label per exact public text and image-content sequence."""
    seen, result = {}, []
    for row in rows:
        key = (row['messages'][0]['content'], tuple(row['image_sha256']))
        target = row['messages'][1]['content']
        if key in seen:
            if seen[key] != target:
                raise ValueError('Conflicting identical public inputs')
            continue
        seen[key] = target
        result.append(row)
    return result
