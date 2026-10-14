import io
import json

import numpy as np
from PIL import Image

from turnitover.oracle.visual_decisions import image_evidence, decide, action_candidates, make_row


def png(a):
    f = io.BytesIO()
    Image.fromarray(a).save(f, format='PNG')
    return f.getvalue()


def test_public_pixels_only_and_ambiguous_quarantine():
    a = np.full((64, 64, 3), 255, dtype=np.uint8)
    a[10:50, 10:50] = 70
    b = np.roll(a, 5, axis=1)
    ev = image_evidence(png(a), png(b))
    assert ev['state'] == 'sufficient'
    assert ev['bbox'] == [10, 10, 55, 50]
    assert decide([ev], 5)[:3] == ('stop', 'fail', 'evidence_sufficient')
    same = image_evidence(png(a), png(a))
    assert decide([same], 5)[0] == 'continue'
    assert decide([same], 0)[:3] == ('stop', 'uncertain', 'budget_exhausted')
    c = a.copy()
    c[20, 20] = 80
    assert image_evidence(png(a), png(c))['state'] == 'ambiguous'
    assert decide([image_evidence(png(a), png(c))], 5) is None


def test_action_gain_is_ex_ante_coverage_not_future_evidence():
    h = [{'action': {'view_id': 'front'}}]
    candidates = action_candidates({'views': ['front', 'back']}, h, 3)
    assert not candidates[0]['eligible']  # repeated view
    assert candidates[1]['eligible']
    assert not candidates[2]['legal']
    assert all(r['information_gain'] == 'unknown_without_execution' for r in candidates)
    assert not any(r['eligible'] for r in action_candidates({'views': ['front']}, [], 0))


def test_forced_fail_and_aliases_do_not_leak_variant_identity():
    history = [{'step': i, 'action': {'type': 'request_view', 'view_id': v},
                'image_ref': f'private_variant/{i}.png', 'payload': {}}
               for i, v in enumerate(('front', 'back', 'left', 'right', 'top', 'bottom'))]
    ev = [{'state': 'sufficient', 'bbox': [1, 2, 3, 4]}] * 6
    row = make_row('private_variant', 0, {'views': ['front', 'back']}, ['ref']*6, history, ev, {})
    assert row['termination_kind'] == 'forced'
    assert row['stop_outcome'] == 'fail'
    assert 'private_variant' not in row['messages'][0]['content']
    assert len(json.loads(row['messages'][0]['content'].rsplit('\n', 1)[1])['history']) == 6


def test_matching_pairs_requires_equal_actions_and_nonimage_context():
    from scripts.audit_visual_decisions import pair_rows
    def row(uid, view, state):
        history = [{'step': 0, 'action': {'type': 'request_view', 'view_id': view},
                    'image_ref': uid, 'payload': {'camera': 'same'}}]
        r = make_row(uid, 0, {'views': ['front', 'back']}, ['ref']*6, history,
                     [{'state': state}], {'split_group': 'object'})
        r['image_sha256'] = ['ref']*6 + [uid]
        return r
    a, b = row('a', 'front', 'sufficient'), row('b', 'front', 'identical')
    assert len(pair_rows([a, b])) == 1
    assert pair_rows([a, row('c', 'back', 'identical')]) == []


def test_identical_public_inputs_deduplicated_and_conflicts_rejected():
    import copy
    import pytest
    from turnitover.oracle.visual_decisions import deduplicate
    row = {'messages': [{'content': 'public'}, {'content': 'continue'}], 'image_sha256': ['abc']}
    assert len(deduplicate([row, copy.deepcopy(row)])) == 1
    conflict = copy.deepcopy(row)
    conflict['messages'][1]['content'] = 'stop'
    with pytest.raises(ValueError, match='Conflicting'):
        deduplicate([row, conflict])


def test_motion_orders_charge_resets_and_keep_reference_state_causal():
    from turnitover.oracle.motion_decisions import schedules, states, advance, candidates
    from turnitover.oracle.supervision import action_json
    refs = states('joint_0')
    for order in schedules('joint_0'):
        current = refs[0]
        history = []
        for i, action in enumerate(order):
            current = advance(current, action_json(action))
            history.append({'step': i, 'action': action_json(action), 'reference_index': refs.index(current)})
        assert len(history) == 6
        assert sum(h['action'].get('detent') == 'rest' for h in history) == 1
        assert not any(r['eligible'] for r in candidates(
            {'reference_states': refs, 'views': ['front', 'left'], 'joints': ['joint_0']}, history, 0))
    # Same candidate can require continue at rest and stop at a visible motion mismatch.
    assert decide([{'state': 'identical'}], 5)[0] == 'continue'
    assert decide([{'state': 'sufficient'}], 5)[:2] == ('stop', 'fail')
