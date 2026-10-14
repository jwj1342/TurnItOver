"""Public-state motion view schedules and ex ante observation eligibility."""
from turnitover.core.actions import RequestView, ActuateJoint, Detent

MOTION_INSTRUCTION = '''Compare the acquired candidate poses against the supplied reference poses.
The payload reference_states lists each reference image's camera view and joint
state. All unspecified joints are at rest. References and observations use the
same framing and lighting. Each history entry identifies its comparable reference
image. Stop/fail only for a visible shape mismatch at a matching state; otherwise
continue acquiring unseen states while budget remains. At budget exhaustion with
no established mismatch stop/uncertain, never infer pass. Do not infer defects at
unobserved states. Return only a stop_decision JSON object.'''


def schedules(joint, detent="limit"):
    front, left, top, back = [RequestView(v) for v in ('front', 'left', 'top', 'back')]
    move, rest = ActuateJoint(joint, Detent(detent)), ActuateJoint(joint, Detent.REST)
    return ((front, move, left, rest, top, back),
            (move, left, rest, front, top, back),
            (left, move, front, rest, top, back))


def states(joint, detent="limit"):
    return ([{'view': v, 'joint': joint, 'detent': 'rest'}
             for v in ('front', 'back', 'left', 'right', 'top', 'bottom')]
            + [{'view': v, 'joint': joint, 'detent': detent} for v in ('front', 'left')])


def advance(state, action):
    state = dict(state)
    if action['type'] == 'request_view':
        state['view'] = action['view_id']
    elif action['type'] == 'actuate_joint' and action['joint_id'] == state['joint']:
        state['detent'] = action['detent']
    else:
        raise ValueError('Action outside public motion subspace')
    return state


def candidates(context, history, remaining):
    references = context['reference_states']
    joint = references[0]['joint']
    current = references[history[-1]['reference_index']] if history else references[0]
    seen = {h['reference_index'] for h in history}
    actions = ([{'type': 'request_view', 'view_id': v} for v in context['views']]
               + [{'type': 'actuate_joint', 'joint_id': joint, 'detent': d} for d in ('rest', references[-1]['detent'])])
    result = []
    for action in actions:
        state = advance(current, action)
        index = references.index(state) if state in references else None
        legal = remaining > 0 and (action['type'] == 'request_view' or joint in context['joints'])
        result.append({'action': action, 'legal': legal, 'reference_available': index is not None,
                       'unseen_state_gain': int(index is not None and index not in seen),
                       'information_gain': 'unknown_without_execution',
                       'eligible': legal and index is not None and index not in seen})
    return result
