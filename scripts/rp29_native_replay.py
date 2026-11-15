"""Subprocess bridge using RP29's native package and harness, preserving runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def state_commands(case, role, info):
    """Resolve an authored sweep; repaired range cases use CURRENT joint limits.

    Numeric commands remain the default for previously frozen recipes. The
    explicit detent policy is recorded by input preparation for range cases
    whose original two arms used different numeric limits.
    """
    commands = case.get('commands_by_role', {}).get(role, case['commands'])
    if [state for state, _ in commands] != ['zero', 'mid', 'limit']:
        raise ValueError('Expected ordered zero/mid/limit native commands')
    policy = case.get('replay_command_policy', 'authored_numeric')
    if policy not in {'authored_numeric', 'current_joint_limits'}:
        raise ValueError('Unknown native command policy')
    if policy == 'current_joint_limits':
        resolved = []
        for index, (state, values) in enumerate(commands):
            current = {}
            for joint in values:
                if joint not in info.joints:
                    continue  # Missing joints are scored by the private ABI check.
                lower, upper = info.joints[joint]['limits']
                current[joint] = (lower, (lower + upper) / 2, upper)[index]
            resolved.append((state, current))
        return resolved
    # A missing joint must not turn an otherwise renderable edit into an
    # infrastructure exception before the independent contract check can run.
    return [(state, {j: v for j, v in values.items() if j in info.joints})
            for state, values in commands]


def save_private_evidence(output, role, value):
    path = output / f'{role}-evidence.private.json'
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
    path.write_bytes(data)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest()}


def replay_authored_history(session, program, framing, case, role, readback):
    """Re-execute frozen source actions through the existing session API.

    This is private regression evidence, never extra feedback to any condition.
    Unknown action schemas stay explicitly incomplete instead of being skipped.
    """
    history = case.get('authored_audit_history')
    if not history:
        return None
    before = session._page.evaluate('()=>({...window.harness.renderer.info.memory})')
    session.load(program, framing)
    session._page.evaluate(readback, {'mode': 'start'})
    commands = dict(state_commands(case, role, session.info))
    aliases = {'rest': 'zero', 'midpoint': 'mid', 'endpoint': 'limit'}
    snapshots, complete, current_state, current_time = [], True, None, None
    for observation in history:
        action = observation['action']
        kind = action['type']
        # Paired observation exports can encode a pose on request_view rather
        # than retaining the preceding setup actuator as a reviewer action.
        state = aliases.get(observation.get('state'), observation.get('state'))
        if state in commands and state != current_state and kind == 'request_view':
            if case.get('time_sequence') is None:
                session.set_joints(commands[state])
            else:
                time = case['time_sequence'][('zero', 'mid', 'limit').index(state)]
                if current_time is None or time > current_time:
                    session.advance_time(time)
                    current_time = time
                elif time < current_time:
                    complete = False
                    break
            current_state = state
        if kind == 'request_view':
            session.request_view(action['view_id'])
        elif kind == 'query_runtime':
            session._call('queryRuntime', {'property': action['property']})
        elif kind == 'actuate_joint':
            joint = action['joint_id']
            if joint in session.info.joints:
                if 'detent' in action:
                    lower, upper = session.info.joints[joint]['limits']
                    value = {'zero': lower, 'mid': (lower + upper) / 2, 'limit': upper}[action['detent']]
                elif 'value' in action:
                    value = action['value']
                else:
                    complete = False
                    break
                session.actuate_joint(joint, value)
        elif kind == 'advance_time':
            session.advance_time(action['time_seconds'])
            current_time = action['time_seconds']
        else:
            complete = False
            break
        snapshots.append({'action': action, 'geometry': session._call('exportGeometry'),
            'readback': session._page.evaluate(readback, {'mode': 'snapshot'})})
    allocated = session._page.evaluate('()=>({...window.harness.renderer.info.memory})')
    response = session.dispose_program()
    after = session._page.evaluate('()=>({...window.harness.renderer.info.memory})')
    disposed = session._page.evaluate(readback, {'mode': 'disposed'})
    return {'complete': complete and len(snapshots) == len(history), 'snapshots': snapshots,
            'lifecycle': {**disposed, 'result': response,
                          'before_load': before, 'before_dispose': allocated, 'after_dispose': after}}


def replay_role(session, program, framing, case, role, output, result):
    """Read the existing native harness; all private exports stay in a sidecar."""
    before = session._page.evaluate('()=>({...window.harness.renderer.info.memory})')
    session.load(program, framing)
    info = session.info
    readback = Path(__file__).with_name('rp29_private_readback.js').read_text(encoding='utf-8')
    initial = session._page.evaluate(readback, {'mode': 'start'})
    private = {'schema_version': 1, 'role': role,
        'program_sha256': result['program_sha256'][role],
        'contract': {'parts': sorted(info.parts),
                     'joints': {key: {name: value[name] for name in ('type', 'part', 'limits')}
                                for key, value in info.joints.items()}},
        'states': {}, 'initial': initial}
    commands = state_commands(case, role, info)
    for index, (state, values) in enumerate(commands):
        if index == 0:
            session.reset_joints()
        session.set_joints(values)
        if case.get('time_sequence') is not None:
            tick = session.advance_time(case['time_sequence'][index])
            result['runtime'].append({'role': role, 'state': state,
                                     'kind': 'public_tick_trace', 'value': tick})
        image = session.request_view(case['view'])
        (output / f'{role}-{state}.png').write_bytes(image.png)
        stats = session._call('queryRuntime', {'property': 'stats'})
        result['runtime'].append({'role': role, 'state': state, 'kind': 'public_stats', 'value': stats})
        private['states'][state] = {'commands': values, 'geometry': session._call('exportGeometry'),
            'readback': session._page.evaluate(readback, {'mode': 'snapshot'})}
    # Construction includes defects triggered only by returning to the original
    # pose. This private regression probe never enters public judge observations.
    # Temporal programs must not be reset via joint setters between timestamps.
    if case.get('time_sequence') is None:
        session.set_joints(commands[0][1])
        session.request_view(case['view'])
        private['return_to_zero'] = {'commands': commands[0][1],
            'geometry': session._call('exportGeometry'),
            'stats': session._call('queryRuntime', {'property': 'stats'}),
            'readback': session._page.evaluate(readback, {'mode': 'snapshot'})}
    before_dispose = session._page.evaluate('()=>({...window.harness.renderer.info.memory})')
    disposal = session.dispose_program()
    after = session._page.evaluate('()=>({...window.harness.renderer.info.memory})')
    private['lifecycle'] = {**session._page.evaluate(readback, {'mode': 'disposed'}),
        'result': disposal, 'before_load': before, 'before_dispose': before_dispose, 'after_dispose': after}
    result['runtime'].append({'role': role, 'kind': 'public_lifecycle_trace', 'value': {
        'action': 'dispose_program', 'result': disposal, 'renderer_memory_before_load': before,
        'renderer_memory_before_dispose': before_dispose, 'renderer_memory_after_dispose': after}})
    # Native tick is absolute time. Replay the same final timestamp with a
    # shorter history, as in balanced_runtime_tick_history.py. Use ONLY the
    # already-authored times; no invented animation timing or phase threshold.
    times = case.get('time_sequence')
    if times is not None:
        if len(times) != 3 or times[0] < 0 or not times[0] < times[1] < times[2]:
            raise ValueError('Expected three strictly increasing authored timestamps')
        private['time_sequence'] = times
        direct_before = session._page.evaluate('()=>({...window.harness.renderer.info.memory})')
        session.load(program, framing)
        session._page.evaluate(readback, {'mode': 'start'})
        for time in (times[0], times[-1]):
            session.advance_time(time)
        session.request_view(case['view'])
        private['direct_time_endpoint'] = {'time_seconds': times[-1],
            'geometry': session._call('exportGeometry'),
            'readback': session._page.evaluate(readback, {'mode': 'snapshot'})}
        direct_allocated = session._page.evaluate('()=>({...window.harness.renderer.info.memory})')
        direct_disposal = session.dispose_program()
        direct_after = session._page.evaluate('()=>({...window.harness.renderer.info.memory})')
        private['direct_time_lifecycle'] = {**session._page.evaluate(readback, {'mode': 'disposed'}),
            'result': direct_disposal, 'before_load': direct_before,
            'before_dispose': direct_allocated, 'after_dispose': direct_after}
    authored = replay_authored_history(session, program, framing, case, role, readback)
    if authored is not None:
        private['authored_history'] = authored
    result['private_evidence'][role] = save_private_evidence(output, role, private)


def run(native_root, case, candidate, output):
    sys.path.insert(0,str(native_root))
    from turnitover.core.program import ObjectProgram
    from turnitover.core.sample import Framing
    from turnitover.render.session import ObservationSession, RenderConfig, HarnessError
    from turnitover.render.transpile import CompileError
    from turnitover.render.views import load_views
    output.mkdir(parents=True,exist_ok=False)
    config=RenderConfig(native_root/'web/dist',native_root/'web/node_modules/.bin/esbuild',width=512,height=512)
    result={'status':'incomplete','fresh':True,'pairs':[],'runtime':[],'program_sha256':{},
            'private_evidence': {}}
    try:
        with ObservationSession(config,load_views(native_root/'configs/views.yaml')) as session:
            result['browser_version'] = session.browser_version()
            for role,path in [('reference',Path(case['reference_program'])),('candidate',candidate)]:
                source=path.read_text(encoding='utf-8');result['program_sha256'][role]=hashlib.sha256(source.encode()).hexdigest()
                try:
                    replay_role(session, ObjectProgram(source), Framing(**case['framing']),
                                case, role, output, result)
                except (HarnessError, CompileError) as exc:
                    # Reference/transport failures are infrastructure failures. A
                    # deterministic candidate rejection is public gate feedback.
                    stage = 'compile' if isinstance(exc, CompileError) else exc.stage
                    if role != 'candidate' or not stage:
                        raise
                    result.update(status='complete', gate={'passed':False,
                        'stage':stage, 'summary':'Candidate failed compile/render.'})
                    if isinstance(exc, CompileError):
                        diagnostic = str(exc)
                        paths = sorted(map(str, (case['reference_program'], native_root, output)),
                                       key=len, reverse=True)
                        for path in paths:
                            diagnostic = diagnostic.replace(path, '[private-path]')
                            diagnostic = diagnostic.replace(path.replace('\\', '/'), '[private-path]')
                        result['gate']['diagnostic'] = diagnostic[:3000]
                    result['gate_error_private'] = str(exc)
                    return result
            result['pairs']=[{'state':state,'reference':str((output/f'reference-{state}.png').resolve()),'candidate':str((output/f'candidate-{state}.png').resolve())} for state,_ in case['commands']]
            result['status']='complete'
            result['gate']={'passed':True}
    finally:(output/'replay.json').write_text(json.dumps(result,indent=2), encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-root','manifest','candidate','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();case=json.loads(a.manifest.read_text())['cases'][0]
    case['reference_program']=str((a.manifest.parent/case['reference_program']).resolve())
    print(json.dumps(run(a.native_root,case,a.candidate,a.output),indent=2))
