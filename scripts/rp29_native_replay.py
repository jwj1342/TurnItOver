"""Subprocess bridge using RP29's native package and harness, preserving runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

def run(native_root, case, candidate, output):
    sys.path.insert(0,str(native_root))
    from turnitover.core.program import ObjectProgram
    from turnitover.core.sample import Framing
    from turnitover.render.session import ObservationSession, RenderConfig
    from turnitover.render.views import load_views
    output.mkdir(parents=True,exist_ok=False)
    config=RenderConfig(native_root/'web/dist',native_root/'web/node_modules/.bin/esbuild',width=512,height=512)
    result={'status':'incomplete','fresh':True,'pairs':[],'runtime':[],'program_sha256':{}}
    try:
        with ObservationSession(config,load_views(native_root/'configs/views.yaml')) as session:
            result['browser_version'] = session.browser_version()
            for role,path in [('reference',Path(case['reference_program'])),('candidate',candidate)]:
                source=path.read_text();result['program_sha256'][role]=hashlib.sha256(source.encode()).hexdigest()
                before=session._page.evaluate('()=>({...window.harness.renderer.info.memory})')
                session.load(ObjectProgram(source),Framing(**case['framing']))
                commands = case.get('commands_by_role', {}).get(role, case['commands'])
                if [state for state, _ in commands] != ['zero', 'mid', 'limit']:
                    raise ValueError('Expected ordered zero/mid/limit native commands')
                for index,(state,values) in enumerate(commands):
                    if index==0:
                        session.reset_joints()
                    session.set_joints(values)
                    if case.get('time_sequence') is not None:
                        tick = session.advance_time(case['time_sequence'][index])
                        result['runtime'].append({'role':role,'state':state,'kind':'public_tick_trace','value':tick})
                    image=session.request_view(case['view']);target=output/f'{role}-{state}.png';target.write_bytes(image.png)
                    stats=session._call('queryRuntime',{'property':'stats'})
                    result['runtime'].append({'role':role,'state':state,'kind':'public_stats','value':stats})
                before_dispose=session._page.evaluate('()=>({...window.harness.renderer.info.memory})')
                disposal=session.dispose_program()
                after=session._page.evaluate('()=>({...window.harness.renderer.info.memory})')
                result['runtime'].append({'role':role,'kind':'public_lifecycle_trace','value':{'action':'dispose_program','result':disposal,'renderer_memory_before_load':before,'renderer_memory_before_dispose':before_dispose,'renderer_memory_after_dispose':after}})
            result['pairs']=[{'state':state,'reference':str((output/f'reference-{state}.png').resolve()),'candidate':str((output/f'candidate-{state}.png').resolve())} for state,_ in case['commands']]
            result['status']='complete'
    finally:(output/'replay.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-root','manifest','candidate','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();case=json.loads(a.manifest.read_text())['cases'][0]
    case['reference_program']=str((a.manifest.parent/case['reference_program']).resolve())
    print(json.dumps(run(a.native_root,case,a.candidate,a.output),indent=2))
