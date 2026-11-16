"""Table 3: frozen eligible single-defect validation subset, actual browser acquisition."""
from pathlib import Path
import sys, json, hashlib, base64, time, argparse, collections, platform
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source/three-tier-900-sft-v1.1-runtime'
sys.dont_write_bytecode=True
sys.path.insert(0,str(SOURCE))
from planner import witness_grid, plan, detent_value
from turnitover.oracle.evidence import pixel_signal, pixels, ExposureThresholds
from turnitover.core.actions import RequestView, ActuateJoint
from browser import Browser

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def encode_action(a,joints):
    if isinstance(a,RequestView):return {'type':'request_view','view_id':str(a.view_id)}
    if isinstance(a,ActuateJoint):return {'type':'actuate_joint','joint_id':a.joint_id,'detent':str(a.detent),'value':detent_value(a.detent,joints[a.joint_id])}
    raise TypeError(a)
def jsload(role,case,framing,views):
    return {'url':'/prepared/'+case['programs'][role]['javascript'],'framing':framing,'width':256,'height':256,'views':views}
def browser_load(browser,request):
    return browser.evaluate('(async()=>{const r='+json.dumps(request)+'; r.source_js=await (await fetch(r.url)).text();return await window.harness.load(r);})()')
def render_case(browser,case,views,out):
    request=jsload('reference',case,'auto',views)
    info=browser_load(browser,request)
    joints={j['id']:tuple(j['limits']) for j in info['joints']}
    grid=witness_grid([v['id'] for v in views],joints)
    states=[{'key':w.key,'view_id':w.view,'joint':w.joint,'detent':str(w.detent) if w.detent is not None else None,
             'values':{**dict.fromkeys(joints,0.),**({w.joint:detent_value(w.detent,joints[w.joint])} if w.joint else {})}} for w in grid]
    records={}
    for role in ['reference','candidate']:
        request=jsload(role,case,info['framing'],views)
        role_info=browser_load(browser,request)
        assert {j['id']:tuple(j['limits']) for j in role_info['joints']}==joints
        images=browser.evaluate('(()=>{const states='+json.dumps(states)+';return states.map(s=>{window.harness.setJoints(s.values);return window.harness.requestView({view_id:s.view_id}).image_png_b64;});})()')
        records[role]=[base64.b64decode(s) for s in images]
        (out/role).mkdir(exist_ok=True)
        for index,raw in enumerate(records[role]):(out/role/f'{index:03d}.png').write_bytes(raw)
    signals=[pixel_signal(pixels(a),pixels(b)) for a,b in zip(records['candidate'],records['reference'],strict=True)]
    metadata={'case_id':case['case_id'],'framing':info['framing'],'joints':joints,'grid':states,'signals':signals,
              'image_sha256':{r:[hashlib.sha256(p).hexdigest() for p in ps] for r,ps in records.items()}}
    dump(out/'grid.json',metadata)
    return metadata

def evaluate_case(browser,case,views,out,meta):
    joints={k:tuple(v) for k,v in meta['joints'].items()}
    grid=witness_grid([v['id'] for v in views],joints)
    assert [w.key for w in grid]==[s['key'] for s in meta['grid']]
    exposures=[(0,) if s['exposed'] else () for s in meta['signals']]
    results=[];unique={}
    for budget in [1,2,4,6]:
        for mode in ['fixed','random','oracle']:
            for seed in range(100) if mode=='random' else [0]:
                path=plan(grid,exposures,joints,budget,mode=mode,seed=seed)
                assert len(path)<=budget
                actions=[dict(encode_action(a,joints),witness_index=i) for a,i in path]
                key=json.dumps(actions,sort_keys=True,separators=(',',':'))
                trace_id=hashlib.sha256(key.encode()).hexdigest()[:20]
                unique[trace_id]=actions
                first=next((t+1 for t,(_,i) in enumerate(path) if exposures[i]),budget+1)
                results.append({'case_id':case['case_id'],'object':case['data_identity']['split_group'],
                    'defect_id':case['annotation']['labels'][0]['defect_id'],'budget':budget,'policy':mode,'seed':seed,
                    'exposed':int(first<=budget),'first_visible':first,'actions_used':len(actions),'trace_id':trace_id})
    dump(out/'trajectories.json',unique)
    dump(out/'outcomes.json',results)
    # Repeated policies/seeds that select exactly the same action sequence share one replay.
    request=jsload('candidate',case,meta['framing'],views)
    browser.evaluate('window.table3Request='+json.dumps(request))
    browser.evaluate('(async()=>{window.table3Request.source_js=await (await fetch(window.table3Request.url)).text();await window.harness.load(window.table3Request);window.table3Rest='+json.dumps(dict.fromkeys(joints,0.))+';window.table3Expected=await Promise.all('+json.dumps(['/results/'+case['case_id']+'/candidate/'+f'{i:03d}.png' for i in range(len(grid))])+'.map(async u=>{const b=new Uint8Array(await (await fetch(u)).arrayBuffer());let s="";for(const v of b)s+=String.fromCharCode(v);return btoa(s);}));})()')
    traces=list(unique.items());replayed=0;steps=0
    for offset in range(0,len(traces),20):
        batch=[{'id':k,'actions':v} for k,v in traces[offset:offset+20]]
        check=browser.evaluate('(()=>{let steps=0;const errors=[];for(const t of '+json.dumps(batch)+'){window.harness.setJoints(window.table3Rest);for(let j=0;j<t.actions.length;j++){const a=t.actions[j];const r=a.type==="request_view"?window.harness.requestView({view_id:a.view_id}):window.harness.actuateJoint({joint_id:a.joint_id,value:a.value});steps++;if(r.image_png_b64!==window.table3Expected[a.witness_index])errors.push({trace_id:t.id,step:j,witness:a.witness_index});}}return {steps,errors};})()')
        if check['errors']:dump(out/'replay-errors.json',check);raise ValueError(f'Replay mismatch: {case["case_id"]} {check["errors"][:2]}')
        steps+=check['steps'];replayed+=len(batch)
    audit={'status':'complete','unique_trajectories_replayed':replayed,'replayed_actions':steps,
           'reported_runs':len(results),'all_replayed_images_equal_grid':True,
           'episode_initialization':'reset all joints before the first paid front-view action; no uncharged reset within a trajectory'}
    dump(out/'replay.json',audit)
    return audit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);args=ap.parse_args()
    cases=load(ROOT/'selection.json');views=load(ROOT/'prepared/views.json')
    assert digest(ROOT/'inputs/manifest.json')=='b64e50f027a8788554f9248b6aa7da51b4010e16103f3ffafc9e6bb41806179b'
    for case in cases:
        a=case['annotation'];assert a['split']=='validation' and a['admission']=='eligible' and len(a['labels'])==1 and not a['is_clean']
        for role in ['reference','candidate']:
            assert digest(ROOT/'prepared'/case['programs'][role]['typescript'])[:16]==a[role+'_sha']
    outroot=ROOT/'results';outroot.mkdir(exist_ok=True)
    protocol={'dataset':'three-tier-900-sft-v1.1','manifest_sha256':digest(ROOT/'inputs/manifest.json'),
      'subset':'all eligible validation cases with exactly one injected defect','cases':len(cases),
      'objects':len({c['data_identity']['split_group'] for c in cases}),
      'defects':dict(collections.Counter(c['annotation']['labels'][0]['defect_id'] for c in cases)),
      'difficulty':dict(collections.Counter(c['difficulty'] for c in cases)),
      'excluded_from_44_eligible_defective':'10 composite cases; no result-dependent selection',
      'grid':'12 configured static views; all joint unique nonzero lower/mid/upper detents from front and oblique_fr; one active joint at a time',
      'initial_action':'request_view front, costs 1; all policies share this action',
      'budgets':[1,2,4,6],'random_seeds':list(range(100)),'bootstrap_seed':20261116,'bootstrap_samples':1000,
      'cost':'every request_view and actuate_joint including restoration costs 1; initial setup not observed',
      'planner_adaptation':'When rest=0 is not lower/mid/upper, issue an explicit actuator request to 0 at cost 1. Existing targets, ordering and greedy tie breaking unchanged.',
      'resolution':[256,256],'framing':'reference-derived auto framing shared by both programs',
      'exposure':'candidate/reference pixel difference; single defect makes reference its leave-one-out counterfactual',
      'thresholds':{'image_l1':.001,'changed_fraction':.001,'pixel_delta':.02},
      'source_files':{str(p.relative_to(ROOT)):digest(p) for p in [SOURCE/'turnitover/oracle/witnesses.py',SOURCE/'turnitover/oracle/evidence.py',ROOT/'prepared/runtime/harness.js',ROOT/'prepared/runtime/three.esm.js',Path(__file__),ROOT/'browser.py',ROOT/'planner.py']},
      'platform':platform.platform(),'python':sys.version,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    dump(ROOT/'protocol.json',protocol)
    with Browser(ROOT) as browser:
        dump(ROOT/'browser-environment.json',{'gl':browser.evaluate('window.harness.gl_info()'),'user_agent':browser.evaluate('navigator.userAgent')})
        for index,case in enumerate(cases[:args.limit] if args.limit else cases):
            out=outroot/case['case_id'];out.mkdir(exist_ok=True)
            if (out/'replay.json').exists():print('RESUME',index+1,case['case_id'],flush=True);continue
            start=time.monotonic()
            meta=load(out/'grid.json') if (out/'grid.json').exists() else render_case(browser,case,views,out)
            print('GRID',index+1,case['case_id'],len(meta['grid']),'visible',sum(s['exposed'] for s in meta['signals']),flush=True)
            audit=evaluate_case(browser,case,views,out,meta)
            print('DONE',index+1,case['case_id'],audit,'seconds',round(time.monotonic()-start,2),flush=True)
    print('RUN_COMPLETE',flush=True)
if __name__=='__main__':main()
