import base64, hashlib, json, re, sys
from pathlib import Path
from turnitover.core.program import ObjectProgram
from turnitover.api_pipeline.fixture import TASK, reference_program, controlled_cases
from turnitover.api_pipeline.scoring import quality, replay_witnesses, diagnosis_metrics, TARGETS
from turnitover.render.session import RenderConfig
from turnitover.render.views import load_views
ROOT=Path.cwd(); OUT=ROOT/'.aris/traces/experiment-audit/2026-10-21_run02'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def check(name, condition, details=None):
    rows.append(dict(check=name, passed=bool(condition), details=details))
    assert condition, name
rows=[]
prepared=ROOT/'output/api-prepared-v2'; manifest=read(prepared/'manifest.json')
for path,digest in manifest['files'].items(): check('prepared_hash:'+path,sha(prepared/path)==digest)
check('reference_is_authored_program',reference_program().source==(prepared/'private/reference.ts').read_text())
check('task_is_authored_public_task',TASK==(prepared/'task.txt').read_text())
base=ObjectProgram((prepared/'shared/final.ts').read_text()); cases=controlled_cases(base)
response=read(ROOT/'output/api-prepared-v2-api/api_calls/001/response.json')
check('generated_source_from_recorded_response',base.source.strip() in response['choices'][0]['message']['content'])
render=RenderConfig(ROOT/'web/dist',ROOT/'web/node_modules/.bin/esbuild',width=256,height=256)
views=load_views(ROOT/'configs/views.yaml'); reference=reference_program(); cache={}; checks_by_arm=[]
for run_name in ('api-minimal-v6','api-paired-v6'):
    run=ROOT/'output'/run_name; result=read(run/'result.json'); scores=read(run/'scores.private.json')
    check(run_name+':prepared_manifest',sha(prepared/'manifest.json')==result['prepared_manifest_sha'])
    for path,digest in result['runtime_files'].items(): check(run_name+':runtime:'+path,sha(render.harness_dir/path)==digest)
    for case,arms in scores['pairs'].items():
        check(case+':controlled_injection',cases[case].source==(run/case/'initial.ts').read_text())
        gates=[run/case/m/'round-000/render-gate/attempt-000/render.png' for m in ('active','fixed')]
        check(case+':same_initial_gate_bytes',sha(gates[0])==sha(gates[1]))
        for mode,arm in arms.items():
            root=run/case/mode
            check(case+'/'+mode+':initial_bytes',(root/'initial.ts').read_bytes()==(run/case/'initial.ts').read_bytes())
            arm_rows=[]
            for record in arm['rounds']:
                folder=root/f"round-{record['round']:03d}/verifier"; vr=read(folder/'result.json')
                trajectory=[json.loads(line) for line in (folder/'trajectory.jsonl').read_text().splitlines()]
                check(str(folder.relative_to(ROOT))+':trajectory',trajectory==vr['trajectory'] and len(trajectory)==vr['spent']==4)
                program=ObjectProgram((folder/'candidate.ts').read_text())
                if program.sha not in cache: cache[program.sha]=quality(program,reference,render,views)
                check(case+'/'+mode+f":round{record['round']}:quality", cache[program.sha]==record['quality'])
                if record['round']==0:
                    witnesses=replay_witnesses(program,reference,trajectory,render,views)
                    diag=diagnosis_metrics(vr,TARGETS[case],witnesses)
                    check(case+'/'+mode+':witnesses',witnesses==record['witnesses'])
                    check(case+'/'+mode+':diagnosis',diag==record['diagnosis'])
                for call in vr['model_calls']:
                    d=folder/call['path']; prompt=(d/'prompt.txt').read_text()
                    data=json.JSONDecoder().raw_decode(prompt.split('\nObserved task data:\n',1)[1])[0]
                    check(str(d.relative_to(ROOT))+':history_prefix',data['history']==trajectory[:len(data['history'])])
                    check(str(d.relative_to(ROOT))+':budget',data['remaining']==4-len(data['history']))
                    check(str(d.relative_to(ROOT))+':context',data['context']==vr['context'])
                    imagepaths=read(d/'images.json')
                    expected=[r['path'] for r in vr['references']]+[o['image_ref'] for o in data['history'] if o.get('image_ref')]
                    check(str(d.relative_to(ROOT))+':image_order',imagepaths==expected)
                    check(str(d.relative_to(ROOT))+':private_tokens_absent', all(s not in prompt for s in ('private/reference.ts','invariant_pass_count','labels_valid','TARGETS','createInitialObject','setFromAxisAngle')))
                arm_rows.append({'round':record['round'],'sha':program.sha,'passed':cache[program.sha]['invariant_pass_count'],'count':cache[program.sha]['invariant_count'],'correct':cache[program.sha]['code_correct']})
            final=ObjectProgram((root/'final.ts').read_text())
            if final.sha not in cache: cache[final.sha]=quality(final,reference,render,views)
            check(case+'/'+mode+':final_quality',cache[final.sha]==arm['final_quality'])
            checks_by_arm.append({'case':case,'mode':mode,'rounds':arm_rows,'final':cache[final.sha]})
old=ROOT/'output/api-minimal-v5'; new=ROOT/'output/api-minimal-v6'
continuation=read(new/'result.json')['continuation']
check('continuation_source_result',sha(old/'result.json')==continuation['source_result_sha256'])
changed=[]
for name,digest in continuation['copied_history_sha256'].items():
    check('continuation_original:'+name,sha(old/name)==digest)
    if sha(new/name)!=digest: changed.append(name)
check('continuation_zero_new_observations', all(sha(p)==sha(new/p.relative_to(old)) for p in old.rglob('*.png')) and all(sha(p)==sha(new/p.relative_to(old)) for p in old.rglob('trajectory.jsonl')))
check('continuation_unchanged_sources',all(sha(p)==sha(new/p.relative_to(old)) for p in old.rglob('*.ts')))
oldwire=read(ROOT/'output/api-minimal-v5-api/api_calls/011/request.json'); newwire=read(ROOT/'output/api-minimal-v6-api/api_calls/001/request.json')
check('continuation_wire_identical_except_stream',{k:v for k,v in oldwire.items() if k not in ('stream','stream_options')}=={k:v for k,v in newwire.items() if k not in ('stream','stream_options')})
(OUT/'deterministic-checks.json').write_text(json.dumps({'status':'PASS','checks':rows,'arm_replay':checks_by_arm,'continuation_changed_files':changed},indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(rows),'unique_programs_revalidated':len(cache),'arms':len(checks_by_arm),'continuation_changed_files':changed},indent=2))
