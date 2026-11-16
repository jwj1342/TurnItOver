"""Batch Table 1 worker: one condition, same frozen cases, three repetitions."""
import argparse,base64,hashlib,json,os,socket,sys,time,traceback,uuid
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from batch_common import *
from audited_environment import RepairEnvironment
from turnitover.models.client import ModelResult
from turnitover.repair import __path__ as repair_path
from turnitover.verifier.contracts import Context,parse_decision,parse_json

def install_dns():
    original=socket.getaddrinfo;host=urlsplit(os.environ['TIO_GENERATOR_BASE_URL']).hostname;ip=os.environ['TIO_GENERATOR_RESOLVED_IP']
    socket.getaddrinfo=lambda name,*a,**kw:original(ip if name==host else name,*a,**kw)
def generator_config():
    return SimpleNamespace(validate=lambda:None,provider='openai_compatible',model=os.environ['TIO_GENERATOR_MODEL'],
          base_url=os.environ['TIO_GENERATOR_BASE_URL'],api_key=os.environ['TIO_GENERATOR_API_KEY'],request_interval=0,
          enable_thinking=None,max_tokens=8192,timeout=180,max_retries=0,stream=False,supports_vision=True)
def qwen_judge(prompt,images):
    content=[{'type':'text','text':prompt}]+[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode()}} for p in images]
    body={'model':os.environ['TIO_GENERATOR_MODEL'],'messages':[{'role':'user','content':content}],
          'max_tokens':512,'temperature':0,'enable_thinking':False,'stream':False}
    request=Request(os.environ['TIO_GENERATOR_BASE_URL'].rstrip('/')+'/chat/completions',
         data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+os.environ['TIO_GENERATOR_API_KEY'],'Content-Type':'application/json'})
    started=time.monotonic()
    try:
        with urlopen(request,timeout=180) as response:value=json.load(response)
    except HTTPError as exc:raise RuntimeError(f'Qwen judge HTTP {exc.code}') from None
    choice=value['choices'][0];usage=value.get('usage',{})
    assert (usage.get('completion_tokens_details') or {}).get('reasoning_tokens',0)==0
    text=choice['message'].get('content') or ''
    if not isinstance(text,str):raise ValueError('Unexpected judge content type')
    return ModelResult(text,os.environ['TIO_GENERATOR_MODEL'],usage,choice.get('finish_reason'),(time.monotonic()-started)*1000)

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('batch',type=Path);p.add_argument('condition',choices=['none','pixel','ordinary','zero-shot-sweep','gold']);p.add_argument('--only',nargs='*');p.add_argument('--trial-list',type=Path,required=True);a=p.parse_args()
    selected={(r['repetition'],r['case_id']) for r in read(a.trial_list)}
    protocol=read(a.batch/'protocol-v2.json')
    assert protocol['status']=='frozen' and protocol['runtime_policy']['status']=='confirmed'
    if a.condition=='gold':
        readiness=read(a.batch/'gold-preflight-v2/summary.json')
        assert len(readiness)==44 and all(r['status']=='ready' for r in readiness)
        from visible_gold_v3 import diagnose
    dataset,records,cases,cfg=setup(a.root,a.batch)
    preflight={r['case_id']:r for r in read(a.batch/protocol['preflight_summary'])}
    cases=[c for c in cases if c['case_id'] in protocol['case_ids']]
    if a.only:cases=[c for c in cases if c['case_id'] in a.only]
    repair_path.insert(0,str(a.batch/'code-v2/turnitover/repair'))
    from turnitover.repair import openhands_runner as runner
    from turnitover.repair.table1_judges import ObservationPair
    from turnitover.checkers.runtime_budget import RuntimeBudgetChecker
    install_dns();config=generator_config();progress=[]
    calibration=read(a.batch/'pixel-calibration.json')
    assert calibration['status']=='frozen' and calibration['split']=='train'
    for repetition in range(1,4):
        for case in cases:
            if (repetition,case['case_id']) not in selected:continue
            uid=case['case_id'];folder=a.batch/'runs-v2'/a.condition/f'rep-{repetition}'/uid
            if (folder/'result.json').exists():
                existing=read(folder/'result.json')
                if existing.get('status')=='complete':continue
                # Retain failed attempt records; an explicit resume policy is
                # required before replaying an incomplete trial.
                print('INCOMPLETE_PRESERVED',a.condition,repetition,uid,flush=True);continue
            folder.parent.mkdir(parents=True,exist_ok=True)
            started=time.time();request=None
            try:
                ref,initial=sources(a.batch,case);reference=ObjectProgram(ref)
                frame=preflight[uid]['framing'];framing=Framing(tuple(frame['center']),frame['radius'])
                with ObservationSession(cfg.render,load_views(cfg.views_path)) as session:
                    audit=RepairEnvironment(session,reference,framing,cfg,{})
                    if protocol['runtime_policy']['kind']=='reference_complexity':
                        runtime=RuntimeBudgetChecker(max_triangles=int(audit.gold.stats['triangles_scene']),max_draw_calls=int(audit.gold.stats['draw_calls']))
                        audit.checkers=(audit.checkers[0],runtime)
                    observe=observer(session,records[uid],case,reference,framing,ObservationPair)
                    def model_call(prompt,images):
                        if a.condition=='zero-shot-sweep':return qwen_judge(prompt,images)
                        assert a.condition=='ordinary'
                        queue=a.batch/'ordinary-queue-v2';key=uuid.uuid4().hex
                        req=queue/(key+'.request.json');response=queue/(key+'.response.json')
                        dump(req,{'request':str(request)})
                        before=time.monotonic()
                        while not response.exists():
                            if time.monotonic()-before>180:raise TimeoutError('Ordinary server did not return a public response')
                            time.sleep(.1)
                        value=read(response)
                        if value['status']!='complete':raise RuntimeError('Ordinary inference failed: '+value['error_type'])
                        dump(request.with_name('joint-response.json'),value)
                        return ModelResult(value['raw'],'Qwen3-VL-2B/ordinary-20261013',{'input_tokens':value['input_tokens'],'output_tokens':value['output_tokens']},value['finish_reason'],(time.monotonic()-before)*1000)
                    def judge(condition,pairs,*,public_record,model_call,**kwargs):
                        nonlocal request
                        if condition=='pixel':
                            scores=[]
                            for pair in pairs:
                                x=np.asarray(Image.open(pair.reference).convert('RGB'),dtype=np.float32)
                                y=np.asarray(Image.open(pair.candidate).convert('RGB'),dtype=np.float32)
                                scores.append(float(np.abs(x-y).mean()/255.))
                            changed=max(scores)>calibration['threshold']
                            return {'status':'fail' if changed else 'pass','feedback':{'summary':'Candidate differs from reference.' if changed else 'No pixel difference above the calibrated threshold.'},
                                    'observations':[{'step':i,'state':pair.state} for i,pair in enumerate(pairs)],'usage':{},'pixel_scores':scores,'threshold':calibration['threshold']}
                        if condition=='gold':
                            return kwargs['gold_call']()
                        request=public_record['request'];r=read(request)
                        value=model_call(r['record']['messages'][0]['content'],[Path(n) for n in r['image_paths']])
                        payload=public_record['payload']
                        try:
                            if value.finish_reason!='stop':raise ValueError('Incomplete diagnosis')
                            verdict=asdict(parse_decision(parse_json(value.text),Context(**payload['context']),{x['step'] for x in payload['history']},final_only=True))
                            for finding in verdict['findings']:
                                finding['evidence_steps']=sorted({public_record['steps'][s] for s in finding['evidence_steps'] if s in public_record['steps']})
                                if not finding['evidence_steps']:raise ValueError('No observed pair cited')
                        except ValueError as exc:
                            verdict={'status':'uncertain','confidence':0.,'findings':[],'summary':'The verifier did not return a supported verdict.','limitations':['Invalid or incomplete structured diagnosis.']}
                            dump(request.with_name('invalid-verdict.json'),{'error':str(exc)})
                        return {'status':verdict['status'],'feedback':verdict,'observations':[{'step':i,'state':pair.state} for i,pair in enumerate(pairs)],'usage':value.usage}
                    runner.evaluate_judge=judge
                    llm=runner.generator_llm(config).model_copy(update={'litellm_extra_body':{'enable_thinking':False}})
                    runner.run_openhands_repair(initial,config,observe,audit.audit,folder,condition=a.condition,max_rounds=3,max_agent_steps=10,
                       judge_model_call=model_call if a.condition in ('ordinary','zero-shot-sweep') else None,llm=llm,
                       gold_call=(lambda source,evidence:diagnose(source,ref,session,case,framing,evidence)) if a.condition=='gold' else None,
                       public_task='Repair the program to match the reference while preserving its part and joint identities.')
                row={'condition':a.condition,'repetition':repetition,'case_id':uid,'status':'complete','result':read(folder/'result.json')}
            except Exception as exc:
                folder.mkdir(parents=True,exist_ok=True)
                row={'condition':a.condition,'repetition':repetition,'case_id':uid,'status':'execution_error','error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
                dump(folder/'execution-error.json',row)
            row['elapsed_seconds']=time.time()-started
            progress.append(row);dump(a.batch/f'progress-{a.condition}-v3-retry.json',progress)
            print('TRIAL',a.condition,repetition,uid,row['status'],row.get('result',{}).get('final_passed'),round(row['elapsed_seconds'],1),flush=True)
    print('CONDITION_COMPLETE',a.condition,flush=True)
if __name__=='__main__':main()
