"""Original input audit, adapted to local paths; missing image events now fail closed."""

import json,pathlib,collections
import argparse
p=argparse.ArgumentParser(description='Audit local recorded model inputs; does not call models')
p.add_argument('--work',type=pathlib.Path,required=True)
p.add_argument('--out',type=pathlib.Path,required=True)
a=p.parse_args()
if a.out.exists():raise ValueError('Output must not exist')
b=a.work;rows=[]
for condition in ['none','pixel','ordinary','zero-shot-sweep','gold']:
 for path in sorted((b/'runs-v2'/condition).glob('rep-*/*/result.json')):
  result=json.loads(path.read_text())
  if result.get('status')!='complete':continue
  folder=path.parent;events=[]
  for p in (folder/'conversation').rglob('*.json'):
   try:events.append(json.loads(p.read_text()))
   except Exception:pass
  counts=[];thought=[];source_lengths=[];leaks=[];none_diagnoses=[]
  def public_check(v):
   if isinstance(v,dict):
    for k,val in v.items():
     if k in ('final_audit','runtime_policy','private_parameter_updates','label_policy','is_clean','admission','annotation'):leaks.append(k)
     if condition=='none' and k=='findings' and val:none_diagnoses.append(k)
     public_check(val)
   elif isinstance(v,list):
    for x in v:public_check(x)
  def walk(v):
   if isinstance(v,dict):
    if 'reasoning_tokens' in v:thought.append(v['reasoning_tokens'])
    for k,val in v.items():
     if k=='content' and isinstance(val,list):
      n=sum(len(x.get('image_urls',[])) if isinstance(x,dict) else 0 for x in val)
      if n:counts.append(n)
     if k=='text' and isinstance(val,str):
      try:
       parsed=json.loads(val)
       if isinstance(parsed,dict) and 'current_program' in parsed:source_lengths.append(len(val))
       public_check(parsed)
      except (ValueError,TypeError):pass
     walk(val)
   elif isinstance(v,list):
    for x in v:walk(x)
  for e in events:walk(e)
  manifest=json.loads((folder/'observation-manifest.json').read_text())
  expected=len(manifest['generator_images'])
  row={'condition':condition,'case_id':folder.name,'rep':folder.parent.name,'expected_images':expected,'event_image_counts':sorted(set(counts)),
    'reasoning_tokens_values':sorted(set(thought)),'completed_rounds':result['completed_rounds'],'generator_calls':result['generator_calls'],'within_budget':result['completed_rounds']<=3 and result['generator_calls']<=10,
    'max_serialized_source_view_chars':max(source_lengths,default=0),'private_field_leaks':sorted(set(leaks)),'none_diagnosis_leaks':none_diagnoses}
  rows.append(row)
summary={'audited':len(rows),'image_counts':sorted({r['expected_images'] for r in rows}),
 'max_serialized_source_view_chars':max((r['max_serialized_source_view_chars'] for r in rows),default=0),
 'anomalies':[r for r in rows if not r['within_budget'] or r['private_field_leaks'] or r['none_diagnosis_leaks'] or not r['event_image_counts'] or any(n!=r['expected_images'] for n in r['event_image_counts']) or any(r['reasoning_tokens_values'])]}
a.out.mkdir(parents=True);(a.out/'.gitignore').write_text('*\n')
(a.out/'input-audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
(a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'audited':summary['audited'],'anomalies':len(summary['anomalies'])}))
if summary['anomalies']:raise SystemExit(1)
