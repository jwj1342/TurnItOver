from pathlib import Path
import json,hashlib
from turnitover.assets.catalog import decode_spec
from turnitover.core.program import ObjectProgram

ROOT=Path.cwd()
V2=ROOT/'docs/evidence/partnet-mobility-easy-objects-v2'
V1=ROOT/'docs/evidence/partnet-mobility-obvious-defects-v1'
H1=ROOT/'docs/evidence/partnet-mobility-obvious-defects-v1-human-review'
OUT=ROOT/'docs/evidence/partnet-mobility-easy-objects-v2-audit'

def read(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def jsonl(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rs))
def norm(x): return {'joint_axis':'kinematics.joint_axis','part_offset':'structure.part_offset'}.get(x,x)
def state_key(s):return 'rest' if s=='rest' else 'joint:target:'+s

def run(bundle,review,is_v2):
 tables=[read(p) for p in [bundle/'review_manifest.jsonl',review/'human_review.jsonl',bundle/'private/gold.jsonl',bundle/'private/checks.jsonl']]
 public,human,gold,checks=[{r['case_id']:r for r in rs} for rs in tables]
 ids=set(public)
 assert all(len(rs)==len(ids) and set(t)==ids for rs,t in zip(tables,[public,human,gold,checks]))
 screens=read(bundle/'private/screening.jsonl') if is_v2 else []
 results=[]
 for cid in sorted(ids):
  p,h,g,c=public[cid],human[cid]['human_judgment'],gold[cid],checks[cid]
  kind=g['kind']; clean=kind=='clean'; aid=p['asset_id']; target=g['joint_id']
  expected_type=None if clean else ('kinematics.joint_axis' if kind=='joint_axis' else 'structure.part_offset')
  expected_joint=target if kind=='joint_axis' else None
  fields={'verdict':h['verdict']==('pass' if clean else 'fail'), 'defect_id':norm(h['defect_id'])==expected_type,
          'joint_id':h['joint_id']==expected_joint,'parts':sorted(h['parts'])==sorted(g['parts'])}
  category='uncertain' if h['verdict']=='uncertain' else 'correct' if all(fields.values()) else 'explicit_error'
  checker_ok=c['code_correct']==clean and all(x['passed'] for x in c['checks'] if x['checker_id']=='runtime.budget')
  states=h['evidence_states']; assert len(set(states))==len(states) and set(states)<=set(p['states'])
  invariants=next(x['invariants'] for x in c['checks'] if x['checker_id']=='geometry.alignment')
  failures={s:[x for x in invariants if x['state']==state_key(s) and not x['passed'] and bool(set(x['parts'])&set(g['parts']))] for s in ['rest','mid','limit']}
  screen=None
  if is_v2 and not clean:
   row=next(r for r in screens if r['asset_id']==aid and r['kind']==kind)
   screen=next(a for a in row['attempts'] if a['selected'])
   assert screen['seed']==g['seed'] and screen['automatic_pass'] and not screen['rejection_reasons']
   assert screen['checks']==c or all(screen['checks'][k]==c[k] for k in screen['checks'])
  if clean:
   evidence='explicit_all_states' if set(states)=={'rest','mid','limit'} else 'board_level_pass_no_explicit_state_citations'
   objective=True
  else:
   supporting=[s for s in states if failures[s] and (kind!='joint_axis' or s!='rest')]
   if kind=='part_offset':objective='rest' in supporting
   else:objective=bool(supporting)
   evidence='matched_reference_supports_cited_defect_states' if objective else 'insufficient_cited_states'
  holds=[]
  if is_v2 and cid=='eor-6a5cc960eafc': holds.append('normal_motion_explanation_conflicts_with_revolute_reference')
  if is_v2 and aid=='PM_8867':holds.append('human_explicitly_relies_on_multi_candidate_context_not_pair_only_validated')
  if not objective:holds.append('insufficient_evidence_states')
  qualified=category=='correct' and checker_ok and not holds
  result={'case_id':cid,'asset_id':aid,'board_row':sorted(r['case_id'] for r in public.values() if r['asset_id']==aid).index(cid)+2 if is_v2 else None,
          'kind':kind,'human_judgment':h,'gold_task_expected':{'verdict':'pass' if clean else 'fail','defect_id':expected_type,'joint_id':expected_joint,'parts':g['parts']},
          'gold_stored_joint_id':target,'literal_joint_id_match':h['joint_id']==target,
          'field_matches':fields,'human_outcome':category,'checker_consistent':checker_ok,
          'geometry_failed_states_for_target':[s for s in failures if failures[s]],'evidence_assessment':evidence,
          'objective_cited_evidence':(set(states)=={'rest','mid','limit'} if clean else objective),
          'objective_evidence_sufficient':objective,
          'selected_screening':None if screen is None else {k:screen[k] for k in ['seed','metrics','automatic_pass','rejection_reasons']},
          'semantic_hold_reasons':holds,'high_confidence_candidate':qualified,'training_eligible':False,
          'candidate_scope':'reference-plus-candidate controlled pilot only; not standalone Level1 validation'}
  if is_v2:
   spec_path=bundle/f'private/{cid}.json'; spec=json.loads(spec_path.read_text());program=ObjectProgram.from_spec(decode_spec(spec))
   assert program.sha==g['candidate_sha']==c['candidate_sha']
   clean_id=next(i for i in ids if public[i]['asset_id']==aid and gold[i]['kind']=='clean')
   ref=json.loads((bundle/f'private/{clean_id}.json').read_text())
   assert gold[clean_id]['candidate_sha']==g['reference_sha']==c['reference_sha']
   assert spec['asset_id']==aid
   if clean:
    assert spec==ref and c['code_correct']
   else:
    component,field=('joints','axis') if kind=='joint_axis' else ('parts','origin')
    assert {k:v for k,v in spec.items() if k!=component}=={k:v for k,v in ref.items() if k!=component}
    diff=[(a,b) for a,b in zip(spec[component],ref[component]) if a!=b]
    assert len(spec[component])==len(ref[component]) and len(diff)==1
    a,b=diff[0]
    assert {k:v for k,v in a.items() if k!=field}=={k:v for k,v in b.items() if k!=field}
   result['candidate_spec_sha256']=sha(spec_path)
   result['normal_joint_types']={j['id']:j['type'] for j in spec['joints']}
  results.append(result)
 by_kind={k:{'total':sum(r['kind']==k for r in results),**{v:sum(r['kind']==k and r['human_outcome']==v for r in results) for v in ['correct','uncertain','explicit_error']},
             'high_confidence_candidates':sum(r['kind']==k and r['high_confidence_candidate'] for r in results)} for k in ['clean','joint_axis','part_offset']}
 return results,by_kind

v2,counts=run(V2,V2,True);v1,baseline=run(V1,H1,False)
assert counts['clean']['correct']==8 and counts['joint_axis']['correct']==8 and counts['part_offset']['correct']==6
assert sum(r['human_outcome']=='explicit_error' for r in v2)==2
assert sum(r['human_outcome']=='uncertain' for r in v2)==0
assert {r['asset_id'] for r in v1}.isdisjoint({r['asset_id'] for r in v2})
manifest=json.loads((V2/'manifest.json').read_text())
for f,s in manifest['files'].items():assert sha(V2/f)==s,(f,'hash mismatch')
for f,s in manifest['baseline_hashes'].items():assert sha(ROOT/f)==s,(f,'baseline changed')
assets=[]
for aid in sorted({r['asset_id'] for r in v2}):
 rs=[r for r in v2 if r['asset_id']==aid]
 assets.append({'asset_id':aid,'by_kind':{r['kind']:{k:r[k] for k in ['case_id','human_outcome','high_confidence_candidate','semantic_hold_reasons','evidence_assessment']} for r in rs},
                'all_three_fields_correct':all(r['human_outcome']=='correct' for r in rs),'all_three_high_confidence':all(r['high_confidence_candidate'] for r in rs)})
screening=read(V2/'private/screening.jsonl')
summary={'version':1,'scope':'offline audit only; no teacher API calls, no training, no frozen Test modification',
 'review_independence':'same-family; see reviewer_trace/ for actual completion status',
 'by_kind':counts,'baseline_by_kind':baseline,'assets':assets,
 'totals':{'cases':len(v2),'independent_assets':len(assets),'structured_correct':sum(r['human_outcome']=='correct' for r in v2),
 'high_confidence_candidates':sum(r['high_confidence_candidate'] for r in v2),'high_confidence_assets':len({r['asset_id'] for r in v2 if r['high_confidence_candidate']}),
 'all_three_fields_correct_assets':sum(a['all_three_fields_correct'] for a in assets),'all_three_high_confidence_assets':sum(a['all_three_high_confidence'] for a in assets),
 'semantic_or_context_held_field_correct':sum(r['human_outcome']=='correct' and not r['high_confidence_candidate'] for r in v2)},
 'automatic_screening':{kind:{'targeted_attempts':sum(len(r['attempts']) for r in screening if r['kind']==kind),'automatic_pass':sum(a['automatic_pass'] for r in screening if r['kind']==kind for a in r['attempts'])} for kind in ['joint_axis','part_offset']},
 'missing_clean_state_citations':[r['case_id'] for r in v2 if r['kind']=='clean' and r['evidence_assessment']!='explicit_all_states'],
 'high_confidence_with_explicit_state_evidence':sum(r['high_confidence_candidate'] and r['objective_cited_evidence'] for r in v2),
 'literal_offset_joint_mismatches':sum(r['kind']=='part_offset' and not r['literal_joint_id_match'] for r in v2),
 'boundaries':{'model_calls':0,'model_calls_scope':'teacher experiment calls; excludes internal same-family audit reviewer',
 'training_runs':0,'training_exports':0,'test_modified':False,'input_reviews_modified':False},
 'statistics_note':'One reviewer, 8 clustered assets. Variants and attempted seeds are not independent human trials. No stability or improvement significance claim.'}
inputs=[V2/'human_review.jsonl',V2/'review_manifest.jsonl',V2/'private/gold.jsonl',V2/'private/checks.jsonl',V2/'private/screening.jsonl',V2/'manifest.json',V2/'private/object_selection.jsonl',V1/'private/gold.jsonl',V1/'private/checks.jsonl',V1/'private/screening.json',V1/'review_manifest.jsonl',H1/'human_review.jsonl',H1/'summary.json']
summary['input_sha256']={str(p.relative_to(ROOT)):sha(p) for p in inputs}
OUT.mkdir(exist_ok=True)
jsonl(OUT/'case_audit.jsonl',v2);jsonl(OUT/'baseline_case_audit.jsonl',v1)
dump(OUT/'summary.json',summary)
for name,rs in [('high_confidence_candidates',[r for r in v2 if r['high_confidence_candidate']]),('human_uncertain',[r for r in v2 if r['human_outcome']=='uncertain']),('explicit_errors',[r for r in v2 if r['human_outcome']=='explicit_error']),('semantic_or_context_holds',[r for r in v2 if r['human_outcome']=='correct' and not r['high_confidence_candidate']])]:jsonl(OUT/f'{name}.jsonl',rs)
dump(OUT/'validation.json',{'status':'passed','case_ids_aligned':24,'candidate_program_sha_verified':24,'parent_file_hashes_verified':len(manifest['files']),'baseline_file_hashes_verified':len(manifest['baseline_hashes']),'original_reviews_unchanged':sha(V2/'human_review.jsonl')==summary['input_sha256'][str((V2/'human_review.jsonl').relative_to(ROOT))]})
table=['# 逐案例核对（含私有答案，不用于盲审）', '',
 '| 资产 | 图板行 | case_id | Gold | 人审 verdict / type | joint / parts | 证据状态 | 四字段 | 后续候选 |',
 '|---|---:|---|---|---|---|---|---|---|']
for r in sorted(v2,key=lambda x:(x['asset_id'],x['board_row'])):
 h=r['human_judgment']
 table.append('| '+ ' | '.join([r['asset_id'],str(r['board_row']),r['case_id'],r['kind'],
  h['verdict']+' / '+str(h['defect_id']),str(h['joint_id'])+' / '+','.join(h['parts']),
  ','.join(h['evidence_states']) or '未逐状态引用',
  '正确' if r['human_outcome']=='correct' else r['human_outcome'],
  '高置信度候选' if r['high_confidence_candidate'] else '误判保留' if r['human_outcome']=='explicit_error' else '语义或上下文待确认'])+' |')
(OUT/'CASE_TABLE.md').write_text('\n'.join(table)+'\n')
print(json.dumps({'by_kind':counts,'baseline':baseline,'totals':summary['totals']},ensure_ascii=False,indent=2))
