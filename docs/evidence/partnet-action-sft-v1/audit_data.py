import hashlib,json,math
from pathlib import Path
from collections import defaultdict,Counter
from turnitover.oracle.ambiguity import observable_key
from turnitover.verifier.contracts import Context,parse_decision
src=Path('../TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4').resolve()
out=Path('data/partnet-action-distributions-v1')
evidence=Path('docs/evidence/partnet-action-sft-v1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'source':str(src),'output':str(out.resolve()),'frozen_test_read':False,'source_files_sha256':{},'splits':{}}
catalog_path=src.parents[2]/'assets/partnet-mobility-trainval-v1/catalog.json'
catalog=json.loads(catalog_path.read_text());assert catalog['formal_training'] is True and catalog['role']=='formal_train_validation'
entries={e['spec'].removesuffix('.json'):e for e in catalog['assets']}
report['source_files_sha256'][str(catalog_path)]=sha(catalog_path)
images={}; identities={}; keys={}; ref_pixels={}; mismatches=[]
for split in ('train','validation'):
 path=src/f'{split}.actions.jsonl';report['source_files_sha256'][str(path)]=sha(path)
 rows=[json.loads(x) for x in path.read_text().splitlines()]
 groups=defaultdict(list); assets=set(); ids=set(); source_ids=set(); content=set(); refs=set(); pixels=set(); steps=Counter();actions=Counter(); episode_rows=Counter()
 for row in rows:
  ident=row['data_identity'];entry=entries[ident['asset_id']];spec_path=catalog_path.parent/entry['spec']
  assert sha(spec_path)==entry['spec_sha256']==ident['spec_sha256']
  assert entry['archive_sha256']==ident['source_content_sha256'] and entry['source_identity']==ident['source_identity']
  assert ident['formal_training'] is True and ident['catalog_role']=='formal_train_validation'
  report['source_files_sha256'][str(spec_path)]=entry['spec_sha256']
  assets.add(ident['asset_id']);ids.add(ident['source_identity_sha256']);source_ids.add(ident['source_identity']);content.add(ident['source_content_sha256']);refs.add(ident['reference_sha'])
  uid,_,step=row['id'].split(':');episode_rows[uid]+=1
  payload=json.loads(row['messages'][0]['content'].rsplit('\n',1)[1]);target=json.loads(row['messages'][1]['content'])
  assert set(payload)=={'context','remaining','history','image_order','taxonomy_ids'}
  assert payload['remaining']==6-int(step) and len(payload['history'])==int(step)
  assert payload['image_order']==row['images'] and row['images']==[row['images'][0]]+[o['image_ref'] for o in payload['history']]
  assert all(o['step']<int(step) for o in payload['history'])
  assert all('render_ms' not in o['payload'] for o in payload['history'])
  parse_decision(target,Context(**payload['context']),{o['step'] for o in payload['history']})
  assert len(row['images'])==len(row['image_sha256'])
  for name,expected in zip(row['images'],row['image_sha256']):
   actual=images.setdefault(name,sha(src/name));assert actual==expected
  pixels.add(images[row['images'][0]])
  groups[observable_key(row,src,images)].append(row)
  steps[int(step)]+=1;actions[json.dumps(target,sort_keys=True)]+=1
 conflicts=[];total_entropy=0;upper=0;initial=[]
 for key,rs in groups.items():
  c=Counter(r['messages'][1]['content'] for r in rs);entropy=-sum((v/len(rs))*math.log(v/len(rs)) for v in c.values());total_entropy+=entropy;upper+=max(c.values())
  if len(c)>1:conflicts.append({'group':key,'source_rows':len(rs),'distinct_actions':len(c),'counts':dict(c),'history_length':len(json.loads(rs[0]['messages'][0]['content'].rsplit('\n',1)[1])['history'])})
  if rs[0]['id'].endswith(':0'):initial.append(c)
 identities[split]={'asset_id':assets,'source_identity_sha256':ids,'source_identity':source_ids,'source_content_sha256':content,'reference_sha':refs};keys[split]=set(groups);ref_pixels[split]=pixels
 report['splits'][split]={'source_action_rows':len(rows),'independent_objects':len(assets),'asset_ids':sorted(assets),'episodes':len(episode_rows),'rows_per_episode':dict(Counter(episode_rows.values())),'steps':dict(steps),'action_counts':dict(actions),'observable_groups':len(groups),'weighted_variants':sum(len(set(r['messages'][1]['content'] for r in rs)) for rs in groups.values()),'conflicts':conflicts,'conflicting_groups':len(conflicts),'affected_source_rows':sum(c['source_rows'] for c in conflicts),'empirical_group_entropy_lower_bound_nats':total_entropy/len(groups),'source_row_deterministic_top1_ceiling':upper/len(rows),'initial_action_counts':dict(sum(initial,Counter()))}
report['split_overlap']={k:sorted(identities['train'][k]&identities['validation'][k]) for k in identities['train']}
report['split_overlap']['observable_key']=sorted(keys['train']&keys['validation']);report['split_overlap']['reference_image_pixels']=sorted(ref_pixels['train']&ref_pixels['validation']);assert not any(report['split_overlap'].values())
report['image_integrity']={'referenced_paths':len(images),'unique_sha256':len(set(images.values())),'all_declared_hashes_verified':True,'paths_sha256':images}
report['source_files_sha256'][str(src/'manifest.json')]=sha(src/'manifest.json')
annotations=[json.loads(x) for x in (src/'annotations.private.jsonl').read_text().splitlines()];assert {a['split'] for a in annotations}<={'train','validation'}
report['source_files_sha256'][str(src/'annotations.private.jsonl')]=sha(src/'annotations.private.jsonl')
report['offline_budget_audit']={'scope':'Only prefixes of existing K=6 traces; no new teacher, no diagnosis, no policy inference','validation':{}}
for k in (1,2,4,6):
 modes={}
 for mode in ('oracle','fixed','random'):
  results=[]
  for a in annotations:
   if a['split']!='validation':continue
   tr=a['trajectories'][mode];idx=tr['witness_indices'][:k];covered=set().union(*(set(a['exposures'][i]) for i in idx));expected=len(a['labels'])
   results.append({'id':a['id'],'asset_id':a['asset_id'],'labels':expected,'valid_evidence':bool(covered),'exposed_labels':len(covered),'all_labels_exposed':expected>0 and len(covered)==expected,'spent':len(idx)})
  defective=[r for r in results if r['labels']]
  modes[mode]={'all_cases':len(results),'defective_cases':len(defective),'valid_evidence_rate_all_cases':sum(r['valid_evidence'] for r in results)/len(results),'valid_evidence_rate_defective_cases':sum(r['valid_evidence'] for r in defective)/len(defective),'all_labels_exposed_rate_defective_cases':sum(r['all_labels_exposed'] for r in defective)/len(defective),'per_case':results}
 report['offline_budget_audit']['validation'][str(k)]=modes
report['export_manifest_sha256']=sha(out/'manifest.json');report['export_files_sha256']={name:sha(out/name) for name in ('train.actions.jsonl','validation.actions.jsonl','groups.private.jsonl')}
(evidence/'data-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{x:v[x] for x in ('independent_objects','episodes','observable_groups','weighted_variants','conflicting_groups','affected_source_rows','source_row_deterministic_top1_ceiling','empirical_group_entropy_lower_bound_nats')} for k,v in report['splits'].items()},indent=2))
for k,v in report['offline_budget_audit']['validation'].items(): print(k,{m:{x:y for x,y in r.items() if x!='per_case'} for m,r in v.items()})
