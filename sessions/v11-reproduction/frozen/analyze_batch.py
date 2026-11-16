"""Aggregate observed trajectories and independently scored original findings."""
from pathlib import Path
import collections,json
import numpy as np
ROOT=Path(__file__).resolve().parent
snapshot=json.loads((ROOT/'monitor-latest.json').read_text())
scores=json.loads((ROOT/'latest-physical-scores-v1-all-scores.json').read_text())
selection=json.loads((ROOT/'selection.private.json').read_text())
measurements=json.loads((ROOT/'measurements.json').read_text())
ids=sorted(c['case_id'] for c in selection)
groups={c['case_id']:c['data_identity']['split_group'] for c in selection}
group_ids=sorted(set(groups.values()))
report={'time':snapshot['time'],'status':'partial','conditions':{}}
report['bootstrap_group_order']='sorted source-object split_group; retain all three trajectories'
versions={(s['code_sha256'],s['oracle_sha256'],s.get('driver_sha256')) for s in scores}
report['physical_scoring_versions']=[{'predicate_sha256':v[0],'oracle_sha256':v[1],'driver_sha256':v[2]} for v in sorted(versions,key=str)]
for name in ('none','pixel','ordinary','zero-shot-sweep','gold'):
 rows=[r for r in snapshot['data'][name]['results'] if r['status']=='complete']
 keys={(r['case_id'],int(r['repetition'].split('-')[1])) for r in rows}
 details=[m for m in measurements if m['condition']==name and (m['case_id'],m['repetition']) in keys]
 source_hashes={(m['case_id'],m['repetition']):m['source_sha256'] for m in details}
 measured=[s for s in scores if s['condition']==name and (s['case_id'],s['repetition']) in keys and source_hashes.get((s['case_id'],s['repetition']))==s['source_sha256']]
 entry={'n':len(rows),'success':sum(r['final_passed'] is True for r in rows),
        'mean_completed_rounds':float(np.mean([r['completed_rounds'] for r in rows])) if rows else None,
        'mean_generator_calls':float(np.mean([r['generator_calls'] for r in rows])) if rows else None,
        'total_generator_tokens':sum((r['total_tokens'] or 0)-(r['judge_tokens'] or 0) for r in rows),
        'total_pipeline_tokens':sum(r['total_tokens'] or 0 for r in rows),
        'zero_submission_trajectories':sum(r['completed_rounds']==0 for r in rows),
        'total_verifier_tokens':sum(r['judge_tokens'] or 0 for r in rows),
        'usage_complete':all(r['usage_complete'] for r in rows),
        'terminations':dict(collections.Counter(r['termination'] for r in rows)),
        'physical_scored_trajectories':len(measured),'physical':{}}
 successes=[m['result'] for m in details if m['result']['final_passed'] is True]
 entry['cost_to_first_success_among_final_successes']={
     'n':len(successes),
     'mean_iterations':float(np.mean([r['first_success_round'] for r in successes])) if successes else None,
     'mean_pipeline_tokens':float(np.mean([r['tokens_to_first_success'] for r in successes])) if successes and all(r.get('tokens_to_first_success') is not None for r in successes) else None,
     'interpretation':'Conditional on final success; failures are budget-censored and not assigned a successful cost'}
 rounds=[r for m in details for r in m['rounds']]
 entry['public_pass_but_private_fail_versions']=sum(r['verdict']=='pass' and r['passed'] is False for r in rounds)
 entry['repaired_then_regressed_trajectories']=sum(m['result']['first_success_round'] is not None and m['result']['final_passed'] is False for m in details)
 for group in ('rest','motion','unexposed','all'):
  findings=[f for s in measured for f in s['defects'] if group=='all' or f['exposure_group']==group]
  n=len(findings);yes=sum(f['resolved'] is True for f in findings);no=sum(f['resolved'] is False for f in findings)
  unknown=n-yes-no
  no_credit=sum(f.get('credit_eligible') is False or f.get('unknown_reason','').startswith('Original injected parameter is physically equivalent') for f in findings)
  entry['physical'][group]={'n_original_items':n,'resolved':yes,'unresolved':no,'unknown':unknown,
      'initially_equivalent_no_repair_credit':no_credit,
      'assessable_coverage':(yes+no)/n if n else None,
      'resolved_fraction_bounds':[yes/n,(yes+unknown-no_credit)/n] if n else None,
      'conditional_assessable_rate':yes/(yes+no) if yes+no else None,
      'unknown_reasons':dict(collections.Counter(f.get('unknown_reason') for f in findings if f['resolved'] is None))}
 if len(keys)==132:
  per_object=np.array([np.mean([r['final_passed'] for r in rows if groups[r['case_id']]==group]) for group in group_ids])
  rng=np.random.default_rng(20261116)
  samples=rng.integers(0,44,(1000,44))
  entry['success_rate']=entry['success']/132
  entry['source_object_bootstrap_ci95']=np.quantile(per_object[samples].mean(axis=1),[.025,.975]).tolist()
 report['conditions'][name]=entry
if len(versions)==1 and all(e['n']==132 and e['physical_scored_trajectories']==132 for e in report['conditions'].values()):report['status']='complete'
(ROOT/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
for name,e in report['conditions'].items():
 print(name,e['n'],e['success'],'physical',e['physical_scored_trajectories'],{g:(v['resolved'],v['unresolved'],v['unknown']) for g,v in e['physical'].items()})
