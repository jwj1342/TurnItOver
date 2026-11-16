"""Incremental private per-finding measurement for completed formal attempts."""
import hashlib,sys
from pathlib import Path
from batch_common import *
from physical_scoring import score_original
root,batch=map(Path,sys.argv[1:3]);cases={c['case_id']:c for c in read(batch/'selection.private.json')}
protocol=read(batch/'physical-scoring-protocol.json');assert protocol['status']=='frozen'
groups={(r['case_id'],r['defect_id'],tuple(r['parts'])):r['group'] for r in read(batch/'initial-defect-groups.json')}
code_sha=sha(Path(__file__).with_name('physical_scoring.py'));oracle_sha=sha(Path(__file__).with_name('visible_gold_v3.py'))
out=batch/'physical-scores-v1';rows=[];controls=[];initial_equivalent={}
for uid,case in cases.items():
    ref,initial=sources(batch,case);labels=case['annotation']['labels']
    good=score_original(ref,ref,labels);bad=score_original(initial,ref,labels)
    assert all(r['resolved'] is True for r in good),(uid,'reference control')
    assert all(r['resolved'] is not None for r in bad),(uid,'initial injection control must be measurable')
    initial_equivalent[uid]={r['label_index'] for r in bad if r['resolved'] is True}
    controls.append({'case_id':uid,'reference_all_restored':True,'initial_independent_deviations':len(labels)-len(initial_equivalent[uid]),
                     'initial_equivalent_labels':sorted(initial_equivalent[uid]),'n_labels':len(labels)})
dump(out/'controls.json',controls)
for path in sorted((batch/'runs-v2').glob('*/rep-*/*/result.json')):
    try:result=read(path)
    except ValueError:continue
    if result.get('status')!='complete':continue
    folder=path.parent;uid=folder.name;case=cases[uid];dest=out/folder.relative_to(batch/'runs-v2')/'score.json'
    final=folder/'final.ts';source_sha=sha(final)
    if dest.exists():
        saved=read(dest)
        if saved.get('source_sha256')==source_sha and saved.get('code_sha256')==code_sha and saved.get('oracle_sha256')==oracle_sha:
            rows.append(saved);continue
    ref,_=sources(batch,case);source=final.read_text();labels=case['annotation']['labels']
    measured=score_original(source,ref,labels,usable=result['final_audit'].get('usable',False))
    for row in measured:
        row['exposure_group']=groups[(uid,row['defect_id'],tuple(row['parts']))]
        if row['label_index'] in initial_equivalent[uid]:
            row.update(resolved=None,unknown_reason='Original injected parameter is physically equivalent in this independent predicate; no restoration credit can be attributed.')
    saved={'condition':folder.parent.parent.name,'repetition':int(folder.parent.name.split('-')[1]),'case_id':uid,
           'source_sha256':source_sha,'code_sha256':code_sha,'oracle_sha256':oracle_sha,'defects':measured}
    dump(dest,saved);rows.append(saved)
    print('PHYSICAL_SCORE',saved['condition'],saved['repetition'],uid,[r['resolved'] for r in measured],flush=True)
dump(out/'all-scores.json',rows)
print('PHYSICAL_SCORING_COMPLETE',len(rows),flush=True)
