"""Independent pixel/hash and trajectory recount before reporting results."""
from pathlib import Path
import sys,json,hashlib,collections
import numpy as np
from PIL import Image
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source/three-tier-900-sft-v1.1-runtime'))
from planner import witness_grid,plan
from run_table3 import encode_action
def load(p):return json.loads(p.read_text(encoding='utf-8'))
cases=load(root/'selection.json');views=load(root/'prepared/views.json');summary=load(root/'summary.json')
image_count=0;trajectory_count=0;totals=collections.defaultdict(list)
for case in cases:
 d=root/'results'/case['case_id'];grid=load(d/'grid.json');outcomes=load(d/'outcomes.json');traces=load(d/'trajectories.json')
 visible=[]
 for i,signal in enumerate(grid['signals']):
  frames=[]
  for role in ['reference','candidate']:
   p=d/role/f'{i:03d}.png'
   assert hashlib.sha256(p.read_bytes()).hexdigest()==grid['image_sha256'][role][i]
   frames.append(np.asarray(Image.open(p).convert('RGB'),dtype=np.int16));image_count+=1
  diff=np.abs(frames[0]-frames[1])
  exposed=bool(diff.mean()>=.001*255 and (diff.max(axis=2)>.02*255).mean()>=.001)
  assert exposed==signal['exposed'];visible.append((0,) if exposed else ())
 joints={k:tuple(v) for k,v in grid['joints'].items()}
 witnesses=witness_grid([v['id'] for v in views],joints)
 for row in outcomes:
  path=plan(witnesses,visible,joints,row['budget'],row['policy'],row['seed'])
  expected=[dict(encode_action(a,joints),witness_index=i) for a,i in path]
  assert expected==traces[row['trace_id']]
  first=next((j+1 for j,(_,i) in enumerate(path) if visible[i]),row['budget']+1)
  assert first==row['first_visible']
  totals[row['budget'],row['policy']].append(int(first<=row['budget']));trajectory_count+=1
for row in summary['rows']:
 assert np.isclose(np.mean(totals[row['budget'],row['policy']]),row['all_recall']['value'])
assert trajectory_count==34*408
result={'status':'passed','images_independently_recomputed':image_count,'policy_runs_independently_recomputed':trajectory_count,
        'all_program_image_hashes_and_exposure_decisions_match':True,'all_current_planner_paths_match_saved_paths':True,
        'all_table_overall_recalls_match':True}
(root/'final-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
