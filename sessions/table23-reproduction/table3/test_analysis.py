"""Check the restored-state cost and source-object aggregation, independently of a browser."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source/three-tier-900-sft-v1.1-runtime'))
from planner import Witness,recipe,next_witness,detent_value,plan,witness_grid
from turnitover.core.actions import Detent,ActuateJoint,RequestView
from summarize import estimate

joints={'a':(-1.,2.),'b':(0.,1.)}
current=Witness('front','a',Detent.LIMIT)
target=Witness('oblique_fr','b',Detent.MID)
actions=recipe(current,target,joints)
assert len(actions)==3,actions
assert isinstance(actions[0],ActuateJoint) and detent_value(actions[0].detent,joints['a'])==0
for action in actions:current=next_witness(current,action,joints)
assert current==target
grid=witness_grid(['front','back','oblique_fr'],joints)
exposures=[(0,) if w==target else () for w in grid]
for mode in ['fixed','random','oracle']:
 for k in [1,2,4,6]:
  for seed in range(100) if mode=='random' else [0]:
   path=plan(grid,exposures,joints,k,mode,seed)
   assert 1<=len(path)<=k
   assert isinstance(path[0][0],RequestView)

# One source object owns two cases: both must enter the bootstrap together.
result=estimate([0.,1.,1.],{0:[0,1],1:[2]},np.array([[0,0],[1,1],[0,1]]))
assert np.isclose(result['value'],2/3)
assert np.allclose(result['ci95'],np.quantile([.5,1.,2/3],[.025,.975]))
print('PASS: explicit paid restoration, legal paths at all budgets, object-level bootstrap grouping')
