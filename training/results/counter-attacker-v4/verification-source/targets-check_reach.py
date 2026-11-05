from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np
from offense_eval import contact_paths
P=np.arange(15,146)[:,None];Q=np.arange(15,146)[None,:]
paths=contact_paths();count=0
for ticks in range(1,67):
 normal=(Q-P)%2==0
 normal&=abs(Q-P)<=2*ticks
 odd=(((P%2)==0)&((Q%2)==1)&((np.ceil((P-15)/2)+(Q-15)/2<=ticks)|(np.ceil((145-P)/2)+(145-Q)/2<=ticks)))
 reachable=normal|odd
 expected=paths[ticks,:,:,3]>=0
 np.testing.assert_array_equal(reachable,expected)
 count+=reachable.size
bands=0
for speed in range(6,14):
 for height in range(4,157):
  physical={}
  for d in range(-15,16):
   center=height-d
   if not 15<=center<=145:continue
   vy=int(np.floor(3*speed*d/65+.5));physical.setdefault(vy,[]).append(center)
  for vy in range(-9,10):
   lo=int(np.ceil(max(15,height-15,height-(vy+.5)*65/(3*speed))-1e-7))
   hi=int(np.floor(min(145,height+15,height-(vy-.5)*65/(3*speed))+1e-7))
   found=list(range(lo,hi+1)) if lo<=hi else []
   np.testing.assert_array_equal(found,sorted(physical.get(vy,[])))
   bands+=1
record=dict(purpose='Exhaustive discrete reach and19vyband grouping verification; no fitting',reach_tuples=count,vy_bands=bands,mismatches=0,formula='sameparity |q−p|≤2t; evenp→oddq via15or145 with ceil travel; odd never returns even')
Path('/tmp/rally-attacker-targets/reach-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(record)
