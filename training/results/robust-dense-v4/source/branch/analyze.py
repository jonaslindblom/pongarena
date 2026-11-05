from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np
from pace_study import tables,CONFIGS
from handcrafted_v4 import target_tables
T=tables(CONFIGS['pace13-v9-scaled']);H=np.arange(4,157);C=np.arange(15,146)
R=np.full((153,131),np.inf)
for i,y in enumerate(H):
 for j,c in enumerate(C):
  if abs(y-c)<=15:R[i,j]=T['risk'][13,y,y-c+15]
safe=R<=0
# Half-integer target c+.5 has only two possible stationary integer centers.
P=(R[:,:-1]<=0)&(R[:,1:]<=0)
# Integer target c has all threepossible centers atdistance1.
Q=safe.copy();Q[:,1:]&=safe[:,:-1];Q[:,:-1]&=safe[:,1:]
def spans(v,coords):
 ix=np.flatnonzero(v);blocks=np.split(ix,np.flatnonzero(np.diff(ix)>1)+1)
 return [[float(coords[b[0]]),float(coords[b[-1]])] for b in blocks if len(b)]
report=dict(robust_pair_heights=int(P.any(1).sum()),robust_triple_heights=int(Q.any(1).sum()),rows=[])
for i,y in enumerate(H):
 report['rows'].append(dict(y=int(y),safe=spans(safe[i],C),pairs=spans(P[i],C[:-1]+.5),triples=spans(Q[i],C),best=float(R[i].min())))
Path('/tmp/rally-safe-branch/analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['robust_pair_heights'],report['robust_triple_heights'])
for row in report['rows']: print(row)
