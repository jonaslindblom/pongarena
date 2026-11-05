"""Exhaustivelegal cap13 directedshotintervals; no games or fitting."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np
from pace_study import tables,CONFIGS
from env import reflect
OUT=Path('/tmp/rally-directed-branches');T=tables(CONFIGS['pace13-v9-scaled'])
H=np.arange(4,157);C=np.arange(15,145)+.5
D=np.zeros((len(H),len(C),2),int);R=np.full_like(D,np.inf,dtype=float)
for i,h in enumerate(H):
 for j,g in enumerate(C):
  for q,p in enumerate([int(g-.5),int(g+.5)]):
   off=h-p
   if abs(off)<=15:
    R[i,j,q]=T['risk'][13,h,off+15];D[i,j,q]=T['landing'][13,h,off+15]
safe=(R<=0).all(2)
masks=dict(top=safe&(D<=22).all(2),bottom=safe&(D>=138).all(2))
report=dict(top_heights=int(masks['top'].any(1).sum()),bottom_heights=int(masks['bottom'].any(1).sum()),both_heights=int((masks['top'].any(1)&masks['bottom'].any(1)).sum()),either_heights=int((masks['top'].any(1)|masks['bottom'].any(1)).sum()),rows=[],diagonal_intervals={})
for i,h in enumerate(H):report['rows'].append(dict(height=int(h),top_offsets=(C[masks['top'][i]]-h).tolist(),bottom_offsets=(C[masks['bottom'][i]]-h).tolist()))
for side,mask in masks.items():
 report['diagonal_intervals'][side]={}
 for delta in np.unique((C[None]-H[:,None])[mask]):
  good=mask[np.arange(len(H)),np.clip(np.rint(H+delta-15.5).astype(int),0,len(C)-1)]&((H+delta>=15.5)&(H+delta<=144.5))
  ix=np.flatnonzero(good);blocks=np.split(ix,np.flatnonzero(np.diff(ix)>1)+1)
  spans=[[int(H[b[0]]),int(H[b[-1]])]for b in blocks if len(b)]
  report['diagonal_intervals'][side][str(float(delta))]=spans
(OUT/'directed-intervals.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k!='rows'},indent=2))
for row in report['rows']:print(row)
