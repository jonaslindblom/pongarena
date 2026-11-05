"""One fixed connected directed-family proposal; physical constraints only."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np
from pace_study import tables,CONFIGS
from env import reflect
OUT=Path('/tmp/rally-directed-branches');T=tables(CONFIGS['pace13-v9-scaled'])
H=np.arange(4,157);C=np.arange(15,146)
BASE_H=np.array([4,46,48,78,82,112,114,156.]);BASE_V=np.array([15.5,57.5,61.5,91.5,68.5,98.5,102.5,144.5])

def base(y):return np.interp(y,BASE_H,BASE_V)
def window(y):return np.maximum(0,y-34)-np.maximum(0,y-35)-np.maximum(0,y-74)+np.maximum(0,y-75)
def top(y):return base(y)-10*window(y)
def bottom(y):return base(y)+10*window(160-y)

def raw_hinges(heights,values):
 raw=[]
 for n in range(-3,4):
  for h in heights:raw.extend([h+304*n,312-h+304*n])
 raw=np.unique(np.r_[-240,raw,400]);raw=raw[(raw>=-240)&(raw<=400)]
 target=np.interp(reflect(raw),heights,values);sl=np.diff(target)/np.diff(raw);co=np.r_[sl[0],np.diff(sl)]
 used=np.flatnonzero(abs(co)>1e-12)
 return dict(raw=raw.tolist(),values=target.tolist(),hinge_indices=used.tolist(),hinge_coefficients=co[used].tolist(),active_hinges=len(used))

report=dict(purpose='Deterministicphysicalconstraintproposal;nomatches,nofit,nomutation',base_height_knots=BASE_H.tolist(),base_values=BASE_V.tolist(),window_height_knots=[4,34,35,74,75,156],window_values=[0,0,1,1,0,0],selector='opponentcenter>=81usestopbranch;<=80usesbottombranch;integerthresholdonly',profiles={})
for name,fn in [('base',base),('top',top),('bottom',bottom)]:
 counts=dict(robust_safe_heights=0,all_stay_centers_catch=0,all_destinations_top=0,all_destinations_bottom=0);rows=[]
 for h in H:
  g=float(fn(h));possible=C[abs(C-g)<=1.101];risk=[];dest=[]
  catch=all(abs(h-p)<=15 for p in possible)
  for p in possible:
   if abs(h-p)<=15:risk.append(float(T['risk'][13,h,h-p+15]));dest.append(int(T['landing'][13,h,h-p+15]))
   else:risk.append(None)
  safe=catch and all(r<=0 for r in risk);counts['robust_safe_heights']+=safe;counts['all_stay_centers_catch']+=catch
  counts['all_destinations_top']+=safe and max(dest)<=22;counts['all_destinations_bottom']+=safe and min(dest)>=138
  rows.append(dict(height=int(h),goal=g,possible_centers=possible.tolist(),safe=bool(safe),destinations=dest,risk=risk))
 report['profiles'][name]=dict(counts=counts,rows=rows)
base_raw=raw_hinges(BASE_H,BASE_V);a_raw=raw_hinges(np.array([4,34,35,74,75,156]),np.array([0,0,1,1,0,0]));b_raw=raw_hinges(np.array([4,85,86,125,126,156]),np.array([0,0,1,1,0,0]));report['raw']=dict(base=base_raw,top_window=a_raw,bottom_window=b_raw)
N=41;K=base_raw['active_hinges']+1+a_raw['active_hinges']+b_raw['active_hinges'];report['proposal_architecture']=[6,N,K,3];report['parameters']=7*N+(N+4)*K+3;report['macs']=6*N+N*K+3*K
(OUT/'feasibility.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(counts={k:v['counts']for k,v in report['profiles'].items()},hinges={k:v['active_hinges']for k,v in report['raw'].items()},architecture=report['proposal_architecture'],parameters=report['parameters'],macs=report['macs']),indent=2))
