"""Enumerate legal cap shots offline; no matches, fitting or candidate selection.

Start fromeachstationaryintegercenteradmittedbythefrozenrobustgoal. Center
untilthefirstwallbounce, thenapplythedeclared8hingephysicalintervalrecoveryprofile.
Reportbothphysicalintervalreachabilityandtheactualtwo-unitfolloweroutcome.
"""
from pathlib import Path
import sys,importlib.util,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
spec=importlib.util.spec_from_file_location('safe_branch','/tmp/rally-safe-branch/build.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
import numpy as np
from pace_study import CONFIGS,tables
from env import reflect
OUT=Path('/tmp/rally-safe-branch');T=tables(CONFIGS['pace13-v9-scaled'])

KNOTS=np.array([-9,4,15,22,138,145,156,169.]);VALUES=np.array([80,93,82,78.5,81.5,78,67,80.])
def profile(q):
 return float(np.interp(q,KNOTS,VALUES))

def move(p,t):return int(np.clip(p+(2 if t-p>1.101 else -2 if p-t>1.101 else 0),15,145))
rows=[]
for h in range(4,157):
 goal=float(builder.goal(h));centers=[p for p in range(15,146)if abs(p-goal)<=1.101]
 if not all(abs(h-p)<=15 and T['risk'][13,h,h-p+15]<=0 for p in centers):continue
 for p0 in centers:
  vy=int(np.floor(3*(h-p0)/5+.5));landing=int(reflect(h+26*vy));low=int(T['low'][13,landing]);high=int(T['high'][13,landing])
  y=h;v=vy;p=p0;first=None;atbounce=None;targets=[]
  for tick in range(26):
   q=y+(26-tick)*v;target=profile(q);targets.append(target);p=move(p,target)
   y+=v
   if y<4 or y>156:
    y=8-y if y<4 else 312-y;v=-v
    if first is None:first=tick+1;atbounce=p
  assert y==landing
  recovery_ticks=26-first
  reachable_low=max(15,atbounce-2*recovery_ticks);reachable_high=min(145,atbounce+2*recovery_ticks)
  possible=max(low,reachable_low)<=min(high,reachable_high)
  exact_reachable={atbounce}
  for step in range(recovery_ticks):exact_reachable={int(np.clip(p+a,15,145))for p in exact_reachable for a in [-2,0,2]}
  exact_possible=any(low<=p<=high for p in exact_reachable)
  # Exactminimumdistancefromcentering-atbounce tothereplyinterval.
  best_risk=max((low-high)/2,low-reachable_high,reachable_low-high)
  risk=max(low-p,p-high)
  rows.append(dict(height=h,start=p0,vy=vy,landing=landing,first_bounce=first,own_at_bounce=atbounce,remaining_ticks=recovery_ticks,low=low,high=high,interval_reachable_after_bounce=bool(possible),lattice_interval_reachable_after_bounce=bool(exact_possible),best_recovery_risk=float(best_risk),final_own=p,coarse_goal=float(profile(y)),actual_reply_interval_risk=risk,coarse_goal_reply_interval_risk=float(max(low-profile(y),profile(y)-high))))
summary=dict(purpose='Onefixed8hingephysicalintervalrecoveryaudit,notgameplayselection',knots=KNOTS.tolist(),values=VALUES.tolist(),shots=len(rows),robust_contact_heights=len(set(r['height']for r in rows)),max_first_bounce=max(r['first_bounce']for r in rows),safe_interval_reachable_after_centering=sum(r['interval_reachable_after_bounce']for r in rows),unreachable_after_centering=sum(not r['interval_reachable_after_bounce']for r in rows),lattice_interval_reachable_after_centering=sum(r['lattice_interval_reachable_after_bounce']for r in rows),lattice_unreachable_after_centering=sum(not r['lattice_interval_reachable_after_bounce']for r in rows),lattice_unreachable_destinations=sorted(set(r['landing']for r in rows if not r['lattice_interval_reachable_after_bounce'])),max_best_recovery_risk=max(r['best_recovery_risk']for r in rows),actual_coarse_follower_safe=sum(r['actual_reply_interval_risk']<=0 for r in rows),actual_coarse_follower_unsafe=sum(r['actual_reply_interval_risk']>0 for r in rows),max_coarse_follower_risk=max(r['actual_reply_interval_risk']for r in rows),coarse_goal_itself_unsafe=sum(r['coarse_goal_reply_interval_risk']>0 for r in rows),coarse_unsafe_destinations=sorted(set(r['landing']for r in rows if r['coarse_goal_reply_interval_risk']>0)),follower_unsafe_destinations=sorted(set(r['landing']for r in rows if r['actual_reply_interval_risk']>0)))
(OUT/'profile8-recovery-audit.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2)+'\n');print(json.dumps(summary,indent=2));print('unreachable',json.dumps([r for r in rows if not r['interval_reachable_after_bounce']],indent=2));print('unsafeexample',json.dumps([r for r in rows if r['actual_reply_interval_risk']>0][:12],indent=2))
