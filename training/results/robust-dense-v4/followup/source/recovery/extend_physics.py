"""Additional geometry diagnostics; no model selection or game outcomes."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np
from pace_study import tables,CONFIGS
from offense_eval import contact_paths
OUT=Path('/tmp/rally-policy-recovery');r=json.loads((OUT/'goal-recovery.json').read_text());T=tables(CONFIGS['pace13-v9-scaled']);H=np.arange(4,157);C=np.arange(15,146);R=contact_paths()[26,:,:,3]>=0;D={h:np.unique(T['landing'][13,h,T['legal'][h]].astype(int))for h in H}
selection=abs(np.arange(157)[:,None]-C[None])<=15;selection[:4]=False
A=np.zeros((157,131),bool);A[4:]=True;history=[int(A.sum())]
for it in range(1000):
 recover=A@R.T;good=np.zeros_like(A)
 for d in H:
  for j in np.flatnonzero(selection[d]):good[d,j]=recover[int(T['landing'][13,d,d-C[j]+15]),j]
 can=good@R.T;following=np.zeros_like(A)
 for h in H:following[h]=can[D[h]].all(0)
 assert not(following&~A).any();history.append(int(following.sum()))
 if np.array_equal(following,A):A=following;break
 A=following
r['unrestricted_cap_control']=dict(state_count_history=history,remaining_states=int(A.sum()),heights_with_any_position=H[A[4:].any(1)].tolist(),interpretation='Samecapcontrolled-invariantoperatorwithalllegalcontactcenters, tovalidateandcontrastthefixedgoalrestriction')
# Affine worst-goalreach violation consists only of known geometry lines.
L=np.array([q['lower']for q in r['rows']]);U=np.array([q['upper']for q in r['rows']]);x=H-80;sl=np.r_[x,-x];bi=np.r_[L-80,80-U];ks=[0.]
for i in range(len(sl)):
 for j in range(i):
  if sl[i]!=sl[j]:
   k=(bi[j]-bi[i])/(sl[i]-sl[j])
   if k>=0:ks.append(k)
ks=np.unique(ks);cost=(sl[:,None]*ks+bi[:,None]).max(0);j=cost.argmin();r['affine_all_height_geometric_minimax']=dict(k=float(ks[j]),worst_goalreach_violation_units=float(cost[j]),method='Exactintersections offinitephysicalinequalityenvelope,notgameplaycoefficientfitting',domain='k>=0')
# Compare actual corrected8-profile endpoints with the stronger preferred-goal
# reaching condition, using only finite-shot physics proof data.
proof=json.loads(Path('/tmp/rally-safe-branch/profile8-parity-recovery-audit.json').read_text());exact=np.load(OUT/'goal-recovery.npz')['exact_goal_reach'];rows=proof['rows'];catch_good=np.array([q['actual_reply_interval_risk']<=0 for q in rows]);goal_good=np.array([exact[q['landing']-4,q['final_own']-15]for q in rows]);r['eight_profile_future_goal_reach']=dict(shots=len(rows),catch_safe=int(catch_good.sum()),all_next_goals_reachable=int(goal_good.sum()),catch_safe_but_goal_reach_failed=int((catch_good&~goal_good).sum()),scope='Only248stationaryrobustcontactshots;notallgameplayoroutcomes')
(OUT/'goal-recovery.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k]for k in ['unrestricted_cap_control','affine_all_height_geometric_minimax','eight_profile_future_goal_reach']},indent=2))
