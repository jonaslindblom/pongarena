"""Geometry-only recovery requirements for a fixed physical contact goal.

No model fitting, gameplay reward selection, training or new runtime graph.
The fixed-point test is an offline controlled-invariant diagnostic, not a policy.
"""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np
from pace_study import CONFIGS,tables
from offense_eval import contact_paths
OUT=Path('/tmp/rally-policy-recovery');T=tables(CONFIGS['pace13-v9-scaled'])
H=np.arange(4,157);C=np.arange(15,146);F=26
KN=np.array([4,50,52,78,82,108,110,156.]);VAL=np.array([15.5,61.5,65.5,91.5,68.5,94.5,98.5,144.5])
def goal(y):return np.interp(y,KN,VAL)
# Exact finite movement reach, including clamp-induced parity changes.
R=contact_paths()[F,:,:,3]>=0
D={int(h):np.unique(T['landing'][13,h,T['legal'][h]].astype(int)) for h in H}
G=goal(np.arange(157));L=np.array([G[D[int(h)]].max()-2*F for h in H]);U=np.array([G[D[int(h)]].min()+2*F for h in H]);L=np.maximum(15,L);U=np.minimum(145,U)
# Target deadband is discrete: at each contact height, every stationary center
# lies within1.101of g and must still physically hit the ball.
selected=(abs(C[None,:]-G[:,None])<=1.101)&(abs(np.arange(157)[:,None]-C[None,:])<=15)
selected[:4]=False
reachable_goal=selected@R.T
exact=np.stack([reachable_goal[D[int(h)]].all(0) for h in H])
# Optional robust requirement: every stationary center should be reachable;
# this is stricter than existence and highlights parity/control uncertainty.
rows=[]
for i,h in enumerate(H):
 dest=D[int(h)];gg=G[dest];rows.append(dict(height=int(h),destinations=dest.tolist(),min_goal=float(gg.min()),max_goal=float(gg.max()),lower=float(L[i]),upper=float(U[i]),width=float(U[i]-L[i]),empty=bool(L[i]>U[i]),goal_midpoint=float((L[i]+U[i])/2),exact_discrete_goal_positions=C[exact[i]].tolist(),old_catch_lower=float(T['low'][13,h]),old_catch_upper=float(T['high'][13,h]),old_catch_mid=float(T['mid'][13,h])))
# A=positions just before the opponent's return from contact height H.
# For each possible enemy destination d, require a reachable own contact c
# within the fixed goal deadband, followed by recovery into A at the next
# opponent contact. Existential choices make this more permissive than the NN.
A=np.zeros((157,131),bool);A[4:]=True;history=[int(A.sum())]
centers=np.arange(157)[:,None]-np.arange(-15,16)[None,:]
for iteration in range(1000):
 recover=A@R.T
 good=np.zeros_like(A)
 for d in H:
  for cidx in np.flatnonzero(selected[d]):
   c=int(C[cidx]);offset=int(d-c);nextH=int(T['landing'][13,d,offset+15])
   good[d,cidx]=recover[nextH,cidx]
 can=good@R.T
 following=np.zeros_like(A)
 for h in H:following[h]=can[D[int(h)]].all(0)
 assert not (following&~A).any()
 history.append(int(following.sum()))
 if np.array_equal(following,A):A=following;break
 A=following
else:raise AssertionError('fixedpoint did not converge')
# Affine recovery r(h)=80-k*(h-80). Solve all nonempty interval constraints
# exactly as scalar inequalities; no least-squares/gameplay fitting.
x=H-80;valid=L<=U;klow=[];khigh=[]
for i in np.flatnonzero(valid):
 if x[i]>0:klow.append(((80-U[i])/x[i],int(H[i])));khigh.append(((80-L[i])/x[i],int(H[i])))
 elif x[i]<0:klow.append(((80-L[i])/x[i],int(H[i])));khigh.append(((80-U[i])/x[i],int(H[i])))
 else:assert L[i]<=80<=U[i]
klo=max(klow);khi=min(khigh)
# The unique wall-midpoint slope is a physically specified control, not tuned.
wall_k=((L[0]+U[0])/2-80)/(80-H[0]);r=80-wall_k*x;violation=np.maximum(L-r,r-U)
old=80+0*x;old_violation=np.maximum(L-old,old-U)
report=dict(purpose='Fixed-goal recovery geometry, no fitting/rewardselection',frozen_source_sha256=hashlib.sha256(Path('/tmp/rally-safe-branch/build.py').read_bytes()).hexdigest(),goal_knots=KN.tolist(),goal_values=VAL.tolist(),heights=len(H),float_intervals=dict(nonempty=int(valid.sum()),empty=int((~valid).sum()),empty_heights=H[~valid].tolist(),min_width=float((U-L).min()),max_width=float((U-L).max())),exact_goal_reach=dict(heights_with_any_position=int(exact.any(1).sum()),empty_heights=H[~exact.any(1)].tolist(),positions=int(exact.sum())),controlled_invariant=dict(state_count_history=history,iterations=iteration+1,remaining_states=int(A.sum()),heights_with_any_position=int(A[4:].any(1).sum()),interpretation='Permissive controlledgoalpolicy with arbitrary recovery and any stationary goal center; emptiness rules out this goal guarantee, nonempty does not prove fixedNN sustainsit'),affine_constraints=dict(lower=float(klo[0]),lower_height=klo[1],upper=float(khi[0]),upper_height=khi[1],feasible=bool(klo[0]<=khi[0]),scope='All nonemptycontinuousRgintervals'),wall_midpoint_affine=dict(k=float(wall_k),root_wall_midpoint=float((L[0]+U[0])/2),covered_all_heights=int((violation<=0).sum()),covered_nonempty=int(((violation<=0)&valid).sum()),unsafe_nonempty=H[(violation>0)&valid].tolist(),worst_violation=float(violation.max()),worst_nonempty_violation=float(violation[valid].max())),center80=dict(covered_all_heights=int((old_violation<=0).sum()),covered_nonempty=int(((old_violation<=0)&valid).sum()),worst_violation=float(old_violation.max())),rows=rows)
(OUT/'goal-recovery.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(OUT/'goal-recovery.npz',heights=H,lower=L,upper=U,exact_goal_reach=exact,controlled_invariant=A,selected_centers=selected)
print(json.dumps({k:v for k,v in report.items() if k not in ['rows']},indent=2))
