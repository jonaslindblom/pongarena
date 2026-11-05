"""Geometry diagnosis of one fixed analytic dense candidate, not weight fitting."""
from pathlib import Path
import sys,json,importlib.util
sys.path.insert(0,str(Path.cwd()/'training/scripts'));sys.path.insert(0,'/tmp/rally-analytic-dense')
from farcorner import candidates
import numpy as np,torch
from pace_study import decode
from env import reflect
OUT=Path('/tmp/rally-analytic-dense');source=Path('/tmp/rally-causal-failures/audit_pacer_failures.py')
spec=importlib.util.spec_from_file_location('candidate_failure_geometry',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(1);seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds'];net=candidates()['stable-cap13-safeaim']
data,summaries,rallies=m.collect(seeds,net);reach,catch=m.reachability();failures=m.classify(data,rallies,reach,catch)
obs=data['observations'];vx,own,other,ticks,h=decode(obs);raw=data['raw'];x=np.rint(obs[:,0].astype(float)*180+180).astype(int);y=np.rint(obs[:,1].astype(float)*80+80).astype(int);vy=np.rint(obs[:,3].astype(float)*6).astype(int);distance=np.where(vx<0,x-15,345-x)
fore=reflect(y+vy*(distance/13+.5));cue=np.clip(10*obs[:,5],-1,1)*6
target=80+80*((data['scores'][:,2]-data['scores'][:,0])/2+obs[:,4])
report=dict(purpose='One fixed503parameter dense candidate failure diagnosis; no fitting',candidate='stable-cap13-safeaim',seeds=seeds,summaries={},examples=[])
for opponent in ['attack','defend']:
 rows=[f for f in failures if f['opponent']==opponent]
 errors=[f for f in rows if f.get('first_incoming_irreversible_row') is not None]
 ids=np.array([f['first_incoming_irreversible_row'] for f in errors],int)
 summary=dict(wins=summaries[opponent]['wins'],losses=summaries[opponent]['losses'],draws=summaries[opponent]['draws'],failure_categories={k:sum(f['category']==k for f in rows) for k in ['incoming_interception_error','lost_reachable_universal_recovery','no_universal_one_reply_safe_center','safe_region_not_reachable_after_own_shot']},incoming_irreversible_by_speed={str(s):sum(abs(vx[i])==s for i in ids) for s in range(5,14)},incoming_errors=len(ids))
 if len(ids):
  summary.update(mean_abs_capforecast_error_at_irreversible=float(abs(fore[ids]-h[ids]).mean()),mean_abs_square_approx_error_at_irreversible=float(abs(target[ids]-cue[ids]-fore[ids]).mean()),mean_abs_intended_aim=float(abs(cue[ids]).mean()))
 report['summaries'][opponent]=summary
 for f in errors[:8]:
  i=f['first_incoming_irreversible_row'];report['examples'].append(dict(opponent=opponent,speed=int(abs(vx[i])),tick=int(raw[i,7]),remaining=int(ticks[i]),x=int(x[i]),ball_y=int(y[i]),vy=int(vy[i]),own_y=int(own[i]),actual_landing=int(h[i]),cap_forecast=float(fore[i]),network_target=float(target[i]),opponent_cue=float(cue[i]),chosen=int(data['actions'][i,0]),safe_actions=np.flatnonzero(data['catch_preserving_actions'][i]).tolist(),rally=f['rally'],returns=f['returns']))
(OUT/'failure-diagnosis.json').write_text(json.dumps(report,indent=2,default=lambda v:v.item())+'\n')
print(json.dumps(report['summaries'],indent=2,default=lambda v:v.item()),flush=True)
