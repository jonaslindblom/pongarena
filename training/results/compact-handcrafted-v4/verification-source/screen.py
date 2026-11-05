from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np, torch
from pace_study import CONFIGS,decode,run_pair,make_policy
from handcrafted_v4 import target_tables,expected_scores
from competition_eval import load_mlp
from policy import ROOT,load_builtin

torch.set_num_threads(1)
rules=CONFIGS['pace13-v9-scaled']; incoming,outgoing=target_tables()
seeds=(np.random.default_rng(2026110509).choice(50_000_000,24,replace=False)+4_000_000_000).tolist()
knots=np.array([4,8,16,24,40,80,120,136,144,152,156])
# Fixed geometry hypotheses, not an outcome-optimized parameter sweep.
def policy(kind):
 def decide(obs):
  vx,own,_,_,h=decode(obs)
  stage=np.minimum(13,abs(vx))
  if kind=='full-reference':target=np.where(vx<0,incoming[stage,h],outgoing[stage,h])
  elif kind=='fixed-cap-table':target=np.where(vx<0,incoming[13,h],outgoing[13,h])
  elif kind=='fixed-cap-contact-center-recovery':target=np.where(vx<0,incoming[13,h],80)
  elif kind=='center-contact-cap-recovery':target=np.where(vx<0,np.clip(h,15,145),outgoing[13,h])
  elif kind=='center-contact-wall-ramp':target=np.where(vx<0,np.clip(h,15,145),80+np.maximum(0,24-h)*2-np.maximum(0,h-136)*2)
  elif kind=='cap-contact-coarse-recovery':target=np.where(vx<0,incoming[13,h],np.interp(h,knots,outgoing[13,knots]))
  elif kind=='coarse-cap-tables':target=np.where(vx<0,np.interp(h,knots,incoming[13,knots]),np.interp(h,knots,outgoing[13,knots]))
  else:raise ValueError(kind)
  return np.where(target<own-1.001,0,np.where(target>own+1.001,2,1))
 return decide
opponents={'pacer':load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'),'tracker':load_builtin('tracker'),'anticipator':load_builtin('anticipator'),'full-handcrafted':lambda obs:expected_scores(obs).argmax(1),'attack':make_policy(rules,'attack'),'defend':make_policy(rules,'defend')}
result={'purpose':'Target approximation with exact scalar integer landing; trajectory encoding cost excluded','seeds':seeds,'knots':knots.tolist(),'pairs':{}}
for name in ['fixed-cap-table','fixed-cap-contact-center-recovery','center-contact-cap-recovery','center-contact-wall-ramp','cap-contact-coarse-recovery','coarse-cap-tables']:
 for other,op in opponents.items():
  row=run_pair(rules,policy(name),op,seeds)
  result['pairs'][name+':'+other]=row
  Path('/tmp/rally-compact-targets/results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
  s=row['summary'];print(name,other,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
print('Done',flush=True)
