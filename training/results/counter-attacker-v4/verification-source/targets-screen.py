from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from pace_study import CONFIGS,decode,run_pair,make_policy,tables,NEXT
from offense_eval import contact_paths
from policy import ROOT,load_builtin
from competition_eval import load_mlp

torch.set_num_threads(1)
rules=CONFIGS['pace13-v9-scaled'];tab=tables(rules);paths=contact_paths()
seeds=(np.random.default_rng(2026110518).choice(30_000_000,24,replace=False)+4_050_000_000).tolist()

def policy(kind):
 def choose(obs):
  vx,own,other,ticks,h=decode(obs);stage=np.minimum(13,abs(vx));n=len(obs)
  recover=tab['mid'][stage,h]
  scores=-abs(NEXT[own-15]-recover[:,None])-np.array([1e-6,0,1e-6])
  result=scores.argmax(1);ids=np.flatnonzero(vx<0)
  if not len(ids):return result
  height,p,t,s=h[ids],own[ids],ticks[ids],stage[ids]
  centers=tab['centers'][height];first=paths[t[:,None],p[:,None]-15,np.clip(centers-15,0,130),3]
  valid=tab['legal'][height]&(first>=0);travel=abs(centers-p[:,None])
  projected=np.clip(tab['mid'][s,height],other[ids]-2*np.maximum(0,t-1),other[ids]+2*np.maximum(0,t-1))
  margin=abs(tab['landing'][s,height]-projected[:,None])-15-2*tab['flight'][s,None]-2
  risk=tab['risk'][s,height]
  if kind=='forcing-else-defense':
   forcing=(np.max(np.where(valid,margin,-1e12),axis=1)>0)
   quality=np.where(forcing[:,None],margin-.05*np.maximum(0,risk),-risk)-.001*travel
  elif kind=='risk-tiebreak-among-forcing':
   forcing=(margin>0)&valid;some=forcing.any(axis=1)
   quality=np.where(some[:,None],np.where(forcing,-risk,-1e12),-risk)-.001*travel
  else:raise ValueError(kind)
  best=np.argmax(np.where(valid,quality,-1e12),axis=1)
  fallback=np.argmin(abs(NEXT[p-15]-np.clip(height,15,145)[:,None])+[1e-6,0,1e-6],axis=1)
  result[ids]=np.where(valid.any(axis=1),first[np.arange(len(ids)),best],fallback)
  return result
 return choose
opponents={'attack':make_policy(rules,'attack'),'defend':make_policy(rules,'defend'),'pacer':load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt')}
result={'purpose':'Principled forcing-shot/defensive fallback variants; exact scalar table diagnostic; no fitting','seeds':seeds,'pairs':{}}
for name in ['forcing-else-defense','risk-tiebreak-among-forcing']:
 for other,op in opponents.items():
  row=run_pair(rules,policy(name),op,seeds);result['pairs'][name+':'+other]=row
  Path('/tmp/rally-attacker-targets/results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
  s=row['summary'];print(name,other,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
