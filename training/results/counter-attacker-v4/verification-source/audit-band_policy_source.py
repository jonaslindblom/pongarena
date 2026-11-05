from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np, torch
from pace_study import CONFIGS,decode,run_pair,make_policy,tables,NEXT
from policy import ROOT
from competition_eval import load_mlp
from env import reflect

torch.set_num_threads(1)
rules=CONFIGS['pace13-v9-scaled'];tab=tables(rules)
seeds=(np.random.default_rng(2026110518).choice(30_000_000,24,replace=False)+4_050_000_000).tolist()
k=np.arange(-9,10)[None,:,None]
def band_policy(obs):
 vx,own,other,t,h=decode(obs);s=np.minimum(13,abs(vx));out=np.minimum(13,s+1)
 recovery=tab['mid'][s,h]
 res=np.argmin(abs(NEXT[own-15]-recovery[:,None])+[1e-6,0,1e-6],axis=1)
 ids=np.flatnonzero(vx<0)
 if not len(ids):return res
 h,p,op,t,s,out=h[ids],own[ids],other[ids],t[ids],s[ids],out[ids]
 nextp=NEXT[p-15][:,None,:]
 dscale=(65/(3*out))[:,None,None]
 lo=np.maximum(np.maximum(h[:,None,None]-(k+.5)*dscale,h[:,None,None]-15),15)
 hi=np.minimum(np.minimum(h[:,None,None]-(k-.5)*dscale,h[:,None,None]+15),145)
 radius=2*(t-1)[:,None,None]
 gap=np.maximum(0,np.maximum(lo-nextp-radius,nextp-hi-radius))
 # Include impossible legal-center intervals themselves in the feasibility gap.
 gap=np.maximum(gap,np.maximum(0,lo-hi))
 flight=np.ceil(330/out).astype(int)
 dest=reflect(h[:,None]+flight[:,None]*np.arange(-9,10)[None,:])
 expected=np.clip(tab['mid'][s,h],op-2*np.maximum(0,t-1),op+2*np.maximum(0,t-1))
 margin=abs(dest-expected[:,None])-15-2*flight[:,None]-2
 immediate=np.maximum(0,np.maximum(lo-nextp,nextp-hi))
 quality=margin[:,:,None]-1000*gap-.001*immediate
 score=quality.max(1)-np.array([1e-6,0,1e-6])[None,:]
 res[ids]=score.argmax(1)
 return res
if __name__=='__main__':
 opponents={'attack':make_policy(rules,'attack'),'defend':make_policy(rules,'defend'),'pacer':load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt')}
 result={'purpose':'19-vy-band receding-horizon attack, relaxed parity-aware future reach; no risk table, no fitting','seeds':seeds,'pairs':{}}
 for name,op in opponents.items():
  row=run_pair(rules,band_policy,op,seeds);result['pairs'][name]=row
  Path('/tmp/rally-attacker-targets/band-results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
  sm=row['summary'];print(name,sm['wins'],sm['losses'],sm['draws'],sm['point_margin_ci95'],flush=True)
