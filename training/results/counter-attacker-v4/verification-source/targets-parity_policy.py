"""Exact discrete contact-center reach per velocity band; fixed formula policies."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
sys.path.insert(0,'/tmp/rally-attacker-audit')
import numpy as np, torch
from pace_study import CONFIGS,decode,run_pair,make_policy,tables,NEXT
from policy import ROOT,OnnxPolicy
from competition_eval import load_mlp
from env import reflect
from screen_endpoint import recovery_policy

torch.set_num_threads(1)
RULES=CONFIGS['pace13-v9-scaled'];T=tables(RULES)
SEEDS=(np.random.default_rng(2026110511).choice(50_000_000,24,replace=False)+4_100_000_000).tolist()
ENDPOINT=recovery_policy('predicted_endpoint','attack')

def parity_band(obs,endpoint=False,cap_risk=False):
 vx,own,other,ticks,height=decode(obs);speed=np.minimum(13,abs(vx))
 target=T['mid'][speed,height]
 result=np.argmin(abs(NEXT[own-15]-target[:,None])+[1e-6,0,1e-6],axis=1)
 if endpoint:
  away=vx>0
  if away.any():result[away]=ENDPOINT(obs[away])
 ids=np.flatnonzero(vx<0)
 if not len(ids):return result
 h,p,op,t,s=height[ids],own[ids],other[ids],ticks[ids],speed[ids]
 out=np.minimum(13,s+1);F=np.ceil(330/out).astype(int)
 k=np.arange(-9,10)[None,:,None]
 lo=np.ceil(np.maximum(np.maximum(h[:,None,None]-(k+.5)*(65/(3*out))[:,None,None],h[:,None,None]-15),15)-1e-7).astype(int)
 hi=np.floor(np.minimum(np.minimum(h[:,None,None]-(k-.5)*(65/(3*out))[:,None,None],h[:,None,None]+15),145)+1e-7).astype(int)
 nextp=NEXT[p-15][:,None,:];remaining=t[:,None,None]-1
 regions=[]
 # Normal2-unit motion keeps parity. Even positions can change to odd only by
 # clamping at15or145, after which all later positions remain odd.
 normalL=np.maximum(lo,nextp-2*remaining);normalH=np.minimum(hi,nextp+2*remaining)
 regions.append((normalL,normalH,nextp%2,np.ones_like(nextp,dtype=bool)))
 for boundary in [15,145]:
  towall=np.ceil(abs(nextp-boundary)/2).astype(int)
  rem=remaining-towall
  L=np.maximum(lo,15 if boundary==15 else 145-2*rem)
  H=np.minimum(hi,15+2*rem if boundary==15 else 145)
  regions.append((L,H,np.ones_like(nextp),((nextp%2)==0)&(rem>=0)))
 dest=reflect(h[:,None]+F[:,None]*np.arange(-9,10)[None,:]).astype(int)
 expected=np.clip(T['mid'][s,h],op-2*np.maximum(0,t-1),op+2*np.maximum(0,t-1))
 margin=abs(dest-expected[:,None])-15-2*F[:,None]-2
 if cap_risk:
  low=T['low'][13,dest][:,:,None];high=T['high'][13,dest][:,:,None]
 best=np.full((len(ids),19,3),-1e12)
 for L,H,parity,allowed in regions:
  qlo=L+(parity-L)%2;qhi=H-(H-parity)%2
  valid=(qlo<=qhi)&allowed
  # Band width≤3.62, so each parity interval contains at most two integers.
  # Evaluating both ends exhausts all legal centers, including clamp paths.
  for center in [qlo,qhi]:
   risk=np.maximum(np.maximum(low-center-2*F[:,None,None],center-2*F[:,None,None]-high),(low-high)/2) if cap_risk else 0
   quality=margin[:,:,None]-.05*np.maximum(0,risk)-.001*abs(center-p[:,None,None])-1e-6*abs(center-nextp)
   best=np.maximum(best,np.where(valid,quality,-1e12))
 scores=best.max(1)-np.array([1e-9,0,1e-9])
 fallback=np.argmin(abs(NEXT[p-15]-np.clip(h,15,145)[:,None])+[1e-6,0,1e-6],axis=1)
 result[ids]=np.where(np.max(scores,axis=1)>-1e11,scores.argmax(1),fallback)
 return result

if __name__=='__main__':
 opponents={'attack':make_policy(RULES,'attack'),'pacer':load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'),'craft-small':OnnxPolicy(ROOT/'training/results/compact-handcrafted-v4/handcrafted-small-v4.onnx')}
 candidates={'parity-standard':lambda obs:parity_band(obs), 'parity-endpoint':lambda obs:parity_band(obs,endpoint=True),'parity-cap-risk-endpoint':lambda obs:parity_band(obs,endpoint=True,cap_risk=True)}
 record={'purpose':'Parity-corrected19-vy band reach, exact future wall clamps; fixed teacher constants; no fitting','rng':2026110511,'seed_range':[4100000000,4150000000],'seeds':SEEDS,'pairs':{}}
 for name,policy in candidates.items():
  for other,op in opponents.items():
   row=run_pair(RULES,policy,op,SEEDS);record['pairs'][name+':'+other]=row
   Path('/tmp/rally-attacker-targets/parity-results.json').write_text(json.dumps(record,separators=(',',':'))+'\n')
   s=row['summary'];print(name,other,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],'paired',s['paired_match_wins'],s['paired_match_losses'],s['paired_match_draws'],flush=True)
