"""One graph-budget simplification: normal parity only, no future clamp paths."""
from pathlib import Path
import sys,json
sys.path.insert(0,'/tmp/rally-attacker-targets')
import numpy as np
source=Path('/tmp/rally-attacker-targets/parity_policy.py').read_text().split("if __name__=='__main__':")[0]
source=source.replace('for boundary in [15,145]:','for boundary in []:')
exec(source)
T=dict(T);T['mid']=np.broadcast_to(T['mid'][13:14],T['mid'].shape).copy()
def normal_endpoint(obs):
 vx,own,other,t,h=decode(obs);s=abs(vx);out=np.minimum(13,s+1);F=np.ceil(330/out).astype(int)
 k=np.arange(-9,10)[None,:]
 lo=np.ceil(np.maximum(np.maximum(h[:,None]-(k+.5)*(65/(3*out))[:,None],h[:,None]-15),15)-1e-7).astype(int)
 hi=np.floor(np.minimum(np.minimum(h[:,None]-(k-.5)*(65/(3*out))[:,None],h[:,None]+15),145)+1e-7).astype(int)
 p=other[:,None]
 L=np.maximum(lo,p-2*t[:,None]);H=np.minimum(hi,p+2*t[:,None])
 qlo=L+(p-L)%2
 valid=qlo<=H
 dest=reflect(h[:,None]+F[:,None]*k).astype(int)
 bottom=np.min(np.where(valid,dest,10000),axis=1);top=np.max(np.where(valid,dest,-10000),axis=1)
 expected=np.clip(T['mid'][s,h],own-2*np.maximum(0,t-1),own+2*np.maximum(0,t-1))
 target=np.where(abs(top-expected)>abs(bottom-expected),top,bottom)
 target=np.where(valid.any(1),target,80)
 return np.argmin(abs(NEXT[own-15]-target[:,None])+[1e-6,0,1e-6],axis=1)
ENDPOINT=normal_endpoint
policy=lambda obs:parity_band(obs,endpoint=True,cap_risk=True)
opponents={'attack':make_policy(RULES,'attack'),'pacer':load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'),'craft-small':OnnxPolicy(ROOT/'training/results/compact-handcrafted-v4/handcrafted-small-v4.onnx')}
record={'purpose':'Budget approximation: normal parity only, cap13 midpoint/risk, endpoint forecast; no fitting','rng':2026110511,'seeds':SEEDS,'pairs':{}}
for name,op in opponents.items():
 row=run_pair(RULES,policy,op,SEEDS);record['pairs'][name]=row
 Path('/tmp/rally-attacker-targets/normal-parity-results.json').write_text(json.dumps(record,separators=(',',':'))+'\n')
 sm=row['summary'];print(name,sm['wins'],sm['losses'],sm['draws'],sm['point_margin_ci95'],'paired',sm['paired_match_wins'],sm['paired_match_losses'],sm['paired_match_draws'],flush=True)
