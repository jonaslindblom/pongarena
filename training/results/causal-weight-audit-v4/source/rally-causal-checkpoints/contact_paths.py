"""Offline causal attribution of learned units at all original own contacts."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from pace_study import CONFIGS,tables,NEXT
from env import reflect
from policy import ROOT
OUT=Path('/tmp/rally-causal-checkpoints');torch.set_num_threads(1)
data=np.load('/tmp/rally-causal-failures/pacer-failure-states.npz')
obs=data['observations'];scores=data['scores'];chosen=data['actions'][:,0]
raw=data['raw'];side=raw[:,8];incoming=obs[:,2]<0;outgoing=~incoming
contacts=data['contact_side']==side
far=incoming&~contacts
net=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');st=net.state_dict()
w0,b0,w2,b2,w4,b4=[st[k].numpy().copy() for k in ['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']]
with torch.no_grad():h2=torch.cat([net[:4](x) for x in torch.from_numpy(obs).split(4096)]).numpy()
ids=np.flatnonzero(contacts);h=reflect(raw[ids,1]+raw[ids,3]).astype(int);stage=abs(np.rint(obs[ids,2]*6).astype(int));own=np.rint(obs[ids,4]*80+80).astype(int)
centers=NEXT[own-15];offset=h[:,None]-centers;legal=abs(offset)<=15
r=np.full((len(ids),3),np.nan);tab=tables(CONFIGS['pace13-v9-scaled'])
rr,cc=np.nonzero(legal);r[rr,cc]=tab['risk'][stage[rr],h[rr],(offset[rr,cc]+15).astype(int)]
base=r[np.arange(len(ids)),chosen[ids]];assert np.isfinite(base).all()
units=[]
for j in range(128):
 c=float(np.sign(np.median(h2[far,j])) or 1)
 changed_scores=scores-(h2[:,j,None]-c)*w4[:,j][None,:]
 actions=changed_scores.argmax(1);a=actions[ids];newr=r[np.arange(len(ids)),a];ok=np.isfinite(newr)
 delta=np.where(ok,base-newr,-100.)
 row=dict(unit=j,constant=c,output_votes=w4[:,j].tolist(),output_stay_vs_move=float(w4[1,j]-(w4[0,j]+w4[2,j])/2),activation={p:{'mean':float(h2[m,j].mean()),'std':float(h2[m,j].std()),'max_deviation':float(abs(h2[m,j]-c).max())} for p,m in [('incoming_far',far),('contacts',contacts),('outgoing',outgoing)]},action_changes={p:int((actions[m]!=chosen[m]).sum()) for p,m in [('incoming_far',far),('contacts',contacts),('outgoing',outgoing)]},contact_risk=dict(improve=int((ok&(newr<base-1e-8)).sum()),worsen=int((ok&(newr>base+1e-8)).sum()),miss=int((~ok).sum()),sum_reduction=float(delta.sum()),mean_reduction=float(delta.mean()),positive_to_nonpositive=int((ok&(base>0)&(newr<=0)).sum()),nonpositive_to_positive=int((ok&(base<=0)&(newr>0)).sum())),by_speed={},by_height={},by_opponent={})
 for key,values,groups in [('by_speed',stage,{str(v):stage==v for v in range(5,14)}),('by_height',h,{'top':h<54,'middle':(h>=54)&(h<=106),'bottom':h>106}),('by_opponent',data['pair_index'][ids],{'attack':data['pair_index'][ids]==0,'defend':data['pair_index'][ids]==1})]:
  for name,m in groups.items():
   row[key][name]=dict(count=int(m.sum()),improve=int((m&ok&(newr<base-1e-8)).sum()),worsen=int((m&ok&(newr>base+1e-8)).sum()),miss=int((m&~ok).sum()),risk_sum_reduction=float(delta[m].sum()))
 units.append(row)
units.sort(key=lambda x:x['contact_risk']['sum_reduction'],reverse=True)
report=dict(purpose='All-contact centered-ablation attribution; NOT proposed weight improvements or fitted labels',states=len(obs),contact_states=len(ids),base_contact_positive_risk=int((base>0).sum()),phase_counts=dict(incoming_far=int(far.sum()),outgoing=int(outgoing.sum())),units=units)
(OUT/'contact-attribution.json').write_text(json.dumps(report,indent=2)+'\n')
for row in units[:20]:print(row['unit'],row['contact_risk'],'changes',row['action_changes'],'std',[row['activation'][k]['std'] for k in ['incoming_far','contacts','outgoing']],flush=True)
