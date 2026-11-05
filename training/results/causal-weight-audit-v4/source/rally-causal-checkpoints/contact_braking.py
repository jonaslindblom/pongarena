"""One source-derived contrast correction for learned unit56; offline attribution."""
from pathlib import Path
import sys,json,copy
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from pace_study import CONFIGS,tables,NEXT
from env import reflect
from policy import ROOT,export
OUT=Path('/tmp/rally-causal-checkpoints');torch.set_num_threads(1)
data=np.load('/tmp/rally-causal-failures/pacer-failure-states.npz');obs=data['observations'];scores=data['scores'];actions=data['actions'][:,0];raw=data['raw'];side=raw[:,8]
net=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');st=net.state_dict();w0,b0,w2,b2,w4,b4=[st[k].numpy().copy() for k in ['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']]
current=w4[1,56];replacement=(w4[0,56]+w4[2,56])/2;gain=float(replacement-current)
with torch.no_grad():
 h1=torch.cat([net[:2](x) for x in torch.from_numpy(obs).split(4096)]).numpy();z=h1@w2[56]+b2[56];a=np.tanh(z)
new_scores=scores.copy();new_scores[:,1]+=gain*(a+1);newactions=new_scores.argmax(1)
contact=data['contact_side']==side;ids=np.flatnonzero(contact);h=reflect(raw[ids,1]+raw[ids,3]).astype(int);speed=abs(np.rint(obs[ids,2]*6).astype(int));own=np.rint(obs[ids,4]*80+80).astype(int);offset=h[:,None]-NEXT[own-15];legal=abs(offset)<=15
risk=np.full((len(ids),3),np.nan);rr,cc=np.nonzero(legal);tab=tables(CONFIGS['pace13-v9-scaled']);risk[rr,cc]=tab['risk'][speed[rr],h[rr],(offset[rr,cc]+15).astype(int)]
base=risk[np.arange(len(ids)),actions[ids]];changed=risk[np.arange(len(ids)),newactions[ids]];ok=np.isfinite(changed);delta=np.where(ok,base-changed,-100)
report={'purpose':'One source-derived correction after causal feature interpretation, no games or label fitting','unit':56,'formula':'W4[stay,56]=(W4[up,56]+W4[down,56])/2; b4[stay]+=new-old; delta_stay=gain*(h56+1)','old_stay_weight':float(current),'new_stay_weight':float(replacement),'gain':gain,'output_direction_contrast_unchanged':float(w4[2,56]-w4[0,56]),'phases':{},'contact_risk':{},'context':{},'feature_derivatives':{},'upstream':[]}
for name,m in [('incoming_far',(obs[:,2]<0)&~contact),('contact',contact),('outgoing',obs[:,2]>0)]:
 changes=newactions[m]!=actions[m]
 report['phases'][name]={'states':int(m.sum()),'action_changes':int(changes.sum()),'action_change_rate':float(changes.mean()),'activation_mean':float(a[m].mean()),'activation_std':float(a[m].std()),'mean_stay_boost':float((gain*(a[m]+1)).mean())}
 d=(1-a[m]*a[m])[:,None]*w2[56][None,:]*(1-h1[m]*h1[m]);deriv=d@w0
 report['feature_derivatives'][name]={'names':['x','ball_y','vx','vy','own_y','other_y'],'mean':deriv.mean(0).tolist(),'mean_abs':abs(deriv).mean(0).tolist()}
report['contact_risk']={'improved':int((ok&(changed<base-1e-8)).sum()),'worsened':int((ok&(changed>base+1e-8)).sum()),'new_misses':int((~ok).sum()),'sum_reduction':float(delta.sum()),'mean_reduction':float(delta.mean()),'positive_to_nonpositive':int((ok&(base>0)&(changed<=0)).sum()),'nonpositive_to_positive':int((ok&(base<=0)&(changed>0)).sum())}
for name,m in [('attack',data['pair_index'][ids]==0),('defend',data['pair_index'][ids]==1),('slow5to9',speed<=9),('fast10to13',speed>=10),('top',h<54),('middle',(h>=54)&(h<=106)),('bottom',h>106)]:
 report['context'][name]={'states':int(m.sum()),'improved':int((m&ok&(changed<base-1e-8)).sum()),'worsened':int((m&ok&(changed>base+1e-8)).sum()),'new_misses':int((m&~ok).sum()),'sum_risk_reduction':float(delta[m].sum()),'activation_mean':float(a[ids[m]].mean())}
# Causal path contribution at contact, not merely raw largest coefficients.
m=contact;d=(1-a[m]*a[m])[:,None]*w2[56][None,:]*(1-h1[m]*h1[m]);importance=(abs(d)[:,:,None]*abs(w0[None,:,:])).mean((0,2))
for i in np.argsort(importance)[-8:][::-1]:
 report['upstream'].append({'first_unit':int(i),'second_connection':float(w2[56,i]),'first_weights':w0[i].tolist(),'first_bias':float(b0[i]),'mean_abs_chain':float(importance[i])})
model=copy.deepcopy(net)
with torch.no_grad():model[-1].weight[1,56]=float(replacement);model[-1].bias[1]+=gain
# The only two changes must exactly express the interpreted source contrast.
with torch.no_grad():actual=torch.cat([model(x) for x in torch.from_numpy(obs).split(4096)]).numpy()
np.testing.assert_allclose(actual-new_scores,0,atol=2e-5,rtol=0)
report['actual_forward_action_agreement']=float((actual.argmax(1)==newactions).mean())
torch.save(model.state_dict(),OUT/'contact-motion-56.pt');export(model,OUT/'contact-motion-56.onnx')
(OUT/'contact-motion-56.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
