"""Causal/phase audit of aligned dense learned checkpoints; no fitting or games."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from policy import ROOT
OUT=Path('/tmp/rally-causal-checkpoints');torch.set_num_threads(1)
obs=np.load('/tmp/rally-dense-transfer/observations.npy')
# These are retained reachable development observations, used for interpretation
# only, never claimed untouched performance data for new checkpoint interventions.
phase={'incoming':obs[:,2]<0,'outgoing':obs[:,2]>0}
paths={str(r):ROOT/f'training/results/pace-study/training/dagger-continuation-s412/round-{r}.pt' for r in [12,18,24]}
paths['36']=ROOT/'training/results/pace-study/training/dagger-continuation-s412/best.pt'
nets={name:load_mlp(p) for name,p in paths.items()}
values={};audit={'purpose':'Causal interpretation of aligned learned dense checkpoints; no fitting/training/game selection','observations':len(obs),'observation_source':'/tmp/rally-dense-transfer/observations.npy; retained reachable development states','phase_counts':{p:int(m.sum()) for p,m in phase.items()},'sources':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},'checkpoints':{}}
for label,net in nets.items():
 state={k:v.numpy().copy() for k,v in net.state_dict().items()}
 w0,b0,w2,b2,w4,b4=[state[k] for k in ['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']]
 with torch.no_grad():
  h1=net[:2](torch.from_numpy(obs)).numpy();h2=net[:4](torch.from_numpy(obs)).numpy();scores=net(torch.from_numpy(obs)).numpy()
 d1=1-h1*h1;d2=1-h2*h2;contrast=w4[2]-w4[0]
 own=d2*((d1*w0[:,4])@w2.T)*contrast
 ball=d2*((d1*w0[:,1])@w2.T)*contrast
 vy=d2*((d1*w0[:,3])@w2.T)*contrast
 other=d2*((d1*w0[:,5])@w2.T)*contrast
 chosen=scores.argmax(1)
 unit=[]
 for j in range(128):
  ablated=scores-h2[:,j,None]*w4[:,j][None,:];changed=ablated.argmax(1)!=chosen
  row={'unit':j,'output_down_up_weight':float(contrast[j]),'output_weight':w4[:,j].tolist()}
  for name,mask in phase.items():
   row[name]={f'{key}_derivative_mean':float(arr[mask,j].mean()) for key,arr in [('own',own),('ball_y',ball),('vy',vy),('other',other)]}
   row[name].update({f'{key}_derivative_abs':float(abs(arr[mask,j]).mean()) for key,arr in [('own',own),('ball_y',ball),('vy',vy),('other',other)]})
   row[name].update(ablation_action_change=float(changed[mask].mean()),activation_mean=float(h2[mask,j].mean()),activation_std=float(h2[mask,j].std()))
  unit.append(row)
 audit['checkpoints'][label]={'units':unit,'whole_phase':{name:{'own_margin_derivative':float(own[mask].sum(1).mean()),'ball_margin_derivative':float(ball[mask].sum(1).mean()),'vy_margin_derivative':float(vy[mask].sum(1).mean()),'other_margin_derivative':float(other[mask].sum(1).mean())} for name,mask in phase.items()}}
 values[label]=dict(state=state,h1=h1,h2=h2,scores=scores,own=own,ball=ball,vy=vy,other=other)

# Correlations retain unit identities from the same continuation run; arbitrary
# independently initialized networks cannot be compared by neuron index.
current=values['36'];audit['drift']={}
for label in ['12','18','24']:
 prev=values[label];drift={}
 for layer in [0,2,4]:
  wprev=prev['state'][f'{layer}.weight'];wnow=current['state'][f'{layer}.weight']
  drift[str(layer)]={'relative_weight_l2':float(np.linalg.norm(wnow-wprev)/np.linalg.norm(wprev)),'relative_bias_l2':float(np.linalg.norm(current['state'][f'{layer}.bias']-prev['state'][f'{layer}.bias'])/(np.linalg.norm(prev['state'][f'{layer}.bias'])+1e-12))}
 correlations={}
 for name,mask in phase.items():
  a=prev['h2'][mask]-prev['h2'][mask].mean(0);b=current['h2'][mask]-current['h2'][mask].mean(0)
  corr=(a*b).sum(0)/np.maximum(np.sqrt((a*a).sum(0)*(b*b).sum(0)),1e-12)
  correlations[name]=corr
  drift[name]={'action_agreement':float((prev['scores'][mask].argmax(1)==current['scores'][mask].argmax(1)).mean()),'unit_correlation_quantiles':np.quantile(corr,[0,.25,.5,.75,1]).tolist(),'unit_correlations':corr.tolist()}
 audit['drift'][label]=drift

# Descriptive ranking: an earlier unit with stronger negative outgoing feedback,
# correlated identity, and little incoming feedback change is a candidate causal
# block. This does not choose a gameplay winner or change any weights.
audit['recovery_candidates']={}
for label in ['12','18','24']:
 rows=[]
 for j in range(128):
  old=audit['checkpoints'][label]['units'][j];new=audit['checkpoints']['36']['units'][j]
  ofb=old['outgoing']['own_derivative_mean'];nfb=new['outgoing']['own_derivative_mean']
  gain=nfb-ofb
  incoming_delta=float(abs(values[label]['own'][phase['incoming'],j]-current['own'][phase['incoming'],j]).mean())
  corr=audit['drift'][label]['outgoing']['unit_correlations'][j]
  evidence=gain/(incoming_delta+.01) if gain>0 and ofb<0 and corr>.9 else 0
  rows.append(dict(unit=j,old_outgoing_feedback=ofb,new_outgoing_feedback=nfb,outgoing_feedback_reduction=gain,incoming_feedback_delta=incoming_delta,outgoing_activation_correlation=corr,evidence_score=evidence,old_outgoing_ablation=old['outgoing']['ablation_action_change'],new_outgoing_ablation=new['outgoing']['ablation_action_change'],old_incoming_ablation=old['incoming']['ablation_action_change'],new_incoming_ablation=new['incoming']['ablation_action_change']))
 rows.sort(key=lambda r:r['evidence_score'],reverse=True);audit['recovery_candidates'][label]=rows

(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
np.savez(OUT/'activations.npz',**{name+'_'+key:v[key] for name,v in values.items() for key in ['h1','h2','scores','own','ball','vy','other']})
for label in nets:print('WHOLE',label,audit['checkpoints'][label]['whole_phase'])
for label in ['12','18','24']:
 print('DRIFT',label,{k:v for k,v in audit['drift'][label].items() if k not in phase})
 print('RECOVERY',label,audit['recovery_candidates'][label][:6])
