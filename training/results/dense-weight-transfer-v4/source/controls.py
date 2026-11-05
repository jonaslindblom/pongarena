"""Two additional algebraic controls requested after initial transfer failures."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from torch import nn
from competition_eval import load_mlp
from pace_study import CONFIGS,run_pair,make_policy
from policy import ROOT,export,load_builtin
OUT=Path('/tmp/rally-dense-transfer');torch.set_num_threads(1)
net=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt')
s=net.state_dict();w0,b0,w2,b2,w4,b4=[s[k].numpy().copy() for k in ['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']]
a0=np.tanh(b0);z0=w2@a0+b2;h0=np.tanh(z0);d=1-h0*h0
v=(w4*d[None,:])@w2
c=b4+w4@(h0-d*(w2@a0))
contrast=v-v.mean(0,keepdims=True);rank=np.linalg.norm(contrast,axis=0)
analysis=json.loads((OUT/'analysis.json').read_text());obs=np.load(OUT/'observations.npy');seeds=json.loads((OUT/'seed-audit.json').read_text())['seeds']
with torch.no_grad():source_scores=net(torch.from_numpy(obs)).numpy()
models={};report={'purpose':'Copied-bias deletion and source-only second-layer Taylor controls; no parameter fitting','seeds':seeds,'source_states':len(obs),'source_neutral_second_tanh_derivative_quantiles':np.quantile(d,[0,.25,.5,.75,1]).tolist(),'candidates':{},'pairs':{}}
for k in [16,32]:
 for mode in ['copy-bias','taylor']:
  if mode=='copy-bias':
   i=np.array(analysis['candidates'][f'prune-{k}']['first']);j=np.array(analysis['candidates'][f'prune-{k}']['second'])
   values=[w0[i],b0[i],w2[np.ix_(j,i)],b2[j],w4[:,j],b4]
   details=dict(first=i.tolist(),second=j.tolist(),method='Same channel masks as first pruning trial, all retained source biases copied unchanged')
  else:
   i=np.argsort(rank)[-k:];removed=np.setdiff1d(np.arange(128),i)
   bias=c+v[:,removed]@a0[removed]
   values=[w0[i],b0[i],np.eye(k,dtype=np.float32)*.01,np.zeros(k,np.float32),v[:,i]/.01,bias]
   details=dict(first=i.tolist(),method='SecondTanh first-order expansion at source neutral activations; action-contrast rank; no fitted labels or activations',linearization_scale=.01)
  model=nn.Sequential(nn.Linear(6,k),nn.Tanh(),nn.Linear(k,k),nn.Tanh(),nn.Linear(k,3)).eval()
  with torch.no_grad():
   for layer,w,b in zip([model[0],model[2],model[4]],values[::2],values[1::2]):
    layer.weight.copy_(torch.from_numpy(w));layer.bias.copy_(torch.from_numpy(b))
   predicted=model(torch.from_numpy(obs)).numpy()
  label=f'{mode}-{k}';models[label]=model
  torch.save(model.state_dict(),OUT/f'{label}.pt');export(model,OUT/f'{label}.onnx')
  details.update(params=sum(p.numel() for p in model.parameters()),action_agreement=float((predicted.argmax(1)==source_scores.argmax(1)).mean()),mean_score_error=float(abs(predicted-source_scores).mean()))
  report['candidates'][label]=details;print('PROBE',label,details['action_agreement'],details['mean_score_error'],flush=True)
ops={'tracker':load_builtin('tracker'),'anticipator':load_builtin('anticipator'),'challenger':load_builtin('challenger'),'pacer':net,'attack':make_policy(CONFIGS['pace13-v9-scaled'],'attack')}
for label,model in models.items():
 for name,op in ops.items():
  row=run_pair(CONFIGS['pace13-v9-scaled'],model,op,seeds);report['pairs'][label+':'+name]=row
  (OUT/'controls.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
  sm=row['summary'];print(label,name,sm['wins'],sm['losses'],sm['draws'],sm['point_margin_ci95'],flush=True)
report['complete']=True;(OUT/'controls.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
