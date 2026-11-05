"""One declared Small dense coarse physical contact-risk prior, no fitting."""
from pathlib import Path
import sys,json
sys.path.insert(0,'/tmp/rally-analytic-dense')
from farcorner import candidates
from build import export,OUT
import torch,numpy as np,onnx
from torch import nn
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
from competition_eval import load_mlp
from pace_study import CONFIGS,make_policy,run_pair
from policy import ROOT,load_builtin

def make_risk():
 source=candidates()['stable-cap13-safeaim'];prior=json.loads((OUT/'risk-feasibility.json').read_text())
 heights=np.array(prior['raw_knots']);values=np.array(prior['target_values'])
 slopes=np.diff(values)/np.diff(heights);co=np.r_[slopes[0],np.diff(slopes)];used=np.flatnonzero(abs(co)>1e-12);K=len(used)
 net=nn.Sequential(nn.Linear(6,27),nn.ReLU(),nn.Linear(27,K+3),nn.ReLU(),nn.Linear(K+3,3)).eval()
 q=source[2].weight[0].detach().numpy()[:27].copy();q[24]+=20
 with torch.no_grad():
  net[0].weight.copy_(source[0].weight[:27]);net[0].bias.copy_(source[0].bias[:27]);net[2].weight.zero_();net[2].bias.zero_();net[4].weight.zero_();net[4].bias.zero_();net[4].bias[1]=1.101/80
  for j,i in enumerate(used):
   w=q.copy();w[24]-=20;net[2].weight[j].copy_(torch.from_numpy(w));net[2].bias[j]=-(heights[i]-80)/80
   net[4].weight[2,j]=co[i];net[4].weight[0,j]=-co[i]
  net[2].weight[K,25]=1;net[2].weight[K,26]=-1
  base=(values[0]-80)/80;net[4].weight[2,K]=base;net[4].weight[0,K]=-base
  net[2].weight[K+1,22]=1;net[2].weight[K+2,23]=1
  net[4].weight[2,K+1]=-1;net[4].weight[2,K+2]=1;net[4].weight[0,K+1]=1;net[4].weight[0,K+2]=-1
 return net
if __name__=='__main__':
 torch.set_num_threads(1);net=make_risk();torch.save(net.state_dict(),OUT/'small-coarse-risk.pt');export(net,OUT/'small-coarse-risk.onnx')
 p=OUT/'small-coarse-risk.onnx';m=onnx.load(p);onnx.helper.set_model_props(m,dict(rules='pong-v4',architecture='6-27-58-3 ReLU',method='Transparent analytic dense cap-speed forecast and fixed12unit coarse physical target prior; no training',target='Incomingcoarsecap13minriskcontacttablecomposedwithreflection;outgoingcenter80',limitation='Coarseinterpolationlosesdiscreteangles;cap13forecastmisseslowerstages'));onnx.save(m,p)
 seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds'];opponents={k:load_builtin(k) for k in ['tracker','anticipator','challenger']};opponents['pacer']=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');opponents.update({k:make_policy(CONFIGS['pace13-v9-scaled'],k) for k in ['attack','defend']})
 result=dict(purpose='One fixed Small dense coarse physical contact-risk target approximation; no fitting/sweeps',architecture=[6,27,58,3],parameters=sum(x.numel() for x in net.parameters()),macs=6*27+27*58+58*3,seeds=seeds,pairs={})
 for name,other in opponents.items():
  row=run_pair(CONFIGS['pace13-v9-scaled'],net,other,seeds);result['pairs'][name]=row
  (OUT/'risk-results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
  s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
 print('Done',result['parameters'],result['macs'])
