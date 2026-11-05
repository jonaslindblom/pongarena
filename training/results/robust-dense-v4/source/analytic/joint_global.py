"""One joint Standard control: exact-cap incoming prior + conservative recovery."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'));sys.path.insert(0,'/tmp/rally-analytic-dense')
from cap_global import make,first
from build import export,OUT
import numpy as np,torch,onnx
from torch import nn
from competition_eval import load_mlp
from pace_study import CONFIGS,make_policy,run_pair
from policy import ROOT,load_builtin

def make_joint():
 source=make(True);N=source[0].out_features;K=source[2].out_features
 net=nn.Sequential(nn.Linear(6,N),nn.ReLU(),nn.Linear(N,K+12),nn.ReLU(),nn.Linear(K+12,3)).eval()
 W,B,z,zb,offset=first();z=z[:N];q=z.copy();q[:offset]*=-1
 knots=np.array([-164,-148,-132,-12,4,20,140,156,172,292,308,324]);co=np.array([-1.25,2.5,-1.25,1.25,-2.5,1.25,-1.25,2.5,-1.25,1.25,-2.5,1.25])
 with torch.no_grad():
  net[0].load_state_dict(source[0].state_dict());net[2].weight.zero_();net[2].bias.zero_();net[4].weight.zero_();net[4].bias.copy_(source[4].bias)
  net[2].weight[:K].copy_(source[2].weight);net[2].bias[:K].copy_(source[2].bias);net[4].weight[:,:K].copy_(source[4].weight)
  for j,k in enumerate(knots):
   w=q.copy();w[offset+4]-=20;w[offset+5]+=20
   net[2].weight[K+j].copy_(torch.from_numpy(w));net[2].bias[K+j]=zb-(k-80)/80
   net[4].weight[2,K+j]=co[j];net[4].weight[0,K+j]=-co[j]
 return net
if __name__=='__main__':
 torch.set_num_threads(1);net=make_joint();torch.save(net.state_dict(),OUT/'standard-joint-cap-global.pt');export(net,OUT/'standard-joint-cap-global.onnx')
 p=OUT/'standard-joint-cap-global.onnx';m=onnx.load(p);widths=[6,net[0].out_features,net[2].out_features,3];onnx.helper.set_model_props(m,dict(rules='pong-v4',architecture='-'.join(map(str,widths))+' ReLU',method='Transparent ordinarydense exactcapflightpriorpluscoarseedge-recovery;no training',target='Fullcap13incomingminriskcurve;outgoing100at4,80from20to140,60at156',limitation='Lowerflightstagesapproximate;targetcap13quantizationusedatallstages'));onnx.save(m,p)
 seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds'];opponents={k:load_builtin(k) for k in ['tracker','anticipator','challenger']};opponents['pacer']=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');opponents.update({k:make_policy(CONFIGS['pace13-v9-scaled'],k) for k in ['attack','defend']})
 result=dict(purpose='One predeclaredjoint physicalroleprior, notfitting or rewardselection',architecture=widths,parameters=sum(p.numel() for p in net.parameters()),macs=sum(a*b for a,b in zip(widths[:-1],widths[1:])),seeds=seeds,pairs={})
 for name,other in opponents.items():
  row=run_pair(CONFIGS['pace13-v9-scaled'],net,other,seeds);result['pairs'][name]=row
  (OUT/'joint-results.json').write_text(json.dumps(result,separators=(',',':'))+'\n');s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
 print('Done',widths,result['parameters'],result['macs'])
