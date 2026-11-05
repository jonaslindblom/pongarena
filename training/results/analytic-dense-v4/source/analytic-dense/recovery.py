"""One Small dense prior: conservative edge-sensitive outgoing recovery."""
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

def make_recovery():
 source=candidates()['stable-cap13-safeaim']
 net=nn.Sequential(nn.Linear(6,29),nn.ReLU(),nn.Linear(29,21),nn.ReLU(),nn.Linear(21,3)).eval()
 q=np.zeros(29,np.float32);C=(6*180)/(13*80)*1.5
 q[:9]=-C*.2;q[9:18]=C*.2;q[18]=1;q[19]=-1
 linear=6*(165/13+.5)/80+C*1.8*(2/3);q[20]=linear;q[21]=-linear
 raw_knots=np.array([-164,-148,-132,-12,4,20,140,156,172,292,308,324])
 co=np.array([-1.25,2.5,-1.25,1.25,-2.5,1.25,-1.25,2.5,-1.25,1.25,-2.5,1.25])
 with torch.no_grad():
  net[0].load_state_dict(source[0].state_dict())
  net[2].weight.zero_();net[2].bias.zero_();net[4].weight.zero_();net[4].bias.copy_(source[4].bias)
  net[2].weight[:9].copy_(source[2].weight);net[2].bias[:9].copy_(source[2].bias);net[4].weight[:,:9].copy_(source[4].weight)
  for j,k in enumerate(raw_knots):
   w=q.copy();w[25]-=20;w[26]+=20
   net[2].weight[9+j].copy_(torch.from_numpy(w));net[2].bias[9+j]=-(k-80)/80
   net[4].weight[2,9+j]=co[j];net[4].weight[0,9+j]=-co[j]
 return net
if __name__=='__main__':
 torch.set_num_threads(1);net=make_recovery();torch.save(net.state_dict(),OUT/'small-edge-recovery.pt');export(net,OUT/'small-edge-recovery.onnx')
 p=OUT/'small-edge-recovery.onnx';m=onnx.load(p);onnx.helper.set_model_props(m,dict(rules='pong-v4',architecture='6-29-21-3 ReLU',method='Transparent analytic dense cap-speed forecast and conservative edge recovery; no training',target='Incoming6unit opponent-side aim; outgoingrecovery100at4,80from20to140,60at156,composedwithreflection',limitation='Cap13 forecast misses slowerflight; conservative recoverycurve ignoresquantizedstagegeometry'));onnx.save(m,p)
 seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds'];opponents={k:load_builtin(k) for k in ['tracker','anticipator','challenger']};opponents['pacer']=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');opponents.update({k:make_policy(CONFIGS['pace13-v9-scaled'],k) for k in ['attack','defend']})
 result=dict(purpose='One declared Small dense recovery approximation; no fitting or sweeps',architecture=[6,29,21,3],parameters=sum(x.numel() for x in net.parameters()),macs=6*29+29*21+21*3,seeds=seeds,pairs={})
 for name,other in opponents.items():
  row=run_pair(CONFIGS['pace13-v9-scaled'],net,other,seeds);result['pairs'][name]=row
  (OUT/'recovery-results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
  s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
 print('Done',result['parameters'],result['macs'])
