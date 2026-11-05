"""One mathematically exact cap13-flight product lattice, ordinary dense only.

A uniform square grid Δ=2/9 has identical interpolation error at u+vy/9 and
u−vy/9 for every integervy, because their separation is an integer grid period.
Difference-of-squares therefore gives the exact product on the active domain.
Combined with8/13ceiloffset, the continuous affine forecast equals integer
landing flight before reflection at every physically generated cap13state.
Lower horizontal speeds remain a cap-speed approximation. No fitting.
"""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'));sys.path.insert(0,'/tmp/rally-analytic-dense')
from build import export,OUT
import numpy as np,torch,onnx
from torch import nn
from competition_eval import load_mlp
from pace_study import CONFIGS,make_policy,run_pair
from policy import ROOT,load_builtin

def first():
 knots=np.arange(-18,17,2)/9;K=len(knots);B=2*K
 W=np.zeros((B+8,6),np.float32);bias=np.zeros(B+8,np.float32)
 for i,k in enumerate(knots):
  W[i,0]=1;W[i,3]=2/3;bias[i]=-k
  W[i+K,0]=1;W[i+K,3]=-2/3;bias[i+K]=-k
 W[B,1]=1;bias[B]=1;W[B+1,3]=1;bias[B+1]=2;W[B+2,4]=1;bias[B+2]=1
 W[B+3,2]=1;W[B+4,2]=-2;bias[B+4]=1;W[B+5,2]=-2
 W[B+6,5]=10;bias[B+6]=1;W[B+7,5]=10;bias[B+7]=-1
 coeff=np.r_[-17/18,np.full(K-1,1/9)];C=(6*180)/(13*80)*1.5;vyco=6*(165/13+8/13)/80
 z=np.zeros(B+8,np.float32);z[:K]=C*coeff;z[K:B]=-C*coeff;z[B]=1;z[B+1]=vyco
 return W,bias,z,-1-2*vyco,B

def make(full_risk=False):
 W1,B1,z,zb,B=first();N=len(B1)
 if not full_risk:
  W2=np.zeros((8,N),np.float32);B2=np.zeros(8,np.float32)
  for i,k in enumerate([-4.75,-2.85,-.95,.95,2.85]):W2[i]=z;W2[i,B+3]-=20;B2[i]=zb-k
  W2[5,B+4]=1;W2[5,B+5]=-1;W2[6,B+2]=1;W2[7,B+6]=1;W2[7,B+7]=-1;W2[7,B+3]=-20
  aim=6/80;down=np.r_[[1,-2,2,-2,2],-.95-aim,-1,aim]
 else:
  prior=json.loads((OUT/'full-risk-feasibility.json').read_text());h=np.array(prior['raw_knots']);v=np.array(prior['target_values']);sl=np.diff(v)/np.diff(h);co=np.r_[sl[0],np.diff(sl)];used=np.flatnonzero(abs(co)>1e-12);K=len(used)
  N-=2;W1=W1[:N];B1=B1[:N];z=z[:N]
  W2=np.zeros((K+2,N),np.float32);B2=np.zeros(K+2,np.float32)
  for j,i in enumerate(used):W2[j]=z;W2[j,B+3]-=20;B2[j]=zb-(h[i]-80)/80
  W2[K,B+4]=1;W2[K,B+5]=-1;W2[K+1,B+2]=1
  down=np.r_[co[used],(v[0]-80)/80,-1]
 W3=np.stack([-down,np.zeros(len(down)),down]).astype(np.float32);B3=np.array([-1,1.101/80,1],np.float32)
 net=nn.Sequential(nn.Linear(6,N),nn.ReLU(),nn.Linear(N,len(B2)),nn.ReLU(),nn.Linear(len(B2),3)).eval()
 with torch.no_grad():
  for layer,w,b in zip([net[0],net[2],net[4]],[W1,W2,W3],[B1,B2,B3]):layer.weight.copy_(torch.from_numpy(w));layer.bias.copy_(torch.from_numpy(b))
 return net
if __name__=='__main__':
 torch.set_num_threads(1);seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds'];opponents={k:load_builtin(k) for k in ['tracker','anticipator','challenger']};opponents['pacer']=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');opponents.update({k:make_policy(CONFIGS['pace13-v9-scaled'],k) for k in ['attack','defend']})
 record=dict(purpose='One exact-cap-flight product lattice applied to two declaredtargetcontrols; no fitting/sweeps',seeds=seeds,candidates={},pairs={})
 for name,flag in [('small-cap-global',False),('standard-full-risk-cap-global',True)]:
  net=make(flag);torch.save(net.state_dict(),OUT/f'{name}.pt');export(net,OUT/f'{name}.onnx');widths=[6,net[0].out_features,net[2].out_features,3];params=sum(p.numel() for p in net.parameters());macs=sum(a*b for a,b in zip(widths[:-1],widths[1:]));record['candidates'][name]=dict(widths=widths,params=params,macs=macs)
  p=OUT/f'{name}.onnx';m=onnx.load(p);onnx.helper.set_model_props(m,dict(rules='pong-v4',architecture='-'.join(map(str,widths))+' ReLU',method='Transparent analytic dense exactcapflightproductlattice; no training',target='Full cap13 physicalcontactrisk prior' if flag else '6unitopponentsideaim',forecast='Cap13ceiloffset8/13;uniformsquaregridΔ2/9cancelsdifferenceerrorsforintegerphysicalvy',limitation='Lower speeds remainapproximate; center80outgoingrecovery'));onnx.save(m,p)
  for other,op in opponents.items():
   row=run_pair(CONFIGS['pace13-v9-scaled'],net,op,seeds);record['pairs'][name+':'+other]=row
   (OUT/'cap-global-results.json').write_text(json.dumps(record,separators=(',',':'))+'\n');s=row['summary'];print(name,other,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
  print('size',name,record['candidates'][name],flush=True)
