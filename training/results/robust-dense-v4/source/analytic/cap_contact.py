"""Dense analytic cap13-contact exactness control, no fitted coefficients.

Cap13 incomingx=345−13n, hence ceil((x−15)/13)=(x−15)/13+8/13.
At final contactx=20, xnorm=−8/9 is a knot on the square grid step4/9.
Symmetric interpolation errors cancel in q(x+v)−q(x−v), so the bilinear
forecast is exact there, apart from float32rounding. Other flight phases and
lower speeds remain approximations. All inference is one ordinary dense MLP.
"""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
sys.path.insert(0,'/tmp/rally-analytic-dense')
from build import export,OUT
import numpy as np,torch,onnx
from torch import nn
from competition_eval import load_mlp
from pace_study import CONFIGS,make_policy,run_pair
from policy import ROOT,load_builtin


def first():
 W=np.zeros((28,6),np.float32);B=np.zeros(28,np.float32)
 knots=np.arange(-20,17,4)/9
 for i,k in enumerate(knots):
  W[i,0]=1;W[i,3]=2/3;B[i]=-k
  W[i+10,0]=1;W[i+10,3]=-2/3;B[i+10]=-k
 W[20,1]=1;B[20]=1
 W[21,3]=1;B[21]=2
 W[22,4]=1;B[22]=1
 W[23,2]=1
 W[24,2]=-2;B[24]=1
 W[25,2]=-2
 W[26,5]=10;B[26]=1;W[27,5]=10;B[27]=-1
 co=np.r_[-1,np.full(9,2/9)]
 C=(6*180)/(13*80)*1.5;tcoef=6*(165/13+8/13)/80
 z=np.zeros(28,np.float32);z[:10]=C*co;z[10:20]=-C*co;z[20]=1;z[21]=tcoef
 bias=-1-2*tcoef
 return W,B,z,bias


def make(full_risk=False):
 W1,B1,z,zb=first()
 if not full_risk:
  W2=np.zeros((8,28),np.float32);B2=np.zeros(8,np.float32)
  for i,k in enumerate([-4.75,-2.85,-.95,.95,2.85]):
   W2[i]=z;W2[i,23]-=20;B2[i]=zb-k
  W2[5,24]=1;W2[5,25]=-1;W2[6,22]=1
  W2[7,26]=1;W2[7,27]=-1;W2[7,23]=-20
  aim=6/80;down=np.r_[[1,-2,2,-2,2],-.95-aim,-1,aim]
  W3=np.stack([-down,np.zeros(8),down]).astype(np.float32);B3=np.array([-1,1.101/80,1],np.float32)
 else:
  prior=json.loads((OUT/'full-risk-feasibility.json').read_text());h=np.array(prior['raw_knots']);v=np.array(prior['target_values']);sl=np.diff(v)/np.diff(h);coeff=np.r_[sl[0],np.diff(sl)];use=np.flatnonzero(abs(coeff)>1e-12);K=len(use)
  W1=W1[:26];B1=B1[:26];z=z[:26]
  W2=np.zeros((K+2,26),np.float32);B2=np.zeros(K+2,np.float32)
  for j,i in enumerate(use):
   W2[j]=z;W2[j,23]-=20;B2[j]=zb-(h[i]-80)/80
  W2[K,24]=1;W2[K,25]=-1;W2[K+1,22]=1
  down=np.r_[coeff[use],(v[0]-80)/80,-1]
  W3=np.stack([-down,np.zeros(K+2),down]).astype(np.float32);B3=np.array([-1,1.101/80,1],np.float32)
 net=nn.Sequential(nn.Linear(6,W1.shape[0]),nn.ReLU(),nn.Linear(W1.shape[0],len(B2)),nn.ReLU(),nn.Linear(len(B2),3)).eval()
 with torch.no_grad():
  for layer,w,b in zip([net[0],net[2],net[4]],[W1,W2,W3],[B1,B2,B3]):
   layer.weight.copy_(torch.from_numpy(w));layer.bias.copy_(torch.from_numpy(b))
 return net

if __name__=='__main__':
 torch.set_num_threads(1);seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds'];opponents={k:load_builtin(k) for k in ['tracker','anticipator','challenger']};opponents['pacer']=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');opponents.update({k:make_policy(CONFIGS['pace13-v9-scaled'],k) for k in ['attack','defend']})
 record=dict(purpose='One cap-contact-exact lattice forecast applied to two predeclaredtargetcontrols; no fitting or sweeps',seeds=seeds,candidates={},pairs={})
 for name,flag in [('nano-cap-contact',False),('standard-full-risk-cap-contact',True)]:
  net=make(flag);torch.save(net.state_dict(),OUT/f'{name}.pt');export(net,OUT/f'{name}.onnx')
  widths=[6,net[0].out_features,net[2].out_features,3];params=sum(p.numel() for p in net.parameters());macs=sum(a*b for a,b in zip(widths[:-1],widths[1:]));record['candidates'][name]=dict(widths=widths,params=params,macs=macs)
  p=OUT/f'{name}.onnx';m=onnx.load(p);onnx.helper.set_model_props(m,dict(rules='pong-v4',architecture='-'.join(map(str,widths))+' ReLU',method='Transparent analytic dense capcontactexactforecast; no training',target='Full cap13 physicalcontactrisk prior' if flag else '6unitopponentsideaim',forecast='Cap13ceiloffset8/13;squaregridstep4/9alignedwithcontactxnorm−8/9;exactatcapcontactmodfloat32',limitation='Lower speeds andintermediatecapflight remainapproximate;constantcenter80outgoingrecovery'));onnx.save(m,p)
  for other,op in opponents.items():
   row=run_pair(CONFIGS['pace13-v9-scaled'],net,op,seeds);record['pairs'][name+':'+other]=row
   (OUT/'cap-contact-results.json').write_text(json.dumps(record,separators=(',',':'))+'\n')
   s=row['summary'];print(name,other,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
  print('size',name,record['candidates'][name],flush=True)
