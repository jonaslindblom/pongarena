"""One openly analytic ordinary dense ReLU policy, no fitting or extra graph ops.

Features approximate cap-speed straight-flight anticipation, reflection, phase,
feedback and conservative opponent-side contact aiming. This is not a learned
source feature discovery. Only raw six observations enter the exported network.
"""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch,onnx
from torch import nn
from onnx import helper,numpy_helper
from policy import ROOT
from competition_eval import load_mlp
from pace_study import CONFIGS,make_policy,run_pair
from policy import load_builtin

OUT=Path('/tmp/rally-analytic-dense');GROUPS={}

def make():
 net=nn.Sequential(nn.Linear(6,29),nn.ReLU(),nn.Linear(29,9),nn.ReLU(),nn.Linear(9,3)).eval()
 W1=np.zeros((29,6),np.float32);B1=np.zeros(29,np.float32)
 # u=xnorm, v=rawvy/9=(2/3)*obsvy. Nine hinges per square term.
 # q(u+v)-q(u-v) gives a coarse bilinear anticipation feature using a
 # fixed piecewise-square prior; every operation is an ordinary hidden unit.
 knots=np.linspace(-1.6,1.6,9)
 for i,k in enumerate(knots):
  W1[i,0]=1;W1[i,3]=2/3;B1[i]=-k
  W1[i+9,0]=1;W1[i+9,3]=-2/3;B1[i+9]=-k
 # Signed linear carries through the first ReLU: y,vy,ownY.
 for i,feature in enumerate([1,3,4]):
  W1[18+2*i,feature]=1;W1[19+2*i,feature]=-1
 W1[24,2]=1 # positive vx phase-sensitive feature
 W1[25,2]=-2;B1[25]=1
 W1[26,2]=-2 # difference of units25/26 is1 incoming,0 outgoing
 W1[27,5]=10;B1[27]=1
 W1[28,5]=10;B1[28]=-1 # clipped opponent-side feature
 # Normalized L: y +vy*(distance/13 +1/2), centered at80.
 # uv=(q(u+v)-q(u-v))/4. With square knot step.4, the linear
 # part contributes−1.8*v and each ridge difference contributes+.2.
 C=(6*180)/(13*80)*1.5
 z=np.zeros(29,np.float32);z[:9]=C*.2;z[9:18]=-C*.2
 z[18]=1;z[19]=-1
 linear_vy=6*(165/13+.5)/80-C*1.8*(2/3)
 z[20]=linear_vy;z[21]=-linear_vy
 W2=np.zeros((9,29),np.float32);B2=np.zeros(9,np.float32)
 fold_knots=np.array([-4.75,-2.85,-.95,.95,2.85])
 for i,k in enumerate(fold_knots):
  W2[i]=z;W2[i,24]-=20;B2[i]=-k
 W2[5,25]=1;W2[5,26]=-1 # incoming indicator carry
 W2[6,22]=1;W2[7,23]=1 # ownY positive/negative carries
 W2[8,27]=1;W2[8,28]=-1;W2[8,24]=-20 # incoming-only opponent cue
 # Conservative6court-unit contact aim; deadband1.1units.
 aim=6/80
 down=np.r_[np.array([1,-2,2,-2,2]),-.95-aim,-1,1,aim].astype(np.float32)
 W3=np.stack([-down,np.zeros(9),down]).astype(np.float32)
 B3=np.array([0,1.1/80,0],np.float32)
 with torch.no_grad():
  for layer,w,b in zip([net[0],net[2],net[4]],[W1,W2,W3],[B1,B2,B3]):
   layer.weight.copy_(torch.from_numpy(w));layer.bias.copy_(torch.from_numpy(b))
 return net

def export(net,path):
 nodes=[];weights=[];current='observation'
 for i,layer in enumerate(net):
  output='action_scores' if i==len(net)-1 else f'h{i}'
  if isinstance(layer,nn.Linear):
   weights.extend([numpy_helper.from_array(layer.weight.detach().numpy(),f'w{i}'),numpy_helper.from_array(layer.bias.detach().numpy(),f'b{i}')])
   nodes.append(helper.make_node('Gemm',[current,f'w{i}',f'b{i}'],[output],transB=1))
  else:nodes.append(helper.make_node('Relu',[current],[output]))
  current=output
 m=helper.make_model(helper.make_graph(nodes,'Analytic dense cap-speed approximation',[helper.make_tensor_value_info('observation',1,[1,6])],[helper.make_tensor_value_info('action_scores',1,[1,3])],weights),opset_imports=[helper.make_opsetid('',18)],ir_version=10)
 helper.set_model_props(m,dict(rules='pong-v4',architecture='6-29-9-3 ReLU',method='Transparent analytic dense feature approximation; no training',limitation='Uses cap13 flight approximation and6unit opponent-side aim; not exact physics or learned-source transfer'))
 onnx.checker.check_model(m);onnx.save(m,path)

if __name__=='__main__':
 torch.set_num_threads(1);net=make();torch.save(net.state_dict(),OUT/'analytic-cap13-safeaim.pt');export(net,OUT/'analytic-cap13-safeaim.onnx')
 seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds']
 policies={k:load_builtin(k) for k in ['tracker','anticipator','challenger']}
 policies['pacer']=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt')
 policies.update({k:make_policy(CONFIGS['pace13-v9-scaled'],k) for k in ['attack','defend']})
 result=dict(purpose='One declared genuine dense analytic approximation; no fitting or coefficient sweeps',architecture=[6,29,9,3],parameters=sum(x.numel() for x in net.parameters()),macs=6*29+29*9+9*3,seeds=seeds,features=dict(first='18square-ridge,6linearcarry,1outgoingvx,2incomingphase,2opponentcue',second='5reflectedlanding,1incomingphase,2owncarry,1incomingopponentcue'),pairs={})
 for name,opponent in policies.items():
  row=run_pair(CONFIGS['pace13-v9-scaled'],net,opponent,seeds);result['pairs'][name]=row
  (OUT/'results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
  s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
 print('DONE',result['parameters'],result['macs'],flush=True)
