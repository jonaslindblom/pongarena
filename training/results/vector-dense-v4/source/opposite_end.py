"""One physical dense opponent-aware contact rule; no fitting or reward search.

At cap13: flight26, vy=round(3d/5). Aim the continuous outgoing ray at
courtY4 or156, opposite the observed opponent paddle. Therefore
p=(83/78)*h-(5/78)*D. Angle/contact quantization remains an approximation.
Inference is exactly Gemm/ReLU/Gemm/ReLU/Gemm, with only six raw inputs.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path.cwd();OUT=Path('/tmp/rally-offense-features')
sys.path.insert(0,str(ROOT/'training/scripts'));sys.path.insert(0,'/tmp/rally-analytic-dense')
import numpy as np,torch,onnx
from torch import nn
from onnx import helper,numpy_helper
from cap_global import first
from pace_study import CONFIGS,make_policy,run_pair
from competition_eval import load_mlp
RULES=CONFIGS['pace13-v9-scaled']


def make():
 W1,B1,z,zb,B=first();assert len(B1)==44
 # Two plateau cues implementS=-1 at integer opponentY<=79, +1 at>=80.
 W1[B+6]=0;W1[B+7]=0;W1[B+6,5]=160;W1[B+7,5]=160;B1[B+6]=2;B1[B+7]=0
 W2=np.zeros((9,44),np.float32);B2=np.zeros(9,np.float32)
 for i,k in enumerate([-4.75,-2.85,-.95,.95,2.85]):W2[i]=z;W2[i,B+3]-=20;B2[i]=zb-k
 W2[5,B+4]=1;W2[5,B+5]=-1
 W2[6,B+2]=1
 W2[7,B+6]=1;W2[7,B+3]=-256
 W2[8,B+7]=1;W2[8,B+3]=-256
 A=83/78;C=19/312
 down=np.r_[A*np.array([1,-2,2,-2,2]),-.95*A-C,-1,C,-C]
 W3=np.stack([-down,np.zeros(9),down]).astype(np.float32);B3=np.array([-1,1.101/80,1],np.float32)
 net=nn.Sequential(nn.Linear(6,44),nn.ReLU(),nn.Linear(44,9),nn.ReLU(),nn.Linear(9,3)).eval()
 with torch.no_grad():
  for layer,w,b in zip([net[0],net[2],net[4]],[W1,W2,W3],[B1,B2,B3]):layer.weight.copy_(torch.from_numpy(w));layer.bias.copy_(torch.from_numpy(b))
 return net


def export(net,path):
 nodes=[];weights=[];source='observation'
 for i,layer in enumerate(net):
  output='action_scores'if i==len(net)-1 else f'h{i}'
  if isinstance(layer,nn.Linear):
   weights.extend([numpy_helper.from_array(layer.weight.detach().numpy(),f'w{i}'),numpy_helper.from_array(layer.bias.detach().numpy(),f'b{i}')]);nodes.append(helper.make_node('Gemm',[source,f'w{i}',f'b{i}'],[output],transB=1))
  else:nodes.append(helper.make_node('Relu',[source],[output]))
  source=output
 model=helper.make_model(helper.make_graph(nodes,'Dense physical opposite-end aim',[helper.make_tensor_value_info('observation',1,[1,6])],[helper.make_tensor_value_info('action_scores',1,[1,3])],weights),opset_imports=[helper.make_opsetid('',18)],ir_version=10)
 helper.set_model_props(model,dict(rules='pong-v4',architecture='6-44-9-3 ReLU',method='One closed-form opposite-opponent court-end aim; no fitting/training',formula='p=(83/78)*landing-(5/78)*D; D4ifopponentY>=80else 156',limitations='cap13time;continuouscontactgoal;integerangle/parityapproximation;outgoingcenter80'))
 onnx.checker.check_model(model);onnx.save(model,path)


def physical_audit(net):
 from env import reflect
 obs=[];truth=[]
 for h in range(4,157):
  for opponent in [15,71,79,80,89,145]:
   for own in [15,40,80,120,145]:
    obs.append([(20-180)/180,(h-80)/80,-13/6,0,(own-80)/80,(opponent-80)/80]);D=4 if opponent>=80 else 156;truth.append((83/78*h-5/78*D-own)/80)
 obs=np.array(obs,np.float32);truth=np.array(truth)
 with torch.no_grad():actual=net(torch.from_numpy(obs)).numpy()
 expected=np.stack([-truth,np.full(len(truth),1.101/80),truth],1)
 np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=0)
 return dict(probes=len(obs),max_score_error=float(np.max(abs(actual-expected))),action_mismatches=int((actual.argmax(1)!=expected.argmax(1)).sum()))


def main():
 torch.set_num_threads(1);net=make();p=OUT/'opposite-end.pt';torch.save(net.state_dict(),p);export(net,OUT/'opposite-end.onnx');checks=physical_audit(net)
 seeds=json.load(open('/tmp/rally-causal-failures/seeds.json'))['seeds'];opponents={k:make_policy(RULES,k) for k in ['attack','defend']};opponents['pacer']=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');opponents['prism']=load_mlp('/tmp/rally-safe-branch/small-robust-branch.pt',activation='relu')
 result=dict(purpose='One physically derived opposite-end target control; no coefficient sweep, optimization, fitting or training',seeds=seeds,parameters=sum(x.numel() for x in net.parameters()),architecture=[6,44,9,3],nodes=5,checks=checks,pairs={})
 for name,opponent in opponents.items():
  row=run_pair(RULES,net,opponent,seeds);result['pairs'][name]=row;(OUT/'opposite-end-results.json').write_text(json.dumps(result,separators=(',',':'))+'\n');s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
 result['complete']=True;(OUT/'opposite-end-results.json').write_text(json.dumps(result,separators=(',',':'))+'\n');print('DONE',result['parameters'],checks,flush=True)

if __name__=='__main__':main()
