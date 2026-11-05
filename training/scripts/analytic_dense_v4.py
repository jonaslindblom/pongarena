"""Reproduce Prism Nano: transparent handcrafted dense weights, no training.

The fixed network has six raw observations, two ReLU hidden layers and three
linear action scores. This builder writes only dense matrices and biases.
Forecasting is an approximate prior encoded in those weights, not a runtime
trajectory helper. See training/results/analytic-dense-v4/README.md for limits.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from onnx import helper, numpy_helper


SHA256 = '1eedf32a2fff0721aa101d4ba8a925a6fac7734f259daa9180ecb6523f65991a'


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
 # Conservative6court-unit contact aim; deadband1.101units.
 aim=6/80
 down=np.r_[np.array([1,-2,2,-2,2]),-.95-aim,-1,1,aim].astype(np.float32)
 W3=np.stack([-down,np.zeros(9),down]).astype(np.float32)
 B3=np.array([0,1.101/80,0],np.float32)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh path; preserve frozen model evidence.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export(make(), args.out)
    model = onnx.load(args.out)
    props = {p.key: p.value for p in model.metadata_props}
    props['candidate'] = 'stable-cap13-safeaim'
    props['deadband'] = '1.101courtunits numericguard'
    props['target'] = 'Only deadband1.1→1.101courtunits, numerical boundary guard'
    onnx.helper.set_model_props(model, props)
    onnx.save(model, args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    if digest != SHA256:
        raise AssertionError(f'Frozen reproduction differs: {digest}')
    print(f'{args.out}: {digest}')


if __name__ == '__main__':
    main()
