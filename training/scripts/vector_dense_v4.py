"""Reproduce Vector Nano: six raw inputs, ordinary dense ReLU weights only.

The fixed 487-parameter network represents a cap-contact anticipation prior
and opposite-opponent endpoint aiming. It has no fitted parameters, inference
helper, preprocessing or extra ONNX operations. See vector-dense-v4 results.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from analytic_dense_v4 import export

SHA256 = 'ad41b0b8b9be1264c7b3daea95aec2341ab14f4802038ef175aea5d4898ac551'


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


def make():
 W1,B1,z,zb=first();B=20;assert len(B1)==28
 # Two plateau cues implementS=-1 at integer opponentY<=79, +1 at>=80.
 W1[B+6]=0;W1[B+7]=0;W1[B+6,5]=160;W1[B+7,5]=160;B1[B+6]=2;B1[B+7]=0
 W2=np.zeros((9,28),np.float32);B2=np.zeros(9,np.float32)
 for i,k in enumerate([-4.75,-2.85,-.95,.95,2.85]):W2[i]=z;W2[i,B+3]-=20;B2[i]=zb-k
 W2[5,B+4]=1;W2[5,B+5]=-1
 W2[6,B+2]=1
 W2[7,B+6]=1;W2[7,B+3]=-256
 W2[8,B+7]=1;W2[8,B+3]=-256
 A=83/78;C=19/312
 down=np.r_[A*np.array([1,-2,2,-2,2]),-.95*A-C,-1,C,-C]
 W3=np.stack([-down,np.zeros(9),down]).astype(np.float32);B3=np.array([-1,1.101/80,1],np.float32)
 net=nn.Sequential(nn.Linear(6,28),nn.ReLU(),nn.Linear(28,9),nn.ReLU(),nn.Linear(9,3)).eval()
 with torch.no_grad():
  for layer,w,b in zip([net[0],net[2],net[4]],[W1,W2,W3],[B1,B2,B3]):layer.weight.copy_(torch.from_numpy(w));layer.bias.copy_(torch.from_numpy(b))
 return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh path; preserve frozen model evidence.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export(make(), args.out)
    model = onnx.load(args.out)
    model.graph.name = 'Dense physical opposite-end aim'
    onnx.helper.set_model_props(model, dict(rules='pong-v4', architecture='6-28-9-3 ReLU',
        method='One closed-form opposite-opponent court-end aim; no fitting/training',
        formula='p=(83/78)*landing-(5/78)*D; D4ifopponentY>=80else 156',
        limitations='cap13time;continuouscontactgoal;integerangle/parityapproximation;outgoingcenter80'))
    onnx.save(model, args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    if digest != SHA256:
        raise AssertionError(f'Frozen reproduction differs: {digest}')
    print(f'{args.out}: {digest}')


if __name__ == '__main__':
    main()
