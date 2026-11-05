"""Reproduce Prism Small: ordinary dense weights, no training or helper at inference.

Construction uses physical constraints offline. The resulting ONNX has only
three fixed Gemm layers and two ReLU activations, six raw inputs, three scores.
Its cap-speed forecast and connected contact branch are represented entirely
by those matrices. Lower speeds and future reachability remain limitations.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from analytic_dense_v4 import export

SHA256 = '64b6f14a604fd6d7ff72b91a2861b7a061f6568b6edde80d4697c2e2160e31ca'


def reflect(y):
    # Weight construction only; the exported network has no modulo operator.
    z = (y-4) % 304
    return 4+np.where(z <= 152, z, 304-z)


def first():
 knots=np.arange(-18,17,2)/9;K=len(knots);B=2*K
 W=np.zeros((B+6,6),np.float32);bias=np.zeros(B+6,np.float32)
 for i,k in enumerate(knots):
  W[i,0]=1;W[i,3]=2/3;bias[i]=-k
  W[i+K,0]=1;W[i+K,3]=-2/3;bias[i+K]=-k
 W[B,1]=1;bias[B]=1;W[B+1,3]=1;bias[B+1]=2;W[B+2,4]=1;bias[B+2]=1
 W[B+3,2]=1;W[B+4,2]=-2;bias[B+4]=1;W[B+5,2]=-2
 coeff=np.r_[-17/18,np.full(K-1,1/9)];C=(6*180)/(13*80)*1.5;vyco=6*(165/13+8/13)/80
 z=np.zeros(B+6,np.float32);z[:K]=C*coeff;z[K:B]=-C*coeff;z[B]=1;z[B+1]=vyco
 return W,bias,z,-1-2*vyco,B

HEIGHTS=np.array([4,50,52,78,82,108,110,156.])
TARGETS=np.array([15.5,61.5,65.5,91.5,68.5,94.5,98.5,144.5])

def goal(y):
 return np.interp(y,HEIGHTS,TARGETS)

def raw_curve():
 raw=[]
 for n in range(-3,4):
  for h in HEIGHTS:raw.extend([h+304*n,312-h+304*n])
 raw=np.unique(np.r_[-240,raw,400]);raw=raw[(raw>=-240)&(raw<=400)]
 values=goal(reflect(raw));slopes=np.diff(values)/np.diff(raw);co=np.r_[slopes[0],np.diff(slopes)]
 use=np.flatnonzero(abs(co)>1e-12)
 return raw,values,co,use

def make():
 W1,B1,z,zb,B=first();N=len(B1)
 raw,values,co,used=raw_curve();K=len(used)
 W2=np.zeros((K+2,N),np.float32);B2=np.zeros(K+2,np.float32)
 for j,i in enumerate(used):W2[j]=z;W2[j,B+3]-=20;B2[j]=zb-(raw[i]-80)/80
 W2[K,B+4]=1;W2[K,B+5]=-1;W2[K+1,B+2]=1
 down=np.r_[co[used],(values[0]-80)/80,-1]
 W3=np.stack([-down,np.zeros(K+2),down]).astype(np.float32);B3=np.array([-1,1.101/80,1],np.float32)
 net=nn.Sequential(nn.Linear(6,N),nn.ReLU(),nn.Linear(N,K+2),nn.ReLU(),nn.Linear(K+2,3)).eval()
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
    onnx.helper.set_model_props(model, dict(rules='pong-v4', architecture='6-42-33-3 ReLU',
        method='Analytic ordinarydense exactcapforecast and continuousparityrobustcontactbranch;no training',
        target='Connectedadjacent-safecontactfamily;outgoingcenter80',
        limitation='One-replyandcap13prior;lowerstagesandmovementreachabilityremainapproximate'))
    onnx.save(model, args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    if digest != SHA256:
        raise AssertionError(f'Frozen reproduction differs: {digest}')
    print(f'{args.out}: {digest}')


if __name__ == '__main__':
    main()
