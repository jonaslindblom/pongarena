"""Exact first-layer redistribution for the frozen robust branch.

Returnsordinarydensefirst-layer arrays plusincoming/outgoing affine forecast
readouts. No game observations, fitting, or matches are used in construction.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'training/scripts'));sys.path.insert(0,'/tmp/rally-analytic-dense')
from cap_global import first as original_first
import numpy as np


def first():
 W,b,q,zb,B=original_first();W=W[:42].astype(float);b=b[:42].astype(float);q=q[:42].astype(float)
 qout=q.copy();qout[:36]*=-1
 forecasts=[]
 keep=[i for i in range(42)if i not in [0,18,40,41]]
 for readout in [q,qout]:
  assert abs(readout[0]+readout[18])<1e-6
  adjustment=readout[0]*4/3
  readout[37]+=adjustment;bias=float(zb-2*adjustment)
  forecasts.append((np.r_[readout[keep],0].astype(np.float32),bias))
 W=W[keep];b=b[keep];negative_vx=np.zeros((1,6));negative_vx[0,2]=-1
 W=np.r_[W,negative_vx].astype(np.float32);b=np.r_[b,0].astype(np.float32)
 return dict(weight=W,bias=b,incoming=forecasts[0][0],incoming_bias=forecasts[0][1],outgoing=forecasts[1][0],outgoing_bias=forecasts[1][1],indices=dict(y=34,vy=35,own=36,positive_vx=37,negative_vx=38),removed_original_indices=[0,18,40,41])


if __name__=='__main__':
 import json,torch
 from torch import nn
 r=first();W,B,z,zb,offset=original_first();p=np.load('/tmp/rally-causal-failures/observations.npy')
 h=np.maximum(0,p@r['weight'].T+r['bias']);original_h=np.maximum(0,p@W.T+B)
 qo=z.copy();qo[:offset]*=-1
 print(json.dumps(dict(architecture=[6,len(r['bias'])],indices=r['indices'],incoming_max_error=float(abs(h@r['incoming']+r['incoming_bias']-(original_h@z+zb)).max()),outgoing_max_error=float(abs(h@r['outgoing']+r['outgoing_bias']-(original_h@qo+zb)).max()))),flush=True)
