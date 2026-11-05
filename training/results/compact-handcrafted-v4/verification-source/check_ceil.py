"""Exact ceil on reachable integer-distance/integer-speed ratios, no fitting."""
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
import onnxruntime as ort
OUT=Path('/tmp/rally-compact-audit')
weights=[numpy_helper.from_array(-(np.arange(1,66,dtype=np.float32)+np.float32(1/64))[:,None],'lower'),
         numpy_helper.from_array(-(np.arange(1,66,dtype=np.float32)+np.float32(1/32))[:,None],'upper'),
         numpy_helper.from_array(np.ones((1,1),np.float32),'one'),
         numpy_helper.from_array(np.ones((65,1),np.float32),'sum')]
nodes=[helper.make_node('Add',['ratio','lower'],['raw_lo']),helper.make_node('Relu',['raw_lo'],['lo']),
       helper.make_node('Add',['ratio','upper'],['raw_hi']),helper.make_node('Relu',['raw_hi'],['hi']),
       helper.make_node('Gemm',['hi','one'],['negative_hi'],alpha=-1.),
       helper.make_node('Add',['lo','negative_hi'],['difference']),
       helper.make_node('Gemm',['difference','sum','one'],['ceil'],transA=1,alpha=64.)]
graph=helper.make_graph(nodes,'Compact exact integer-grid ceiling',[helper.make_tensor_value_info('ratio',1,[1,1])],[helper.make_tensor_value_info('ceil',1,[1,1])],weights)
model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=10);onnx.checker.check_model(model);onnx.save(model,OUT/'ceil.onnx')
opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
session=ort.InferenceSession(str(OUT/'ceil.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
records=[];maxerr=0.;failures=[]
for speed in range(5,14):
 for distance in range(331):
  for kind in ['ratio','observation_arithmetic']:
   d=np.float32(distance) if kind=='ratio' else (np.float32((distance-165)/180)*np.float32(180)+np.float32(165))
   ratio=np.float32(d*np.float32(1/speed))
   actual=float(session.run(None,{'ratio':np.array([[ratio]],np.float32)})[0][0,0]);expected=max(1,int(np.ceil(distance/speed)))
   err=abs(actual-expected);maxerr=max(maxerr,err)
   if err:failures.append(dict(distance=distance,speed=speed,ratio=float(ratio),actual=actual,expected=expected,kind=kind))
   records.append([distance,speed,kind])
result=dict(cases=len(records),failures=failures,max_error_ticks=maxerr,initializer_scalars=sum(np.prod(w.dims).item() for w in weights),macs=130,nodes=len(nodes),method='Column hinge vectors, binary thresholds1/64and1/32; difference before reduction; sharedone')
(OUT/'ceil-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert not failures
