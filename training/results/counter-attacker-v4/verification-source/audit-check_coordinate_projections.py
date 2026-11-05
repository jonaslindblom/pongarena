"""Native proof probes: observation normalization inverts to integer features."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,onnx,onnxruntime as ort
from onnx import helper
from compact_handcrafted_v4 import Graph
OUT=Path('/tmp/rally-attacker-audit')
g=Graph();outputs=[g.project(i,s,b) for i,s,b in [(0,180,0),(1,80,80),(2,6,0),(3,6,0),(4,80,80),(5,80,80)]]
model=helper.make_model(helper.make_graph(g.nodes,'Coordinate reconstruction probes',[helper.make_tensor_value_info('observation',1,[1,6])],[helper.make_tensor_value_info(o,1,[1,1]) for o in outputs],g.weights),opset_imports=[helper.make_opsetid('',18)],ir_version=10);onnx.checker.check_model(model);onnx.save(model,OUT/'coordinate-probe.onnx')
opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
s=ort.InferenceSession(str(OUT/'coordinate-probe.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
counts=[];failures=[]
for column,values,center,scale in [(0,range(15,346),180,180),(1,range(4,157),80,80),(2,list(range(-13,-4))+list(range(5,14)),0,6),(3,range(-9,10),0,6),(4,range(15,146),80,80),(5,range(15,146),80,80)]:
 for value in values:
  obs=np.zeros((1,6),np.float32);obs[0,column]=(value-center)/scale
  out=float(s.run(None,{'observation':obs})[column][0,0]);expected=value-180 if column==0 else value
  if out!=expected:failures.append(dict(column=column,world=value,actual=out,expected=expected))
 counts.append([column,len(values)])
result=dict(component_probes=counts,total=sum(v for _,v in counts),failures=failures,all_reconstructions_exact=not failures)
(OUT/'coordinate-projection-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert not failures
