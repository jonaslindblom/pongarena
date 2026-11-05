"""Native exhaustive parity upper/lower lattice probe for grid-fixed counter."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,onnx,onnxruntime as ort
from onnx import helper
from counter_attacker_v4 import CounterGraph
OUT=Path('/tmp/rally-attacker-audit')
g=CounterGraph();actor=g.project(4,80,80);rawlo=g.project(1,80,80);rawhi=g.project(0,80,80)
fp=g.affine(g.op('Tanh',g.add(g.scalar(actor,4096),-8192*(np.arange(8,73)-1/512))),np.ones((65,1)),39.5,alpha=.5)
parity=g.sub(actor,g.scalar(fp,2));z=g.scalar(g.sub(rawlo,parity),4096)
q=g.affine(g.op('Tanh',g.add(z,-8192*(np.arange(7,73)+1/512))),np.ones((66,1)),40,alpha=.5)
lo=g.op('Add',parity,g.scalar(q,2));upperbit=g.op('Tanh',g.scalar(g.sub(rawhi,lo),8192,-8192*(2-1/512)));hi=g.op('Add',lo,g.add(upperbit,[1]))
model=helper.make_model(helper.make_graph(g.nodes,'Grid lattice proof probes',[helper.make_tensor_value_info('observation',1,[1,6])],[helper.make_tensor_value_info(lo,1,[1,1]),helper.make_tensor_value_info(hi,1,[1,1])],g.weights),opset_imports=[helper.make_opsetid('',18)],ir_version=10);onnx.checker.check_model(model);onnx.save(model,OUT/'grid-lattice-probe.onnx')
opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1;s=ort.InferenceSession(str(OUT/'grid-lattice-probe.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
count=0;valid=0;failures=[]
for stage in range(5,14):
 scale=65/(3*min(13,stage+1))
 for h in range(4,157):
  for v in range(-9,10):
   l=max(15,h-15,h-(v+.5)*scale);u=min(145,h+15,h-(v-.5)*scale)
   for p in [0,1]:
    obs=np.zeros((1,6),np.float32);obs[0,4]=p/80;obs[0,1]=(l-80)/80;obs[0,0]=(u-80)/80
    result=s.run(None,{'observation':obs});actual_lo=float(result[0][0,0]);actual_hi=float(result[1][0,0])
    expected_lo=p+2*np.clip(np.ceil((l-p)/2),7,73);expected_hi=expected_lo+(2 if u-expected_lo>=2 else 0)
    if actual_lo!=expected_lo or actual_hi!=expected_hi:failures.append(dict(stage=stage,h=h,v=v,parity=p,rawlo=l,rawhi=u,expected_lo=expected_lo,expected_hi=expected_hi,actual_lo=actual_lo,actual_hi=actual_hi))
    if expected_lo<=u:
     valid+=1
     assert expected_hi==p+2*np.floor((u-p)/2)
    count+=1
result=dict(lattice_band_cases=count,nonempty_lattice_bands=valid,failures=failures,empty_band_gap_preserved='Must use lower−raw_upper before quantizing upper')
(OUT/'grid-lattice-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert not failures
