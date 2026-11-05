"""Native ONNX proof probes for the compact counter's saturated-Tanh lattice."""
import importlib.util,json
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper
OUT=Path('/tmp/rally-attacker-audit')
spec=importlib.util.spec_from_file_location('counter_builder','/tmp/rally-counter-root/builder.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
g=m.G();actor=g.project(4,80,80);rawlo=g.project(1,80,80)
fp=g.affine(g.op('Tanh',g.add(g.scalar(actor,4096),-8192*(np.arange(8,73)-1/512))),np.ones((65,1)),39.5,alpha=.5)
parity=g.sub(actor,g.scalar(fp,2))
z=g.scalar(g.sub(rawlo,parity),4096)
q=g.affine(g.op('Tanh',g.add(z,-8192*(np.arange(7,73)+1/512))),np.ones((66,1)),40,alpha=.5)
lo=g.op('Add',parity,g.scalar(q,2))
model=helper.make_model(helper.make_graph(g.nodes,'Independent lattice probe',[helper.make_tensor_value_info('observation',1,[1,6])],[helper.make_tensor_value_info(fp,1,[1,1]),helper.make_tensor_value_info(lo,1,[1,1])],g.weights),opset_imports=[helper.make_opsetid('',18)],ir_version=10)
onnx.checker.check_model(model);onnx.save(model,OUT/'lattice-ceil-probe.onnx')
opt=ort.SessionOptions();opt.intra_op_num_threads=opt.inter_op_num_threads=1
s=ort.InferenceSession(str(OUT/'lattice-ceil-probe.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
failures=[];normal=0;invalid=0;maxerr=0
for actor in range(15,146):
 obs=np.zeros((1,6),np.float32);obs[0,4]=(actor-80)/80
 values=s.run(None,{'observation':obs});assert values[0][0,0]==actor//2,(actor,values)
for stage in range(5,14):
 outgoing=min(13,stage+1);scale=65/(3*outgoing)
 for h in range(4,157):
  for v in range(-9,10):
   raw=max(15,h-15,h-(v+.5)*scale)
   for parity in [0,1]:
    actor=80+parity;expected=parity+2*np.ceil((raw-parity)/2)
    obs=np.zeros((1,6),np.float32);obs[0,4]=(actor-80)/80;obs[0,1]=(raw-80)/80
    actual=float(s.run(None,{'observation':obs})[1][0,0])
    if raw<=145:
     normal+=1;err=abs(actual-expected);maxerr=max(maxerr,err)
     if err:failures.append(dict(stage=stage,h=h,v=v,rawlo=raw,parity=parity,expected=float(expected),actual=actual))
    else:
     invalid+=1
     # Finite hinge bank saturates above legal paddle centers. It must still
     # reject every such band, whose upper bound is at most145.
     if not actual>145:failures.append(dict(kind='invalid_legal_rejection',rawlo=raw,actual=actual,parity=parity))
result=dict(actor_floor_probes=131,valid_band_lattice_probes=normal,above_court_band_rejection_probes=invalid,max_valid_lattice_error=maxerr,failures=failures,
 scope='All stage5..13, landing4..156, vy-9..9 and both ordinary movement parities; clamps that change parity remain a disclosed approximation')
(OUT/'lattice-ceil-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert not failures
