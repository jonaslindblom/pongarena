"""Native inference checks for the single fixed analytic dense candidate."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch,onnx
from onnx import numpy_helper
from policy import OnnxPolicy
from competition_eval import load_mlp
OUT=Path('/tmp/rally-speed-dense');torch.set_num_threads(1)
model=load_mlp(OUT/'variable-speed.pt','relu');graph=onnx.load(OUT/'variable-speed.onnx')
assert [n.op_type for n in graph.graph.node]==['Gemm','Relu','Gemm','Relu','Gemm']
weights={v.name:numpy_helper.to_array(v) for v in graph.graph.initializer}
for layer,i in [(model[0],0),(model[2],2),(model[4],4)]:
 np.testing.assert_array_equal(layer.weight.detach().numpy(),weights[f'w{i}']);np.testing.assert_array_equal(layer.bias.detach().numpy(),weights[f'b{i}'])
obs=np.load('/tmp/rally-causal-failures/observations.npy')
ids=np.random.default_rng(2026110532).choice(len(obs),8192,replace=False);x=obs[ids]
with torch.no_grad():torch_scores=model(torch.from_numpy(x)).numpy()
session=OnnxPolicy(OUT/'variable-speed.onnx').session
native=np.array([session.run(None,{'observation':row[None]})[0][0] for row in x])
error=abs(native-torch_scores);disagree=native.argmax(1)!=torch_scores.argmax(1)
assert np.isfinite(native).all()
records=[{'observation':x[i].tolist(),'torch':torch_scores[i].tolist(),'native':native[i].tolist()} for i in np.flatnonzero(disagree)]
result={'purpose':'NativeONNX parity on retained reachable audit states; implementation not selection','observations':len(x),'action_mismatches':int(disagree.sum()),'maximum_score_error':float(error.max()),'mean_score_error':float(error.mean()),'first_mismatches':records[:10],'sha256':hashlib.sha256((OUT/'variable-speed.onnx').read_bytes()).hexdigest(),'graph_nodes':['Gemm','Relu','Gemm','Relu','Gemm'],'architecture':[6,125,9,3],'embedded_weight_tensors':len(graph.graph.initializer)}
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
