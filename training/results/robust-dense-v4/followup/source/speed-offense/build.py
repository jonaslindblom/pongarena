"""One fixed far-corner output transfer onto the frozen variable-speed MLP.

Hiddenweightsandbiasesarecopiedexactly;onlythefinaldensematrixandstaybias
change. Incomingtargetnorm=(1+1/15.5)*landingnorm+65/(80*15.5)*opponentcue.
Coefficientsaresameaspreviousrule-derivedfar-cornercontrol.No fitting,
optimizer, coefficient sweep, or game reward determines these weights.
"""
from pathlib import Path
import sys,json,hashlib,importlib.util
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch,onnx
from onnx import numpy_helper
from competition_eval import load_mlp
from pace_study import CONFIGS,make_policy,run_pair
from policy import ROOT,OnnxPolicy
OUT=Path('/tmp/rally-speed-offense');SOURCE=Path('/tmp/rally-speed-dense/variable-speed-lattice.pt')
spec=importlib.util.spec_from_file_location('lattice_builder','/tmp/rally-speed-dense/build_lattice.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

def make():
 net=load_mlp(SOURCE,'relu');source={k:v.clone()for k,v in net.state_dict().items()}
 gain=1+1/15.5;cue=65/(80*15.5)
 down=np.r_[gain*np.array([1,-2,2,-2,2]),-.95*gain-cue,-1,1,cue].astype(np.float32)
 with torch.no_grad():
  net[4].weight[0].copy_(torch.from_numpy(-down));net[4].weight[2].copy_(torch.from_numpy(down));net[4].bias[1]=1.101/80
 for k,v in net.state_dict().items():
  if not k.startswith('4.'):assert torch.equal(v,source[k]),k
 return net,dict(architecture=[6,123,9,3],parameters=sum(p.numel()for p in net.parameters()),macs=1872,gain=gain,cue=cue,deadband_units=1.101,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),hidden_tensors_identical=True,target='Incominggain*approximatereflectedlandingnorm+cue*clipped(10*opponentnorm,−1,+1);outgoingcenter80',limitations=['Approximatevariable-speedforecast','Targetusesfar-corner15/145 approximationand15.5angleflightcoefficient','Notlearnedsourceweighttransfer;notformaltraining'])

def verify(net,path):
 graph=onnx.load(path);ops=[n.op_type for n in graph.graph.node];assert ops==['Gemm','Relu','Gemm','Relu','Gemm']
 weights={v.name:numpy_helper.to_array(v)for v in graph.graph.initializer}
 for layer,i in [(net[0],0),(net[2],2),(net[4],4)]:
  np.testing.assert_array_equal(layer.weight.detach().numpy(),weights[f'w{i}']);np.testing.assert_array_equal(layer.bias.detach().numpy(),weights[f'b{i}'])
 obs=np.load('/tmp/rally-causal-failures/observations.npy');ids=np.random.default_rng(2026110537).choice(len(obs),8192,replace=False);x=obs[ids]
 with torch.no_grad():
  hidden=net[:4](torch.from_numpy(x)).numpy();actual=net(torch.from_numpy(x)).numpy()
 incoming=hidden[:,5];landing=hidden[:,:5]@np.array([1,-2,2,-2,2])-.95*incoming;own=hidden[:,6]-hidden[:,7];opp=hidden[:,8]-incoming
 error=(1+1/15.5)*landing+65/(80*15.5)*opp-own;expected=np.stack([-error,np.full(len(x),1.101/80),error],1)
 source=load_mlp(SOURCE,'relu')
 for i in [0,2]:
  np.testing.assert_array_equal(source[i].weight.detach().numpy(),net[i].weight.detach().numpy());np.testing.assert_array_equal(source[i].bias.detach().numpy(),net[i].bias.detach().numpy())
 session=OnnxPolicy(path).session;native=np.array([session.run(None,{'observation':row[None]})[0][0]for row in x]);assert np.isfinite(native).all()
 mismatch=int((native.argmax(1)!=actual.argmax(1)).sum())
 return dict(purpose='Implementationaudit;notselection',probes=len(x),graph_nodes=ops,embedded_tensors=len(weights),hidden_arrays_identical=True,formula_max_score_error=float(abs(actual-expected).max()),formula_action_mismatches=int((actual.argmax(1)!=expected.argmax(1)).sum()),native_max_score_error=float(abs(native-actual).max()),native_mean_score_error=float(abs(native-actual).mean()),native_action_mismatches=mismatch)

if __name__=='__main__':
 torch.set_num_threads(1);net,description=make();torch.save(net.state_dict(),OUT/'variable-speed-farcorner.pt');path=OUT/'variable-speed-farcorner.onnx';builder.export(net,path,description);graph=onnx.load(path)
 onnx.helper.set_model_props(graph,dict(rules='pong-v4',architecture='6-123-9-3 ReLU',method='Fixedrule-derivedfarcorneroutputprioronfrozenanalyticvariable-speedlatticeMLP;noformaltraining',target=description['target'],limitations='Approximateforecasterandangleflight;conditionaloffensivepriornotprovenupgrade'));onnx.save(graph,path)
 description['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(OUT/'construction.json').write_text(json.dumps(description,indent=2)+'\n');checks=verify(net,path);(OUT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n');print('BUILD',json.dumps(description),flush=True);print('VERIFY',json.dumps(checks),flush=True)
 seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds'];rules=CONFIGS['pace13-v9-scaled'];opponents=dict(pacer=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'),attack=make_policy(rules,'attack'),defend=make_policy(rules,'defend'))
 results=dict(purpose='Onedeclaredoutputonlyoffensivecontrol;unchangedhiddenweights;nofittingorrewardsweep',seeds=seeds,architecture=description['architecture'],parameters=description['parameters'],sha256=description['sha256'],pairs={})
 for name,other in opponents.items():
  row=run_pair(rules,net,other,seeds);results['pairs'][name]=row;(OUT/'results.json').write_text(json.dumps(results,separators=(',',':'))+'\n');s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
 results['complete']=True;(OUT/'results.json').write_text(json.dumps(results,separators=(',',':'))+'\n')
