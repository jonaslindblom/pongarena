"""One physically corrected8-hinge outgoing recovery prior, ordinary dense only.

The frozen incoming robust contact branch is preserved by source-weight algebra.
Outgoing recovery centers until the first wall bounce, then uses an inner raw
forecast profile selected by exact lattice safety inequalities, not game scores.
"""
from pathlib import Path
import sys,json,hashlib,subprocess,importlib.util
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
sys.path.insert(0,'/tmp/rally-safe-branch')
from compact_first import first
import numpy as np,torch,onnx,onnxruntime as ort
from torch import nn
from onnx import helper,numpy_helper
from competition_eval import load_mlp
from policy import ROOT,load_builtin
from pace_study import CONFIGS,tables,run_pair,make_policy
from env import reflect
OUT=Path('/tmp/rally-policy-recovery');RULES=CONFIGS['pace13-v9-scaled'];T=tables(RULES)
KNOTS=np.array([-8.6,4,15,22,138,145,156,168.6]);VALUES=np.array([80,95.75,82,78.5,81.5,78,64.25,80])
SLOPES=np.diff(VALUES)/np.diff(KNOTS);COEFF=np.r_[SLOPES[0],np.diff(SLOPES),-SLOPES[-1]]
SOURCE=Path('/tmp/rally-safe-branch/small-robust-branch.pt')

def profile(q):
 return 80+np.maximum(np.asarray(q)[...,None]-KNOTS,0)@COEFF

def make():
 source=load_mlp(Path('/tmp/rally-safe-branch/algebra-identical-compact.pt'),activation='relu');api=first();N=39;K=source[2].out_features
 assert K==32
 np.testing.assert_array_equal(source[0].weight.detach().numpy(),api['weight'][:38])
 np.testing.assert_array_equal(source[0].bias.detach().numpy(),api['bias'][:38])
 net=nn.Sequential(nn.Linear(6,N),nn.ReLU(),nn.Linear(N,K+8),nn.ReLU(),nn.Linear(K+8,3)).eval()
 with torch.no_grad():
  net[0].weight.copy_(torch.from_numpy(api['weight']));net[0].bias.copy_(torch.from_numpy(api['bias']))
  net[2].weight.zero_();net[2].bias.zero_();net[4].weight.zero_();net[4].bias.copy_(source[4].bias)
  net[2].weight[:K,:38].copy_(source[2].weight);net[2].bias[:K].copy_(source[2].bias);net[4].weight[:,:K].copy_(source[4].weight)
  for j,k in enumerate(KNOTS):
   w=api['outgoing'].copy();w[api['indices']['negative_vx']]-=20
   net[2].weight[K+j].copy_(torch.from_numpy(w));net[2].bias[K+j]=api['outgoing_bias']-(k-80)/80
   net[4].weight[2,K+j]=COEFF[j];net[4].weight[0,K+j]=-COEFF[j]
 return net

def export(net,path):
 nodes=[];weights=[];previous='observation'
 for i,layer in enumerate(net):
  output='action_scores' if i==len(net)-1 else f'h{i}'
  if isinstance(layer,nn.Linear):
   weights.extend([numpy_helper.from_array(layer.weight.detach().numpy(),f'w{i}'),numpy_helper.from_array(layer.bias.detach().numpy(),f'b{i}')]);nodes.append(helper.make_node('Gemm',[previous,f'w{i}',f'b{i}'],[output],transB=1))
  else:nodes.append(helper.make_node('Relu',[previous],[output]))
  previous=output
 m=helper.make_model(helper.make_graph(nodes,'Dense robust contact with delayed parity-corrected recovery',[helper.make_tensor_value_info('observation',1,[1,6])],[helper.make_tensor_value_info('action_scores',1,[1,3])],weights),opset_imports=[helper.make_opsetid('',18)],ir_version=10)
 helper.set_model_props(m,dict(rules='pong-v4',architecture='6-39-40-3 ReLU',method='Source-weight algebra plus physically constrained8-hinge recovery prior; no training',incoming='Frozen continuous connected adjacent-safe contact branch unchanged',outgoing='Delayed raw-forecast innerprofile; forced15→82,145→78 and deadband-corrected slopes1.25',limitations='Onlycap13forecaster exact;lowerstagesapproximate;stationary-shot proofdoesnotguaranteeallgameplay'))
 onnx.checker.check_model(m);onnx.save(m,path)

def verify(net,path):
 original=load_mlp(SOURCE,activation='relu');obs=np.load('/tmp/rally-causal-failures/observations.npy');incoming=obs[:,2]<0
 with torch.no_grad():
  old=torch.cat([original(t) for t in torch.from_numpy(obs).split(4096)]).numpy();new=torch.cat([net(t)for t in torch.from_numpy(obs).split(4096)]).numpy()
 inc_changed=int((old[incoming].argmax(1)!=new[incoming].argmax(1)).sum());assert inc_changed==0
 # Independent complete26-tick NN action check for every admitted stationary
 # center at the124 robust contact heights. Include both possible parity cases.
 hkn=np.array([4,50,52,78,82,108,110,156]);hval=np.array([15.5,61.5,65.5,91.5,68.5,94.5,98.5,144.5]);shots=[]
 for h in range(4,157):
  target=float(np.interp(h,hkn,hval));centers=[p for p in range(15,146) if abs(p-target)<=1.101]
  if not all(abs(h-p)<=15 and T['risk'][13,h,h-p+15]<=0 for p in centers):continue
  for p in centers:shots.append((h,p,int(np.floor(.6*(h-p)+.5))))
 y=np.array([r[0] for r in shots]);p=np.array([r[1] for r in shots]);v=np.array([r[2] for r in shots]);p_ref=p.copy();mismatches=0;max_target_error=0.
 for tick in range(26):
  x=15+13*tick;q=y+(26-tick)*v;target=profile(q)
  batch=np.stack([np.full(len(y),(x-180)/180),(y-80)/80,np.full(len(y),13/6),v/6,(p-80)/80,np.zeros(len(y))],axis=1).astype(np.float32)
  with torch.no_grad():scores=net(torch.from_numpy(batch)).numpy()
  action=scores.argmax(1);expected=np.where(target-p>1.101,2,np.where(p-target>1.101,0,1));mismatches+=int((action!=expected).sum());reconstructed=80+80*(scores[:,2]+batch[:,4]);max_target_error=max(max_target_error,float(abs(reconstructed-target).max()));p=np.clip(p+2*(action-1),15,145);p_ref=np.clip(p_ref+2*(expected-1),15,145)
  y=y+v;wall=(y<4)|(y>156);y=np.where(y<4,8-y,np.where(y>156,312-y,y));v=np.where(wall,-v,v)
 assert mismatches==0 and np.array_equal(p,p_ref)
 low=T['low'][13,y];high=T['high'][13,y];risks=np.maximum(low-p,p-high)
 idx=np.random.default_rng(2026110601).choice(len(obs),8192,replace=False);probe=obs[idx];options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1;session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);native=np.array([session.run(None,{'observation':r[None]})[0][0] for r in probe]);expected=new[idx];numeric_mismatches=int((native.argmax(1)!=expected.argmax(1)).sum());assert numeric_mismatches==0;assert np.isfinite(native).all()
 code="""import{readFileSync}from'node:fs';import{validateArenaModel}from'./src/validate.ts';import{smallestDivision}from'./src/competition.ts';const b=readFileSync(process.argv[1]);const s=validateArenaModel(b);console.log(JSON.stringify({bytes:b.length,params:s.params,macs:s.macs,nodes:s.nodes,widths:s.architecture.widths,division:smallestDivision(s)}));"""
 arena=json.loads(subprocess.run(['node','--import','tsx','--input-type=module','-e',code,str(path)],check=True,capture_output=True,text=True).stdout);assert arena['division']=='small' and arena['params']==1996
 result=dict(purpose='Verification before developmentgames; originalsource and protectedfinalresults untouched',source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),arena=arena,incoming_preservation=dict(probes=int(incoming.sum()),action_mismatches=inc_changed,max_score_error=float(abs(old[incoming]-new[incoming]).max())),wholeflight_stationary=dict(shots=len(shots),decisions=len(shots)*26,action_mismatches=mismatches,max_target_error_units=max_target_error,safe_final=int((risks<=0).sum()),unsafe_final=int((risks>0).sum()),maximum_risk=int(risks.max())),native_numeric=dict(probes=8192,action_mismatches=numeric_mismatches,max_score_error=float(abs(native-expected).max())),profile=dict(knots=KNOTS.tolist(),values=VALUES.tolist(),coefficients=COEFF.tolist(),basis='Exactlegalshot/lattice/deadbandconstraints, not game outcomes'))
 (OUT/'inner-verification.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
 torch.set_num_threads(1);net=make();path=OUT/'small-inner-recovery.onnx';torch.save(net.state_dict(),OUT/'small-inner-recovery.pt');export(net,path);check=verify(net,path);print('VERIFIED',json.dumps(check),flush=True)
 seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds'];opponents={k:load_builtin(k)for k in ['tracker','anticipator','challenger']};opponents['pacer']=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');opponents.update({k:make_policy(RULES,k)for k in ['attack','defend']});opponents['frozen-robust-small']=load_mlp(SOURCE,activation='relu')
 result=dict(purpose='One predeclaredphysicallycorrected outgoing prior; no fitting/gainsearch; development only',seeds=seeds,verification=check,pairs={})
 for name,other in opponents.items():
  row=run_pair(RULES,net,other,seeds);result['pairs'][name]=row;(OUT/'inner-results.json').write_text(json.dumps(result,separators=(',',':'))+'\n');s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
 result['complete']=True;(OUT/'inner-results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
