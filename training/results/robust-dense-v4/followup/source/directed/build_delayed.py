"""One Small dense opponent-selected delayed safe-shot branch control.

Globalcapforecastisunchangedordinarydensealgebra. Sharedbasebranchusesramps
46..48and112..114. Onlywhenunfoldedforecastisphysicaldoesopponentposition
selectthealternativefamily: lower51..73goal−10,upper87..109goal+10.
Bothfamilieshaveparity-robustone-replysafetyifattained. Lastwallhasoccurred
atleastsixstepsbeforetheselandings; thatpermitsa10-unitgoalswitchfromnear
thebasegoal,butdoesnotguaranteereachabilityfromeveryactualpaddlestate.
No game rewards, optimizer, fitting, random mutation, or coefficient sweep.
"""
from pathlib import Path
import sys,json,hashlib,importlib.util
sys.path.insert(0,str(Path.cwd()/'training/scripts'));sys.path.insert(0,'/tmp/rally-safe-branch')
from compact_first import first
import numpy as np,torch,onnx
from torch import nn
from onnx import helper,numpy_helper
from env import reflect
from pace_study import CONFIGS,tables,make_policy,run_pair
from competition_eval import load_mlp
from policy import ROOT,OnnxPolicy
OUT=Path('/tmp/rally-directed-branches');RULES=CONFIGS['pace13-v9-scaled']
H=np.array([4,46,48,78,82,112,114,156.]);V=np.array([15.5,57.5,61.5,91.5,68.5,98.5,102.5,144.5])
LOWER=np.array([50,51,73,74.]);UPPER=np.array([86,87,109,110.]);CO=np.array([1,-1,-1,1.])

def base(q):return np.interp(reflect(q),H,V)
def pulse(q,knots):return np.maximum(q-knots[0],0)-np.maximum(q-knots[1],0)-np.maximum(q-knots[2],0)+np.maximum(q-knots[3],0)
def target(q,other):return base(q)-10*pulse(q,LOWER)*(other>=80)+10*pulse(q,UPPER)*(other<80)

def make():
 f=first();w1=np.zeros((40,6));b1=np.zeros(40);w1[:38]=f['weight'][:38];b1[:38]=f['bias'][:38];w1[38,5]=80;b1[38]=1;w1[39,5]=80
 q=np.r_[f['incoming'][:38],0,0];qb=f['incoming_bias']
 raw=[]
 for n in range(-3,4):
  for h in H:raw.extend([h+304*n,312-h+304*n])
 raw=np.unique(np.r_[-240,raw,400]);raw=raw[(raw>=-240)&(raw<=400)];v=base(raw);sl=np.diff(v)/np.diff(raw);co=np.r_[sl[0],np.diff(sl)];used=np.flatnonzero(abs(co)>1e-12);assert len(used)==31
 w2=np.zeros((40,40));b2=np.zeros(40);down=np.zeros(40)
 for j,i in enumerate(used):w2[j]=q;w2[j,37]-=20;b2[j]=qb-(raw[i]-80)/80;down[j]=co[i]
 # Foldtheincomingconstantintothealways-activefirstforecastanchor.
 b2[0]+=(v[0]-80)/80/co[used[0]]
 w2[31,36]=1;down[31]=-1
 for j,(knots,delta,bottom_mode)in enumerate([(LOWER,-10,True),(UPPER,10,False)]):
  for i,k in enumerate(knots):
   ix=32+4*j+i;w2[ix]=q;w2[ix,37]-=20;b2[ix]=qb-(k-80)/80;down[ix]=delta*CO[i]
   if bottom_mode:w2[ix,38]+=20;w2[ix,39]-=20;b2[ix]-=20
   else:w2[ix,38]-=20;w2[ix,39]+=20
 w3=np.stack([-down,np.zeros(40),down]);b3=np.array([-1,1.101/80,1])
 net=nn.Sequential(nn.Linear(6,40),nn.ReLU(),nn.Linear(40,40),nn.ReLU(),nn.Linear(40,3)).eval()
 with torch.no_grad():
  for layer,w,b in zip([net[0],net[2],net[4]],[w1,w2,w3],[b1,b2,b3]):layer.weight.copy_(torch.from_numpy(w.astype(np.float32)));layer.bias.copy_(torch.from_numpy(b.astype(np.float32)))
 return net,dict(architecture=[6,40,40,3],parameters=sum(p.numel()for p in net.parameters()),macs=1960,base_height_knots=H.tolist(),base_values=V.tolist(),lower_pulse_knots=LOWER.tolist(),upper_pulse_knots=UPPER.tolist(),opponent_selector='center>=80selectstopalternativeonlower51..73;center<80selectsbottomalternativeonupper87..109',correction_units=10,method='Fixedanalyticalfamilyselectionandtiminggeometry;notraining,fitting,optimizer,mutationorrewardselection',outgoing='center80',limitations=['Delayedgoalcanbeunreachableifbasegoalwasalreadybarelyreachable','Conditionalbranchesonlyinthephysicalunfoldedforecastwindow,notearlierreflectioncopies','Lowerflightstagesusecap13approximation','One-replysafetyifstationarygoalattained;notinfinitewinningguarantee'])

def export(net,path,description):
 nodes=[];initial=[];current='observation'
 for i,layer in enumerate(net):
  output='action_scores'if i==4 else f'h{i}'
  if isinstance(layer,nn.Linear):
   initial.extend([numpy_helper.from_array(layer.weight.detach().numpy(),f'w{i}'),numpy_helper.from_array(layer.bias.detach().numpy(),f'b{i}')]);nodes.append(helper.make_node('Gemm',[current,f'w{i}',f'b{i}'],[output],transB=1))
  else:nodes.append(helper.make_node('Relu',[current],[output]))
  current=output
 graph=helper.make_model(helper.make_graph(nodes,'Denseopponentconditionaldelayedcontactfamily',[helper.make_tensor_value_info('observation',1,[1,6])],[helper.make_tensor_value_info('action_scores',1,[1,3])],initial),opset_imports=[helper.make_opsetid('',18)],ir_version=10)
 helper.set_model_props(graph,dict(rules='pong-v4',architecture='6-40-40-3 ReLU',method=description['method'],target=description['opponent_selector'],limitations='Delayedgoalreachabilityandlowerstageforecastremainapproximate;outgoingcenter80'));onnx.checker.check_model(graph);onnx.save(graph,path)

def audit(net,path):
 T=tables(RULES);physical={}
 for other in [40,120]:
  safe=0;catch=0;direct=0
  for h in range(4,157):
   g=target(h,other);centers=[p for p in range(15,146)if abs(p-g)<=1.101];valid=all(abs(h-p)<=15 for p in centers);risks=[T['risk'][13,h,h-p+15]for p in centers]if valid else[1e9];s=valid and max(risks)<=0;safe+=s;catch+=valid
   dest=[T['landing'][13,h,h-p+15]for p in centers]if valid else[];direct+=s and (max(dest)<=22 if other>=80 else min(dest)>=138)
  physical[str(other)]=dict(robust_safe_heights=int(safe),catch_valid_heights=int(catch),desired_edge_heights=int(direct))
 obs=np.load('/tmp/rally-causal-failures/observations.npy');ids=np.random.default_rng(2026110542).choice(len(obs),8192,replace=False);x=obs[ids]
 f=first();qh=np.maximum(0,x@f['weight'].T+f['bias']);q=(qh@f['incoming']+f['incoming_bias'])*80+80;own=x[:,4]*80+80;other=np.rint(x[:,5]*80+80);incoming=x[:,2]<0;t=np.where(incoming,target(q,other),80);error=(t-own)/80;expected=np.stack([-error,np.full(len(x),1.101/80),error],1)
 with torch.no_grad():scores=net(torch.from_numpy(x)).numpy()
 session=OnnxPolicy(path).session;native=np.array([session.run(None,{'observation':p[None]})[0][0]for p in x]);assert np.isfinite(native).all()
 return dict(physical=physical,native=dict(probes=len(x),action_mismatches=int((scores.argmax(1)!=native.argmax(1)).sum()),max_score_error=float(abs(scores-native).max())),formula=dict(action_mismatches=int((scores.argmax(1)!=expected.argmax(1)).sum()),max_score_error=float(abs(scores-expected).max())),graph_nodes=[n.op_type for n in onnx.load(path).graph.node],initializer_tensors=len(onnx.load(path).graph.initializer))

if __name__=='__main__':
 torch.set_num_threads(1);net,desc=make();assert desc['parameters']==2043;torch.save(net.state_dict(),OUT/'small-delayed-directed.pt');path=OUT/'small-delayed-directed.onnx';export(net,path,desc);desc['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(OUT/'delayed-construction.json').write_text(json.dumps(desc,indent=2)+'\n');checks=audit(net,path);(OUT/'delayed-verification.json').write_text(json.dumps(checks,indent=2)+'\n');print('BUILD',json.dumps(desc),flush=True);print('AUDIT',json.dumps(checks),flush=True)
 seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds'];policies=dict(pacer=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'),attack=make_policy(RULES,'attack'),defend=make_policy(RULES,'defend'),prism_small=OnnxPolicy(ROOT/'public/models/prism-small-v4.onnx'))
 result=dict(purpose='OnefixedSmallconditionaldirectedcontactfamilycontrol;nooutcomeselection',seeds=seeds,architecture=desc['architecture'],parameters=desc['parameters'],sha256=desc['sha256'],pairs={})
 for name,other in policies.items():
  row=run_pair(RULES,net,other,seeds);result['pairs'][name]=row;(OUT/'delayed-results.json').write_text(json.dumps(result,separators=(',',':'))+'\n');s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
 result['complete']=True;(OUT/'delayed-results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
