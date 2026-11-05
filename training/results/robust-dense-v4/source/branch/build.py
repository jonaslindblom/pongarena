"""One analytically selected continuous robust contact branch, no fitting.

The target curve follows one connected safe shot family instead of selecting
pointwise minimum-risk contacts, which jump between disconnected families.
At cap13 each y+11.5 goal yields only adjacent integer centers y+11,y+12;
these are both safe on17..50. A two-unit ramp connects to y+13.5 on52..78.
The upper half mirrors this branch. A continuous crossover uses the three
central heights79..81, where no pair of adjacent centers is safe anyway.
All constants come from legal-shot safety constraints, not gameplay rewards.
"""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
sys.path.insert(0,'/tmp/rally-analytic-dense')
from cap_global import first
from build import export as parent_export
# Importabove resolves thisfile ifrunbyscript! Loadexplicitlyinstead.
import importlib.util
spec=importlib.util.spec_from_file_location('parent_dense_build','/tmp/rally-analytic-dense/build.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
import numpy as np,torch,onnx
from torch import nn
from env import reflect
from pace_study import tables,CONFIGS,make_policy,run_pair
from competition_eval import load_mlp
from policy import ROOT,load_builtin
OUT=Path('/tmp/rally-safe-branch');RULES=CONFIGS['pace13-v9-scaled']
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
 W1,B1,z,zb,B=first();N=len(B1)-2;W1=W1[:N];B1=B1[:N];z=z[:N]
 raw,values,co,used=raw_curve();K=len(used)
 W2=np.zeros((K+2,N),np.float32);B2=np.zeros(K+2,np.float32)
 for j,i in enumerate(used):W2[j]=z;W2[j,B+3]-=20;B2[j]=zb-(raw[i]-80)/80
 W2[K,B+4]=1;W2[K,B+5]=-1;W2[K+1,B+2]=1
 down=np.r_[co[used],(values[0]-80)/80,-1]
 W3=np.stack([-down,np.zeros(K+2),down]).astype(np.float32);B3=np.array([-1,1.101/80,1],np.float32)
 net=nn.Sequential(nn.Linear(6,N),nn.ReLU(),nn.Linear(N,K+2),nn.ReLU(),nn.Linear(K+2,3)).eval()
 with torch.no_grad():
  for layer,w,b in zip([net[0],net[2],net[4]],[W1,W2,W3],[B1,B2,B3]):layer.weight.copy_(torch.from_numpy(w));layer.bias.copy_(torch.from_numpy(b))
 return net,dict(architecture=[6,N,K+2,3],parameters=sum(p.numel() for p in net.parameters()),macs=6*N+N*(K+2)+(K+2)*3,height_knots=HEIGHTS.tolist(),height_targets=TARGETS.tolist(),raw_knots=raw.tolist(),raw_targets=values.tolist(),active_hinges=K,
 method='Deterministic legal-shot interval reasoning and manual connected branch selection; no DP, observed-label fitting, optimization, training, mutation or game-reward selection',outgoing='center80;12hinge edge recovery would exceedSmall2048budget',limitations=['Cap13forecaster exactonphysicalcaptrajectorylattice;lowerstagesapproximate','Goal may beunreachablefromcurrentpaddle','Onlyone-replyuniversalsafety,notindefinitewinningguarantee','Parityrobustnessrequiresstationarywithin1.101goal;forecastdisturbanceandmovementpathcanstillmatter'])

def audit(net):
 T=tables(RULES);h=np.arange(4,157);c=np.arange(15,146);R=np.full((153,131),np.inf)
 for i,y in enumerate(h):
  for j,p in enumerate(c):
   if abs(y-p)<=15:R[i,j]=T['risk'][13,y,y-p+15]
 possible=abs(c[None]-goal(h)[:,None])<=1.101
 worst=np.where(possible,R,-np.inf).max(1)
 pair=(R[:,:-1]<=0)&(R[:,1:]<=0)
 direct=dict(heights=153,adjacent_safe_pair_exists=int(pair.any(1).sum()),robust_goal_safe=int((worst<=0).sum()),positive_risk=int((worst>0).sum()),worst_risk=float(worst.max()),mean_worst_risk=float(worst.mean()),unsafe_heights=h[worst>0].tolist(),risk_per_height=worst.tolist())
 # Actual integer cap-contact states withoneflighttick (x20), respectingwallreflection.
 # y isprecontactcomponent; generatedcapstatecanhaveseitherverticalsign.
 obs=[];truth=[]
 for y in range(4,157):
  for vy in range(-9,10):
   land=reflect(y+vy)
   for own in range(15,146):
    obs.append([(20-180)/180,(y-80)/80,-13/6,vy/6,(own-80)/80,0]);truth.append((goal(land)-own)/80)
 obs=np.array(obs,np.float32);truth=np.array(truth)
 with torch.no_grad():scores=torch.cat([net(p) for p in torch.from_numpy(obs).split(4096)]).numpy()
 expected=np.stack([-truth,np.full(len(truth),1.101/80),truth],1)
 numeric=dict(component_contact_probes=len(obs),action_mismatches=int((scores.argmax(1)!=expected.argmax(1)).sum()),max_score_error=float(abs(scores-expected).max()),max_target_error_units=float(abs(scores[:,2]-expected[:,2]).max()*80))
 return dict(physical=direct,numeric=numeric)

if __name__=='__main__':
 torch.set_num_threads(1);net,description=make();assert description['parameters']<=2048
 torch.save(net.state_dict(),OUT/'small-robust-branch.pt');parent.export(net,OUT/'small-robust-branch.onnx');p=OUT/'small-robust-branch.onnx';m=onnx.load(p)
 onnx.helper.set_model_props(m,dict(rules='pong-v4',architecture='6-42-33-3 ReLU',method='Analytic ordinarydense exactcapforecast and continuousparityrobustcontactbranch;no training',target='Connectedadjacent-safecontactfamily;outgoingcenter80',limitation='One-replyandcap13prior;lowerstagesandmovementreachabilityremainapproximate'));onnx.save(m,p)
 description['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(OUT/'construction.json').write_text(json.dumps(description,indent=2)+'\n')
 checks=audit(net);(OUT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n');print('BUILT',json.dumps(description),flush=True);print('AUDIT',json.dumps(checks),flush=True)
 seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds'];opponents={k:load_builtin(k) for k in ['tracker','anticipator','challenger']};opponents['pacer']=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');opponents.update({k:make_policy(RULES,k) for k in ['attack','defend']})
 result=dict(purpose='One declared Small coherentparityrobustphysicalbranch;nofitting/rewardselection',architecture=description['architecture'],parameters=description['parameters'],macs=description['macs'],sha256=description['sha256'],seeds=seeds,pairs={})
 for name,other in opponents.items():
  row=run_pair(RULES,net,other,seeds);result['pairs'][name]=row;(OUT/'results.json').write_text(json.dumps(result,separators=(',',':'))+'\n');s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
 result['complete']=True;(OUT/'results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
