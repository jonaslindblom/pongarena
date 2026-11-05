"""Read-only learned MLP audit and explained channel transfer, without fitting."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np, torch
from torch import nn
from competition_eval import load_mlp,openings
from pace_study import CONFIGS,PaceArena
from policy import ROOT,load_builtin,actions,export

OUT=Path('/tmp/rally-dense-transfer');OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
seeds=(np.random.default_rng(2026110520).choice(10_000_000,24,replace=False)+1_150_000_000).tolist()
seen=set();files=0;leaves=0;errors=[]
def visit(x):
 global leaves
 if isinstance(x,dict):
  for k,v in x.items():visit(k);visit(v)
 elif isinstance(x,list):
  for v in x:visit(v)
 elif isinstance(x,int) and not isinstance(x,bool):seen.add(x);leaves+=1
 elif isinstance(x,str) and x.isdigit():seen.add(int(x))
for root in [ROOT/'training/results',ROOT/'public/training']:
 for p in root.rglob('*'):
  if p.suffix not in ['.json','.jsonl']:continue
  try:
   objects=[json.loads(v) for v in p.read_text().splitlines() if v.strip()] if p.suffix=='.jsonl' else [json.loads(p.read_text())]
   for x in objects:visit(x)
   files+=1
  except Exception as e:errors.append([str(p),str(e)])
assert not errors,errors
assert not set(seeds)&seen, sorted(set(seeds)&seen)
(OUT/'seed-audit.json').write_text(json.dumps(dict(seeds=seeds,files=files,integer_leaves=leaves,overlaps=[]),indent=2)+'\n')
net=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt')
old=load_mlp(ROOT/'training/results/v2/ppo.pt')
tracker=load_builtin('tracker')
frames,sides=openings(seeds)
frames=frames[:len(frames)//2]
env=PaceArena(len(frames),CONFIGS['pace13-v9-scaled'],loop_detection=True);env.load(frames)
active=np.ones(env.n,bool);states=[]
for tick in range(1600):
 if not active.any():break
 left,right=env.obs(),env.obs(1)
 if tick%4==0:states.extend([left[active].copy(),right[active].copy()])
 a=actions(net,left);b=np.zeros(env.n,dtype=int)
 for code,op in enumerate([net,old,tracker]):
  mask=np.arange(env.n)%3==code;b[mask]=actions(op,right[mask])
 terminal,_=env.step(np.stack([a,b],1),active)
 active&=terminal<0
obs=np.concatenate(states)
if len(obs)>32768:obs=obs[np.random.default_rng(2026110521).choice(len(obs),32768,replace=False)]
np.save(OUT/'observations.npy',obs)
params={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
w0,b0,w2,b2,w4,b4=[params[k] for k in ['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']]
with torch.no_grad():
 a1=net[:2](torch.from_numpy(obs)).numpy();a2=net[:4](torch.from_numpy(obs)).numpy();scores=net(torch.from_numpy(obs)).numpy()
centered=w4-w4.mean(axis=0,keepdims=True)
# Sensitivity is an audit, not labels or a fitting loss.
d2=1-a2*a2;d1=1-a1*a1
jac=np.einsum('oj,nj,ji,ni,ik->nok',centered,d2,w2,d1,w0,optimize=True)
feature=np.mean(abs(jac),axis=(0,1))
con2=np.linalg.norm(centered,axis=0)*np.std(a2,axis=0)
con1=np.std(a1,axis=0)*np.linalg.norm((np.linalg.norm(centered,axis=0)*d2.mean(0))[:,None]*w2,axis=0)
features=np.c_[w0,b0]
normalized=features/np.linalg.norm(features,axis=1,keepdims=True)
cos=abs(normalized@normalized.T);np.fill_diagonal(cos,0)
first=[]
for j in np.argsort(con1)[::-1][:16]:
 first.append(dict(neuron=int(j),weights=w0[j].tolist(),bias=float(b0[j]),importance=float(con1[j]),activation_std=float(a1[:,j].std()),neutral=float(np.tanh(b0[j]))))
report=dict(source_sha256=hashlib.sha256((ROOT/'training/results/pace-study/training/candidate/pacer/best.pt').read_bytes()).hexdigest(),states=len(obs),purpose='Learned weight/activation interpretation; no fitting',feature_names=['ball_x','ball_y','vx','vy','own_y','other_y'],mean_absolute_centered_score_jacobian=feature.tolist(),input_column_l1=np.mean(abs(w0),axis=0).tolist(),top_first_layer=first,nearest_affine_detector_cosine_quantiles=np.quantile(cos.max(1),[0,.25,.5,.75,1]).tolist(),tanh_saturation=[float((abs(a1)>.99).mean()),float((abs(a2)>.99).mean())],candidates={})

# Channel deletion preserves only neutral-input (x=0) constant contributions.
def prune(k):
 j=np.argsort(con2)[-k:];strength=np.linalg.norm(centered[:,j],axis=0)*d2[:,j].mean(0)
 ranking=np.std(a1,axis=0)*np.linalg.norm(strength[:,None]*w2[j],axis=0)
 i=np.argsort(ranking)[-k:]
 gone1=np.setdiff1d(np.arange(128),i);gone2=np.setdiff1d(np.arange(128),j)
 nh1=np.tanh(b0)
 nb2=b2[j]+w2[np.ix_(j,gone1)]@nh1[gone1]
 fullneutral=np.tanh(w2@nh1+b2)
 # Correct original neutral-output logits after replacing missing h1 channels.
 nb4=b4+w4[:,gone2]@fullneutral[gone2]
 return [w0[i],b0[i],w2[np.ix_(j,i)],nb2,w4[:,j],nb4],dict(first=i.tolist(),second=j.tolist(),basis='Activation contribution rank; copied weights with neutral-input bias compensation')

# Existing neuron votes may be combined when their affine detector is redundant.
def representatives(w,b,importance,k):
 emb=np.c_[w,b];norm=np.linalg.norm(emb,axis=1);unit=emb/np.maximum(norm[:,None],1e-12)
 sim=unit@unit.T;selected=[int(np.argmax(importance))]
 for _ in range(1,k):
  distance=1-np.max(abs(sim[:,selected]),axis=1)
  priority=importance*distance;priority[selected]=-1
  selected.append(int(np.argmax(priority)))
 idx=np.array(selected);assigned=np.argmax(abs(sim[:,idx]),axis=1);sign=np.sign(sim[np.arange(len(w)),idx[assigned]])
 return idx,assigned,sign,sim

def merge(k):
 i,assign1,sign1,sim1=representatives(w0,b0,con1,k)
 nw2=np.zeros((128,k),np.float32)
 for col in range(128):nw2[:,assign1[col]]+=sign1[col]*w2[:,col]
 j,assign2,sign2,sim2=representatives(nw2,b2,con2,k)
 nw4=np.zeros((3,k),np.float32)
 for col in range(128):nw4[:,assign2[col]]+=sign2[col]*w4[:,col]
 return [w0[i],b0[i],nw2[j],b2[j],nw4,b4],dict(first=i.tolist(),second=j.tolist(),basis='Sign-aware cosine merge of affine detectors; downstream votes summed, no coefficient fitting',first_min_assigned_cosine=float(np.min(abs(sim1[np.arange(128),i[assign1]]))),second_min_assigned_cosine=float(np.min(abs(sim2[np.arange(128),j[assign2]]))))

for k in [16,32]:
 for name,fn in [('prune',prune),('merge',merge)]:
  values,detail=fn(k)
  model=nn.Sequential(nn.Linear(6,k),nn.Tanh(),nn.Linear(k,k),nn.Tanh(),nn.Linear(k,3)).eval()
  with torch.no_grad():
   for layer,w,b in zip([model[0],model[2],model[4]],values[::2],values[1::2]):
    layer.weight.copy_(torch.from_numpy(w));layer.bias.copy_(torch.from_numpy(b))
   predicted=model(torch.from_numpy(obs)).numpy()
  label=f'{name}-{k}'
  torch.save(model.state_dict(),OUT/f'{label}.pt');export(model,OUT/f'{label}.onnx')
  detail.update(params=sum(x.numel() for x in model.parameters()),action_agreement=float(np.mean(scores.argmax(1)==predicted.argmax(1))),max_score_error=float(np.max(abs(predicted-scores))),mean_score_error=float(np.mean(abs(predicted-scores))))
  report['candidates'][label]=detail
  print(label,detail['params'],detail['action_agreement'],detail['mean_score_error'],flush=True)
(OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print('FEATURE',feature,flush=True)
print('SATURATION',report['tanh_saturation'],'COS',report['nearest_affine_detector_cosine_quantiles'],flush=True)
