"""Controlled-component response and downstream-path audit. No fitting."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from policy import ROOT
OUT=Path('/tmp/rally-causal-features')
torch.set_num_threads(1)
net=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt')
state=net.state_dict();w0,b0,w2,b2,w4,b4=[state[k].numpy().copy() for k in ['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']]
rows=[]
for direction in [-1,1]:
 for speed in [5,9,13]:
  ymax=5 if speed==5 else int(np.floor(9*speed/13+.5))
  vys=np.unique([-ymax,-max(1,ymax//2),0,max(1,ymax//2),ymax])
  for x in [15,70,125,180,235,290,345]:
   for y in [4,22,60,80,100,138,156]:
    for vy in vys:
     for own in [15,48,80,112,145]:
      for other in [15,80,145]:
       rows.append([(x-180)/180,(y-80)/80,direction*speed/6,vy/6,(own-80)/80,(other-80)/80])
obs=np.array(rows,np.float32);np.save(OUT/'controlled-grid.npy',obs)
with torch.no_grad():
 x=torch.from_numpy(obs);a1=net[:2](x).numpy();a2=net[:4](x).numpy();scores=net(x).numpy()
v=w4[2]-w4[0]
# Absolute per-path derivative of down-vs-up score through each second neuron.
J=np.empty((len(obs),128,6),np.float32)
for start in range(0,len(obs),512):
 h1=a1[start:start+512];h2=a2[start:start+512]
 for f in range(6):
  J[start:start+len(h1),:,f]=v[None,:]*(1-h2*h2)*(((1-h1*h1)*w0[None,:,f])@w2.T)
byphase={}
for name,mask in [('incoming',obs[:,2]<0),('outgoing',obs[:,2]>0)]:
 byphase[name]=dict(mean=a2[mask].mean(0),std=a2[mask].std(0),jac=np.mean(abs(J[mask]),axis=0),scores=scores[mask])
records=[]
for j in range(128):
 inc,out=byphase['incoming'],byphase['outgoing'];fraction=out['jac'][j,4]/(out['jac'][j].sum()+1e-9)
 records.append(dict(neuron=j,incoming_mean=float(inc['mean'][j]),incoming_std=float(inc['std'][j]),outgoing_mean=float(out['mean'][j]),outgoing_std=float(out['std'][j]),down_up_output_weight=float(v[j]),output_column=w4[:,j].tolist(),incoming_abs_path_jacobian=inc['jac'][j].tolist(),outgoing_abs_path_jacobian=out['jac'][j].tolist(),outgoing_own_fraction=float(fraction)))
selected=[r for r in records if r['incoming_std']<.05 and r['outgoing_std']>.3 and abs(r['incoming_mean'])>.98 and abs(r['down_up_output_weight'])>.1]
selected.sort(key=lambda r:r['outgoing_abs_path_jacobian'][4],reverse=True)
(OUT/'grid-response.json').write_text(json.dumps(dict(purpose='Controlled component-valid grid; not all observations claimed reachable',states=len(obs),feature_names=['x','y','vx','vy','own_y','other_y'],source_checkpoint_sha256=hashlib.sha256((ROOT/'training/results/pace-study/training/candidate/pacer/best.pt').read_bytes()).hexdigest(),records=records,outgoing_gated_candidates=selected),indent=2)+'\n')
print('states',len(obs),'outgoing candidates',len(selected))
for r in selected:print(r)
print('smallest incoming std',[(r['neuron'],round(r['incoming_std'],4),round(r['outgoing_std'],3)) for r in sorted(records,key=lambda r:r['incoming_std'])[:20]])
