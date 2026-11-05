from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from policy import ROOT
OUT=Path('/tmp/rally-causal-features');torch.set_num_threads(1)
net=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt')
st=net.state_dict();w0,b0,w2,b2,w4,b4=[st[k].numpy().copy() for k in ['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']]
obs=np.load(OUT/'controlled-grid.npy');G=[34,62,68];v=w4[2]-w4[0]
with torch.no_grad():
 x=torch.from_numpy(obs);a1=net[:2](x).numpy();z2=net[2](net[:2](x)).numpy();a2=np.tanh(z2)
mask=obs[:,2]>0;d1=1-a1*a1;d2=1-a2*a2
record=dict(group=G,scope='Controlled component grid',units=[],feature_sweeps={})
for j in G:
 path=(v[j]*d2[mask,j])[:,None]*w2[j,None,:]*d1[mask]*w0[None,:,4]
 importance=abs(path).mean(0)
 ids=np.argsort(importance)[-6:][::-1]
 record['units'].append(dict(second_neuron=j,down_minus_up_output_weight=float(v[j]),incoming_preactivation_range=[float(z2[~mask,j].min()),float(z2[~mask,j].max())],outgoing_preactivation_range=[float(z2[mask,j].min()),float(z2[mask,j].max())],strongest_first_layer_own_paths=[dict(first_neuron=int(i),mean_abs_own_path=float(importance[i]),second_layer_weight=float(w2[j,i]),first_weights=w0[i].tolist(),first_bias=float(b0[i])) for i in ids]))
for f,name,values in [(0,'x',np.arange(15,346,15)),(1,'ball_y',np.arange(4,157,8)),(3,'vy',np.arange(-6,7)),(4,'own_y',np.arange(15,146,5)),(5,'other_y',np.arange(15,146,5))]:
 rows=[]
 for direction in [-1,1]:
  base=np.zeros((len(values),6),np.float32);base[:,2]=direction*9/6
  if f==0:base[:,f]=(values-180)/180
  elif f==3:base[:,f]=values/6
  else:base[:,f]=(values-80)/80
  with torch.no_grad():h2=net[:4](torch.from_numpy(base)).numpy()[:,G]
  centered=(h2-np.array([-1,1,-1]))*v[G]
  rows.append(dict(direction=direction,values=values.tolist(),group_activations=h2.tolist(),per_unit_centered_direction_contribution=centered.tolist(),summed_centered_direction_contribution=centered.sum(1).tolist()))
 record['feature_sweeps'][name]=rows
(OUT/'downstream-path-evidence.json').write_text(json.dumps(record,indent=2)+'\n')
for r in record['units']:
 print('second',r['second_neuron'],'incoming z',r['incoming_preactivation_range'],'outgoing z',r['outgoing_preactivation_range'],'firstpathids',[a['first_neuron'] for a in r['strongest_first_layer_own_paths']],flush=True)
