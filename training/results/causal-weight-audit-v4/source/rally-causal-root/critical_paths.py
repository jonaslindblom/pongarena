"""Per-path causal attribution, not candidate selection or training."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from policy import ROOT

torch.set_num_threads(1)
net=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt')
f=np.load('/tmp/rally-causal-failures/pacer-failure-states.npz');critical=np.load('/tmp/rally-causal-failures/critical-states.npz')
obs=f['observations'];incoming=obs[:,2]<0;outgoing=~incoming
with torch.no_grad():
 h=torch.cat([net[:4](x) for x in torch.from_numpy(obs).split(4096)]).numpy()
 hc=net[:4](torch.from_numpy(critical['observations'])).numpy()
 wc=net[-1].weight.numpy();base=f['scores'];chosen=base.argmax(1)
rows=[]
for j in range(128):
 c=float(np.sign(h[incoming,j].mean()));std=float(h[incoming,j].std());dev=float(np.max(abs(h[incoming,j]-c)))
 if std>.01 or dev>.1:continue
 scores=base-(h[:,j]-c)[:,None]*wc[:,j][None]
 actions=scores.argmax(1);flips=actions!=chosen
 cs=critical['scores']-(hc[:,j]-c)[:,None]*wc[:,j][None]
 ca=cs.argmax(1);safe=critical['safe_actions'][np.arange(len(ca)),ca]
 fixes=int(np.sum(critical['occurrences']*safe*(critical['phase']==1)))
 ids=np.flatnonzero((critical['phase']==1)&safe)
 rows.append(dict(unit=j,incoming_constant=c,incoming_activation_std=std,incoming_max_deviation=dev,incoming_action_flips=int(flips[incoming].sum()),outgoing_action_flips=int(flips[outgoing].sum()),outgoing_flip_rate=float(flips[outgoing].mean()),critical_outgoing_events_fixed=fixes,critical_states_fixed=ids.tolist(),critical_new_actions=ca.tolist(),output_votes=wc[:,j].tolist()))
rows.sort(key=lambda x:(-x['critical_outgoing_events_fixed'],x['outgoing_action_flips']))
report={'purpose':'Individual centered-ablation attribution only; these ablations are NOT proposed weight improvements','observations':len(obs),'incoming':int(incoming.sum()),'outgoing':int(outgoing.sum()),'critical_outgoing_events':39,'units':rows}
Path('/tmp/rally-causal-root/critical-paths.json').write_text(json.dumps(report,indent=2)+'\n')
print('PHASESELECTIVE',len(rows))
for row in rows[:10]:print(row['unit'],'fixes',row['critical_outgoing_events_fixed'],'states',row['critical_states_fixed'],'flips incoming',row['incoming_action_flips'],'outgoing',row['outgoing_action_flips'])
