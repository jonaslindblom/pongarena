"""Fixed-gain causal controls at proven recovery-option discard decisions."""
from pathlib import Path
import sys,json,importlib.util
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from pace_study import CONFIGS,tables,NEXT,decode
from policy import ROOT
OUT=Path('/tmp/rally-causal-features');source=Path('/tmp/rally-causal-failures')
meta=json.loads((source/'failure-audit.json').read_text());data=np.load(source/'pacer-failure-states.npz')
events=[f for f in meta['failures'] if 'first_outgoing_safe_region_loss_row' in f]
rows=np.array([f['first_outgoing_safe_region_loss_row'] for f in events])
obs=data['observations'][rows];scores=data['scores'][rows];base=data['actions'][rows,0]
torch.set_num_threads(1);net=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt')
with torch.no_grad():a2=net[:4](torch.from_numpy(obs)).numpy()
w4=net[4].weight.detach().numpy();G=[34,62,68];c=np.array([-1,1,-1])
contrib=(a2[:,G]-c)@w4[:,G].T
spec=importlib.util.spec_from_file_location('failure_reach',source/'audit_pacer_failures.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
reach,catch=m.reachability();tab=tables(CONFIGS['pace13-v9-scaled'])
vx,own,other,ticks,landing=decode(obs)
preserves=[]
for i,event in enumerate(events):
 stage=event['outgoing_speed'];h=event['opponent_contact_height'];dest=tab['landing'][stage,h,tab['legal'][h]].astype(int);flight=int(tab['flight'][stage])
 safe=catch[flight,:,dest-4].all(axis=0)
 options=(reach[ticks[i]-1,NEXT[own[i]-15]-15]&safe).any(1)
 assert not options[base[i]]
 assert options.any()
 preserves.append(options)
preserves=np.array(preserves)
result=dict(purpose='One-step causal gain checks at originally proven first irreversible recovery-option losses; not outcome counterfactual',group=G,events=len(events),by_opponent={},controls={})
for name,delta in [('gain0.9',-.1*contrib),('gain1.1',.1*contrib),('ablation0',-contrib)]:
 action=(scores+delta).argmax(1);fixed=preserves[np.arange(len(events)),action]
 result['controls'][name]=dict(changed=int((action!=base).sum()),restored_safe_recovery_option=int(fixed.sum()),fixed_event_indices=[int(events[i]['index']) for i in np.flatnonzero(fixed)],by_opponent={op:dict(events=sum(e['opponent']==op for e in events),fixed=sum(bool(fixed[i]) for i,e in enumerate(events) if e['opponent']==op)) for op in ['attack','defend']})
 result['by_opponent']={op:sum(e['opponent']==op for e in events) for op in ['attack','defend']}
(OUT/'critical-recovery-interventions.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
