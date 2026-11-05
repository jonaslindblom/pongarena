"""Fixed-candidate effects on fresh source states; diagnosis only, no fitting."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from policy import ROOT

torch.set_num_threads(1);f=np.load('/tmp/rally-causal-failures/pacer-failure-states.npz');obs=f['observations'];source=f['scores'].argmax(1);truth=f['teacher_actions'];result={}
for name in ['centered-gain-0.9','centered-gain-1.1','restore-up-96-r24','restore-down-91-r18']:
 net=load_mlp(Path('/tmp/rally-causal-root')/(name+'.pt'))
 with torch.no_grad():scores=torch.cat([net(x) for x in torch.from_numpy(obs).split(4096)]).numpy()
 actions=scores.argmax(1);changes=actions!=source;record={}
 for phase,mask in [('incoming',obs[:,2]<0),('outgoing',obs[:,2]>0)]:
  changed=changes&mask
  record[phase]={'states':int(mask.sum()),'action_flips':int(changed.sum()),'teacher_corrections':int(np.sum(changed&(actions==truth)&(source!=truth))),'teacher_harms':int(np.sum(changed&(actions!=truth)&(source==truth))),'mean_score_change':float(np.mean(np.abs(scores[mask]-f['scores'][mask])))}
 record['note']='Teacher labels are diagnostic; agreeing with the heuristic is not proof of stronger gameplay.'
 result[name]=record;print(name,record,flush=True)
Path('/tmp/rally-causal-root/fresh-candidate-effects.json').write_text(json.dumps(result,indent=2)+'\n')
