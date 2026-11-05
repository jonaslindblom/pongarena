import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import torch,numpy as np
from competition_eval import load_mlp
from policy import ROOT,load_builtin
from pace_study import CONFIGS,run_pair,make_policy
OUT=Path('/tmp/rally-dense-transfer');torch.set_num_threads(1)
seeds=json.loads((OUT/'seed-audit.json').read_text())['seeds'];rules=CONFIGS['pace13-v9-scaled']
ops={'tracker':load_builtin('tracker'),'anticipator':load_builtin('anticipator'),'challenger':load_builtin('challenger'),'pacer':load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'),'attack':make_policy(rules,'attack')}
result={'purpose':'Four fixed transfer candidates, no fitting; all outcomes retained','seeds':seeds,'pairs':{}}
for label in ['prune-16','merge-16','prune-32','merge-32']:
 model=load_mlp(OUT/f'{label}.pt')
 for name,op in ops.items():
  row=run_pair(rules,model,op,seeds);result['pairs'][label+':'+name]=row
  (OUT/'screen.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
  s=row['summary'];print(label,name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
result['complete']=True;(OUT/'screen.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
for name in ['tracker','anticipator','challenger']:
 model=load_builtin(name)
 print('BUILTIN',name)
 for k,v in model.state_dict().items():
  arr=v.numpy();print(k,arr.tolist())
