import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from pace_study import CONFIGS,make_policy,run_pair
from policy import ROOT,load_builtin

torch.set_num_threads(1)
seeds=(np.random.default_rng(2026110520).choice(10_000_000,24,replace=False)+1_150_000_000).tolist()
final=set(seeds);collisions=[];files=0

def visit(v,p):
 if isinstance(v,dict):
  for k,w in v.items():visit(k,p);visit(w,p)
 elif isinstance(v,list):
  for w in v:visit(w,p)
 elif isinstance(v,int) and not isinstance(v,bool) and v in final:collisions.append([str(p),v])
 elif isinstance(v,str) and v.isdigit() and int(v) in final:collisions.append([str(p),v])
for root in [ROOT/'training/results',ROOT/'public/training']:
 for p in root.rglob('*'):
  if p.suffix not in ['.json','.jsonl']:continue
  objs=[json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.suffix=='.jsonl' else [json.loads(p.read_text())]
  for x in objs:visit(x,p)
  files+=1
assert not collisions,collisions
policies={name:load_builtin(name) for name in ['tracker','anticipator','challenger']}
policies['pacer']=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt')
policies['attack']=make_policy(CONFIGS['pace13-v9-scaled'],'attack')
long=copy.deepcopy(policies['anticipator'])
with torch.no_grad():long[0].weight[0,3]=1.35
center=copy.deepcopy(policies['challenger'])
with torch.no_grad():center[0].weight[0,4]*=1.1
result={'purpose':'Two explained weights-only edits in the existing6-16-3ReLU architecture, no fitting','seeds':seeds,'seed_audit_files':files,'designs':{'longer-lookahead':'Anticipator w1[0,3]:.525→1.35; horizon7→18ticks','stronger-feedback':'Challenger w1[0,4]×1.1; stronger own-position feedback with every other tensor unchanged'},'pairs':{}}
for label,net in [('longer-lookahead',long),('stronger-feedback',center)]:
 for name,opp in policies.items():
  row=run_pair(CONFIGS['pace13-v9-scaled'],net,opp,seeds)
  result['pairs'][label+':'+name]=row
  Path('/tmp/rally-dense-root/edit-screen.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
  s=row['summary'];print(label,name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
