"""One late-recovery hypothesis using an existing learned phase feature."""
import sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from pace_study import CONFIGS,run_pair,make_policy
from policy import ROOT,load_builtin

torch.set_num_threads(1);out=Path('/tmp/rally-causal-root')
pacer=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');m=copy.deepcopy(pacer)
with torch.no_grad():m[-1].weight[1,89]-=.5;m[-1].bias[1]+=.5
torch.save(m.state_dict(),out/'phase-braking.pt')
seeds=json.load(open('/tmp/rally-causal-failures/seeds.json'))['seeds'];rules=CONFIGS['pace13-v9-scaled']
ops={'attack':make_policy(rules,'attack'),'defend':make_policy(rules,'defend'),'pacer':pacer,'apprentice':load_mlp(ROOT/'training/results/pace-study/training/candidate/apprentice/best.pt'),'contender':load_mlp(ROOT/'training/results/fully-learned/candidate/best.pt'),'tracker':load_builtin('tracker')}
r={'purpose':'One manually fixed two-weight braking edit after causal late-recovery audit; no coefficient sweep/fitting/training','formula':'W4[stay,89]-=.5; b4[stay]+=.5; delta_stay=.5*(1-h2_89)','seeds':seeds,'pairs':{}}
for name,op in ops.items():
 row=run_pair(rules,m,op,seeds);r['pairs'][name]=row;(out/'braking-development.json').write_text(json.dumps(r,separators=(',',':'))+'\n');s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
old=json.load(open(out/'development.json'));idx=np.random.default_rng(2026110524).integers(0,len(seeds),(4000,len(seeds)))
def sm(row):return np.array([x['paired_point_margin'] for x in row['summary']['per_seed']])/10
for label,roster in [('strong',['attack','defend']),('weak',['apprentice','contender','tracker'])]:
 delta=np.mean([sm(r['pairs'][p])-sm(old['pairs']['baseline:'+p]) for p in roster],axis=0)
 r[label+'_delta']={'mean':float(delta.mean()),'ci95':np.quantile(delta[idx].mean(1),[.025,.975]).tolist()};print(label,r[label+'_delta'],flush=True)
r['complete']=True;(out/'braking-development.json').write_text(json.dumps(r,separators=(',',':'))+'\n')
