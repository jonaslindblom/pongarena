"""Four predeclared causal dense-weight interventions, no fitting or training."""
import copy,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from pace_study import CONFIGS,make_policy,run_pair
from policy import ROOT,load_builtin

torch.set_num_threads(1)
OUT=Path('/tmp/rally-causal-root')
seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds']
source=ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'
pacer=load_mlp(source)
models={}
for gain in [.9,1.1]:
 m=copy.deepcopy(pacer)
 with torch.no_grad():
  cols=pacer[-1].weight[:,[34,62,68]].clone();c=torch.tensor([-1.,1.,-1.])
  m[-1].weight[:,[34,62,68]]=cols*gain
  m[-1].bias+=(1-gain)*(cols@c)
 models[f'centered-gain-{gain}']=m
for label,p in [('restore-up-96-r24','restore-upward-96-r24.pt'),('restore-down-91-r18','restore-downward-91-r18.pt')]:
 models[label]=load_mlp(Path('/tmp/rally-causal-checkpoints')/p)
for name,m in models.items():torch.save(m.state_dict(),OUT/(name+'.pt'))
rules=CONFIGS['pace13-v9-scaled']
ops={'attack':make_policy(rules,'attack'),'defend':make_policy(rules,'defend'),'pacer':pacer,'apprentice':load_mlp(ROOT/'training/results/pace-study/training/candidate/apprentice/best.pt'),'contender':load_mlp(ROOT/'training/results/fully-learned/candidate/best.pt'),'tracker':load_builtin('tracker')}
result={'purpose':'Development only;4causally chosen same-architecture dense weight edits; no training/optimizer/fitting','seeds':seeds,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'selection':'Mean paired-seed margin improvement vs actual attack anddefend; report weak controls and Pacer separately; no selection from finalsets','pairs':{},'comparison':{}}
for label,model in [('baseline',pacer)]+list(models.items()):
 for name,op in ops.items():
  if label=='baseline' and name=='pacer':continue
  row=run_pair(rules,model,op,seeds);result['pairs'][label+':'+name]=row
  (OUT/'development.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
  s=row['summary'];print(label,name,s['wins'],s['losses'],s['draws'],'margin',round(s['point_margin'],4),flush=True)
rng=np.random.default_rng(2026110524);idx=rng.integers(0,len(seeds),(4000,len(seeds)))
def seed_margin(row):return np.array([v['paired_point_margin'] for v in row['summary']['per_seed']])/10
for label in models:
 delta=np.mean([seed_margin(result['pairs'][label+':'+op])-seed_margin(result['pairs']['baseline:'+op]) for op in ['attack','defend']],axis=0)
 weak=np.mean([seed_margin(result['pairs'][label+':'+op])-seed_margin(result['pairs']['baseline:'+op]) for op in ['apprentice','contender','tracker']],axis=0)
 result['comparison'][label]={'strong_mean_margin_delta':float(delta.mean()),'strong_delta_ci95':np.quantile(delta[idx].mean(1),[.025,.975]).tolist(),'weak_mean_margin_delta':float(weak.mean()),'weak_delta_ci95':np.quantile(weak[idx].mean(1),[.025,.975]).tolist()}
 print('COMPARISON',label,result['comparison'][label],flush=True)
result['complete']=True;(OUT/'development.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
