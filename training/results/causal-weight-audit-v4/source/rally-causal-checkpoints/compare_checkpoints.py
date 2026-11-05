"""Existing checkpoint source controls only; no fitting or weight changes."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path.cwd();OUT=Path('/tmp/rally-checkpoint-games');sys.path.insert(0,str(ROOT/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from pace_study import CONFIGS,make_policy,run_pair


def main():
 torch.set_num_threads(1);seeds=json.load(open('/tmp/rally-causal-failures/seeds.json'))['seeds'];rules=CONFIGS['pace13-v9-scaled']
 directory=ROOT/'training/results/pace-study/training';sources={f'continuation-r{r}':directory/f'dagger-continuation-s412/round-{r}.pt' for r in [3,6,12,18,24]};sources['initial-best']=directory/'dagger-s411/best.pt'
 pacer=load_mlp(directory/'candidate/pacer/best.pt');ops={'attack':make_policy(rules,'attack'),'defend':make_policy(rules,'defend'),'pacer':pacer}
 baseline=json.load(open('/tmp/rally-causal-failures/failure-audit.json'))['summaries'];rng=np.random.default_rng(2026110526);boot=rng.integers(0,len(seeds),(4000,len(seeds)))
 z=np.load('/tmp/rally-causal-failures/pacer-failure-states.npz');idx=np.random.default_rng(2026110527).choice(len(z['observations']),8192,replace=False);obs=z['observations'][idx];old_actions=z['actions'][idx,0];phase=obs[:,2]>0
 critical=np.load('/tmp/rally-causal-failures/critical-states.npz');units=[56,68,89,91,96]
 with torch.no_grad():base_h=pacer[:4](torch.from_numpy(obs)).numpy()
 result=dict(purpose='All six explicitly requested existing source controls; development comparison only, not final checkpoint selection',seeds=seeds,sources={name:dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in sources.items()},pairs={},interpretation={},complete=False)
 for name,path in sources.items():
  net=load_mlp(path)
  for opp,policy in ops.items():
   row=run_pair(rules,net,policy,seeds);result['pairs'][name+':'+opp]=row;s=row['summary'];print(name,opp,s['wins'],s['losses'],s['draws'],'median',s['median_decisive_returns'],'ci',s['point_margin_ci95'],flush=True)
   (OUT/'checkpoint-comparison.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
  with torch.no_grad():h=net[:4](torch.from_numpy(obs)).numpy();a=net(torch.from_numpy(obs)).argmax(1).numpy();hc=net[:4](torch.from_numpy(critical['observations'])).numpy();ca=net(torch.from_numpy(critical['observations'])).argmax(1).numpy();weights=net[-1].weight.numpy()
  safe=critical['safe_actions'][np.arange(len(ca)),ca]
  unitrows=[]
  for unit in units:
   corr=float(np.corrcoef(h[phase,unit],base_h[phase,unit])[0,1]) if h[phase,unit].std()>1e-8 and base_h[phase,unit].std()>1e-8 else None
   unitrows.append(dict(unit=unit,incoming_mean=float(h[~phase,unit].mean()),incoming_std=float(h[~phase,unit].std()),outgoing_mean=float(h[phase,unit].mean()),outgoing_std=float(h[phase,unit].std()),outgoing_correlation_with_final=corr,output_column=weights[:,unit].tolist(),critical_activations=hc[:,unit].tolist()))
  deltas=[]
  for opp in ['attack','defend']:
   current=result['pairs'][name+':'+opp]['summary']['per_seed'];base=baseline[opp]['per_seed'];deltas.append(np.array([v['paired_point_margin'] for v in current])/10-np.array([v['paired_point_margin'] for v in base])/10)
  delta=np.mean(deltas,axis=0);result['interpretation'][name]=dict(strong_delta_mean=float(delta.mean()),strong_delta_ci95=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),action_agreement_on_final_visited_states=float((a==old_actions).mean()),incoming_action_agreement=float((a[~phase]==old_actions[~phase]).mean()),outgoing_action_agreement=float((a[phase]==old_actions[phase]).mean()),critical_actions=ca.tolist(),critical_outgoing_safety_occurrences=int(np.sum(critical['occurrences']*safe*(critical['phase']==1))),units=unitrows)
  print(name,'strong delta',result['interpretation'][name]['strong_delta_mean'],result['interpretation'][name]['strong_delta_ci95'],'critical fixes',result['interpretation'][name]['critical_outgoing_safety_occurrences'],flush=True)
  (OUT/'checkpoint-comparison.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
 result['complete']=True;(OUT/'checkpoint-comparison.json').write_text(json.dumps(result,separators=(',',':'))+'\n')

if __name__=='__main__':main()
