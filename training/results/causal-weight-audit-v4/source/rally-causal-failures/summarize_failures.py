"""Deduplicate offline failure evidence; retain repeated-rally counts separately."""
import json,sys
from pathlib import Path
ROOT=Path.cwd();OUT=Path('/tmp/rally-causal-failures');sys.path.insert(0,str(OUT));sys.path.insert(0,str(ROOT/'training/scripts'))
import numpy as np
from audit_pacer_failures import reachability,TAB,NEXT
j=json.loads((OUT/'failure-audit.json').read_text());z=np.load(OUT/'pacer-failure-states.npz');data={k:z[k] for k in z.files};reach,catch=reachability()
for name in ['incoming_catch_possible','selected_action_preserves_catch']:
 data[name]&=data['phase']==0
data['catch_preserving_actions']&=(data['phase']==0)[:,None]
with (OUT/'pacer-failure-states.tmp.npz').open('wb') as f:np.savez_compressed(f,**data)
(OUT/'pacer-failure-states.tmp.npz').replace(OUT/'pacer-failure-states.npz')
groups={}
for r in j['failures']:
 if 'first_incoming_irreversible_row'in r:row=r['first_incoming_irreversible_row'];kind='incoming'
 elif 'first_outgoing_safe_region_loss_row'in r:row=r['first_outgoing_safe_region_loss_row'];kind='outgoing'
 else:continue
 key=(kind,data['observations'][row].tobytes())
 if key not in groups:
  if kind=='incoming':safe_actions=data['catch_preserving_actions'][row]
  else:
   dest=TAB['landing'][r['outgoing_speed'],r['opponent_contact_height'],TAB['legal'][r['opponent_contact_height']]].astype(int)
   F=int(TAB['flight'][r['outgoing_speed']]);safe=catch[F][:,dest-4].all(1)
   rem=int(data['ticks_to_contact'][row]);p=int(round(data['observations'][row,4]*80+80));safe_actions=np.array([(reach[rem-1,nxt-15]&safe).any() for nxt in NEXT[p-15]])
  groups[key]=dict(kind=kind,row=int(row),observations=data['observations'][row].tolist(),scores=data['scores'][row].tolist(),chosen=int(data['actions'][row,0]),teacher=int(data['teacher_actions'][row]),safe_actions=np.flatnonzero(safe_actions).tolist(),occurrences=0,opponents={})
  assert not safe_actions[data['actions'][row,0]]
  assert safe_actions[data['teacher_actions'][row]]
 g=groups[key];g['occurrences']+=1;g['opponents'][r['opponent']]=g['opponents'].get(r['opponent'],0)+1
critical=sorted(groups.values(),key=lambda r:(r['kind'],-r['occurrences']))
np.save(OUT/'critical-observations.npy',np.array([r['observations'] for r in critical],np.float32))
np.savez_compressed(OUT/'critical-states.npz',observations=np.array([r['observations'] for r in critical],np.float32),scores=np.array([r['scores'] for r in critical],np.float32),phase=np.array([int(r['kind']=='outgoing') for r in critical],np.int8),teacher=np.array([r['teacher'] for r in critical],np.int8),chosen=np.array([r['chosen'] for r in critical],np.int8),occurrences=np.array([r['occurrences'] for r in critical]),safe_actions=np.array([[a in r['safe_actions'] for a in range(3)] for r in critical],bool),source_row=np.array([r['row'] for r in critical]))
summary={}
for opp in ['attack','defend']:
 failures=[r for r in j['failures'] if r['opponent']==opp];a=np.array([r.get('actual_contact_risk',np.nan) for r in failures]);b=np.array([r.get('best_one_action_contact_risk',np.nan) for r in failures]);delta=np.maximum(0,a)-np.maximum(0,b)
 summary[opp]=dict(losses=len(failures),counts={name:sum(r['category']==name for r in failures) for name in j['category_codes'].values()},final_flight_contains_wall=sum(r['final_flight_wall_reflections']>0 for r in failures),actual_contact_positive_risk=int((a>0).sum()),positive_risk_reduced_by_one_contact_action=int((delta>0).sum()),positive_risk_to_nonpositive_by_one_contact_action=int(((a>0)&(b<=0)).sum()),median_positive_risk_reduction=float(np.median(delta[delta>0])))
result=dict(summary=summary,critical_states=critical,unique_critical_states=len(critical),critical_occurrences=sum(r['occurrences'] for r in critical),limitations='Repeated states are not independent examples. Geometric alternatives are offline diagnostics; neither geometry nor labels are used in Pacer inference. No universal safe region does not prove a forced loss.')
(OUT/'failure-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary));print('critical_unique',len(critical),'occurrences',result['critical_occurrences']);print(json.dumps(critical))
