"""Causal rollout diagnosis of the one fixed braking edit; no fitting/search."""
from pathlib import Path
import sys,json,copy
ROOT=Path.cwd();OUT=Path('/tmp/rally-causal-failures/braking-diagnosis');OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'training/scripts'));sys.path.insert(0,str(OUT.parent))
import numpy as np,torch
from competition_eval import load_mlp
from audit_pacer_failures import collect,reachability,classify,TAB,NEXT

def main():
 torch.set_num_threads(1);seeds=json.load(open(OUT.parent/'seeds.json'))['seeds'];basez=np.load(OUT.parent/'pacer-failure-states.npz');base={k:basez[k] for k in basez.files};old=json.load(open(OUT.parent/'failure-audit.json'))
 baseline=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');brake=load_mlp('/tmp/rally-causal-root/phase-braking.pt')
 data,summaries,rallies=collect(seeds,brake);reach,catch=reachability();failures=classify(data,rallies,reach,catch)
 np.savez_compressed(OUT/'braking-states.npz',**data)
 def robust_actions(obs):
  raw=obs.astype(float);vx=int(round(raw[2]*6));p=int(round(raw[4]*80+80));x=int(round(raw[0]*180+180));y=int(round(raw[1]*80+80));vy=int(round(raw[3]*6));s=abs(vx)
  from env import reflect
  ticks=max(1,int(np.ceil(((x-15) if vx<0 else(345-x))/s)));h=int(reflect(y+ticks*vy));nxt=NEXT[p-15]
  if vx<0:return catch[ticks-1,nxt-15,h-4]
  F=int(TAB['flight'][s]);dest=TAB['landing'][s,h,TAB['legal'][h]].astype(int);safe=catch[F][:,dest-4].all(1)
  return np.array([(reach[ticks-1,n-15]&safe).any() for n in nxt])
 divergences=[]
 for r,newr in zip(old['rallies'],rallies):
  a=np.flatnonzero(base['rally_index']==r['index']);b=np.flatnonzero(data['rally_index']==r['index']);limit=min(len(a),len(b));different=np.flatnonzero(base['actions'][a[:limit],0]!=data['actions'][b[:limit],0])
  if not len(different):continue
  j=int(different[0]);i=int(a[j]);k=int(b[j]);np.testing.assert_array_equal(base['observations'][i],data['observations'][k]);obs=base['observations'][i];safe=robust_actions(obs)
  with torch.no_grad():h=baseline[:4](torch.from_numpy(obs[None])).numpy()[0]
  divergences.append(dict(rally=r['index'],opponent=r['opponent'],seed=r['seed'],side=r['side'],old_outcome=r['outcome'],new_outcome=newr['outcome'],delta=newr['outcome']-r['outcome'],observation=obs.tolist(),old_action=int(base['actions'][i,0]),new_action=int(data['actions'][k,0]),old_scores=base['scores'][i].tolist(),new_scores=data['scores'][k].tolist(),safe_actions=np.flatnonzero(safe).tolist(),direct_safety_gain=bool(safe[data['actions'][k,0]] and not safe[base['actions'][i,0]]),direct_safety_loss=bool(safe[base['actions'][i,0]] and not safe[data['actions'][k,0]]),h68=float(h[68]),h89=float(h[89]),ticks_to_contact=int(base['ticks_to_contact'][i]),phase=int(base['phase'][i])))
 # One concrete narrower causal edit: remove only the down vote from unit68,
 # centering at incoming h68=-1. No new units/features or fitted coefficient.
 candidate=copy.deepcopy(baseline);w=float(candidate[-1].weight[2,68])
 with torch.no_grad():candidate[-1].weight[2,68]=0;candidate[-1].bias[2]-=w
 torch.save(candidate.state_dict(),OUT/'centered-down68.pt')
 obs=base['observations'];weights=baseline[-1].weight.detach().numpy();critical=np.load(OUT.parent/'critical-states.npz')
 with torch.no_grad():
  hidden=torch.cat([baseline[:4](x) for x in torch.from_numpy(obs).split(4096)]).numpy();hc=baseline[:4](torch.from_numpy(critical['observations'])).numpy()
 proposed=base['scores'].copy();proposed[:,2]-=w*(hidden[:,68]+1)
 crit=critical['scores'].copy();crit[:,2]-=w*(hc[:,68]+1);ca=crit.argmax(1);safe=critical['safe_actions'][np.arange(len(ca)),ca]
 for r in divergences:
  score=np.array(r['old_scores']);score[2]-=w*(r['h68']+1);r['proposed_action']=int(score.argmax());r['proposed_retains_old_action']=r['proposed_action']==r['old_action'];r['proposed_action_safe']=r['proposed_action'] in r['safe_actions']
 phase=base['phase'];flips=proposed.argmax(1)!=base['actions'][:,0]
 result=dict(purpose='One existing fixed braking rollout diagnosis; one mechanistically chosen narrower path edit, no coefficient sweep or fitting',summaries=summaries,failures=failures,divergences=divergences,
  proposed=dict(path=str(OUT/'centered-down68.pt'),formula='W4[down,68]=0; b4[down] -= originalW4[down,68]; delta_down=−originalW*(h2_68+1)',original_weight=w,incoming_action_flips=int(flips[phase==0].sum()),outgoing_action_flips=int(flips[phase==1].sum()),outgoing_flip_rate=float(flips[phase==1].mean()),critical_new_actions=ca.tolist(),critical_outgoing_occurrences_safe=int(np.sum(critical['occurrences']*safe*(critical['phase']==1))),critical_incoming_occurrences_safe=int(np.sum(critical['occurrences']*safe*(critical['phase']==0)))))
 (OUT/'diagnosis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
 for opp in ['attack','defend']:
  rr=[r for r in divergences if r['opponent']==opp]
  print(opp,'diverged',len(rr),'outcome_gains',sum(r['delta']>0 for r in rr),'losses',sum(r['delta']<0 for r in rr),'directsafetygain',sum(r['direct_safety_gain'] for r in rr),'directsafetyloss',sum(r['direct_safety_loss'] for r in rr),'newrecoveryfails',sum(r['opponent']==opp and r['category']=='lost_reachable_universal_recovery' for r in failures),flush=True)
 print('PROPOSED',json.dumps(result['proposed']),flush=True)

if __name__=='__main__':main()
