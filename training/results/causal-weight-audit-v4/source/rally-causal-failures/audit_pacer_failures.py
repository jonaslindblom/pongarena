"""Offline geometry audit of original Pacer failures. No inference edits/training."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path.cwd();OUT=Path('/tmp/rally-causal-failures')
sys.path.insert(0,str(ROOT/'training/scripts'))
import numpy as np,torch,onnx
from onnx import numpy_helper
from competition_eval import load_mlp,openings,summarize
from pace_study import CONFIGS,PaceArena,make_policy,decode,tables,NEXT
from env import reflect
RULES=CONFIGS['pace13-v9-scaled'];TAB=tables(RULES)


def seed_check():
 seeds=(np.random.default_rng(2026110522).choice(10_000_000,24,replace=False)+1_160_000_000).tolist();target=set(seeds);files=0;leaves=0;overlaps=[];errors=[]
 for root in [ROOT/'training/results',ROOT/'public/training']:
  for p in sorted(root.rglob('*')):
   if not p.is_file() or p.suffix not in ('.json','.jsonl'):continue
   try:objects=[json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.suffix=='.jsonl' else [json.loads(p.read_text())]
   except Exception as error:errors.append([str(p.relative_to(ROOT)),str(error)]);continue
   files+=1
   def walk(v,route='',ctx=False):
    nonlocal leaves
    if isinstance(v,dict):
     for k,x in v.items():walk(x,route+'/'+str(k),ctx or 'seed' in str(k).lower())
    elif isinstance(v,list):
     for i,x in enumerate(v):walk(x,route+'/'+str(i),ctx)
    elif isinstance(v,int) and not isinstance(v,bool):
     leaves+=1
     if v in target:overlaps.append([str(p.relative_to(ROOT)),route,v])
    elif isinstance(v,str) and ctx and v.isdigit() and int(v) in target:overlaps.append([str(p.relative_to(ROOT)),route,int(v)])
   for obj in objects:walk(obj)
 result=dict(seeds=seeds,rng=2026110522,range=[1160000000,1170000000],files=files,integer_leaves=leaves,overlaps=overlaps,parse_errors=errors)
 (OUT/'seeds.json').write_text(json.dumps(result,indent=2)+'\n');assert not overlaps and not errors,(overlaps,errors)
 return seeds


def reachability(max_ticks=132):
 reach=np.zeros((max_ticks+1,131,131),bool);reach[0]=np.eye(131,dtype=bool)
 for t in range(1,max_ticks+1):reach[t]=reach[t-1,NEXT-15].any(1)
 cumulative=np.pad(np.cumsum(reach,axis=2,dtype=np.int16),((0,0),(0,0),(1,0)))
 y=np.arange(4,157);lo=np.maximum(15,y-15)-15;hi=np.minimum(145,y+15)-15+1
 catch=cumulative[:,:,hi]-cumulative[:,:,lo]>0
 return reach,catch


def collect(seeds,actor):
 frames,sides=openings(seeds);rows={k:[] for k in ['observations','scores','raw','actions','teacher_actions','rally_index','pair_index','contact_side','terminal_after','wall_next']};summaries={};rallies=[]
 teacher=make_policy(RULES,'attack')
 for pair_index,kind in enumerate(['attack','defend']):
  opponent=make_policy(RULES,kind);env=PaceArena(len(frames),RULES,loop_detection=True);env.load(frames)
  outcome=np.full(env.n,-1);lengths=np.zeros(env.n,int);durations=np.zeros(env.n,int);reasons=np.zeros(env.n,int)
  for tick in range(1600):
   active=outcome<0;ids=np.flatnonzero(active)
   left,right=env.obs(),env.obs(1);ours=np.where(sides[:,None]==0,left,right);theirs=np.where(sides[:,None]==0,right,left)
   with torch.no_grad():score=actor(torch.from_numpy(ours[ids])).numpy()
   ours_action=score.argmax(1);enemy_action=opponent(theirs[ids]);diagnostic=teacher(ours[ids])
   raw=np.column_stack([env.ball[ids],env.paddles[ids],env.returns[ids],env.ticks[ids],sides[ids]]).astype(np.int16)
   selected=np.ones((env.n,2),int);selected[ids,0]=np.where(sides[ids]==0,ours_action,enemy_action);selected[ids,1]=np.where(sides[ids]==0,enemy_action,ours_action)
   wall=(raw[:,1]+raw[:,3]<4)|(raw[:,1]+raw[:,3]>156)
   terminal,contact=env.step(selected,active)
   rows['observations'].append(ours[ids]);rows['scores'].append(score);rows['raw'].append(raw);rows['actions'].append(np.stack([ours_action,enemy_action],1).astype(np.int8));rows['teacher_actions'].append(diagnostic.astype(np.int8));rows['rally_index'].append((pair_index*len(frames)+ids).astype(np.int16));rows['pair_index'].append(np.full(len(ids),pair_index,np.int8));rows['contact_side'].append(contact[ids].astype(np.int8));rows['terminal_after'].append(terminal[ids].astype(np.int8));rows['wall_next'].append(wall)
   done=active&(terminal>=0);outcome[done]=terminal[done];lengths[done]=env.returns[done];durations[done]=env.ticks[done];reasons[done]=env.draw_reason[done]
   if not active.any() or (outcome>=0).all():break
  boot=np.random.default_rng(2026110522).integers(0,len(seeds),(2000,len(seeds)))
  summary=summarize(outcome,sides,lengths,durations,reasons,seeds,boot);summaries[kind]=summary
  print(kind,summary['wins'],summary['losses'],summary['draws'],flush=True)
  for i in range(env.n):rallies.append(dict(index=pair_index*env.n+i,opponent=kind,side=int(sides[i]),seed=int(seeds[(i//5)%len(seeds)]),rally=i%5+1,outcome=int(1 if outcome[i]==sides[i] else -1 if outcome[i]==1-sides[i] else 0),terminal_side=int(outcome[i]),ticks=int(durations[i]),returns=int(lengths[i]),draw_reason=int(reasons[i])))
 return {k:np.concatenate(v) for k,v in rows.items()},summaries,rallies


def classify(data,rallies,reach,catch):
 obs=data['observations'];vx,own,other,ticks,landing=decode(obs);phase=(vx>0).astype(np.int8);side=data['raw'][:,8]
 exact_can=catch[ticks,own-15,landing-4];nextp=NEXT[own-15];after=catch[(ticks-1)[:,None],nextp-15,(landing-4)[:,None]]
 chosen_can=after[np.arange(len(obs)),data['actions'][:,0]]
 data.update(phase=phase,ticks_to_contact=ticks.astype(np.int8),landing=landing.astype(np.int16),incoming_catch_possible=exact_can,selected_action_preserves_catch=chosen_can,catch_preserving_actions=after)
 eventual=np.array([r['outcome'] for r in rallies],np.int8);data['eventual_outcome']=eventual[data['rally_index']]
 data['ticks_to_end']=np.array([r['ticks'] for r in rallies],np.int16)[data['rally_index']]-data['raw'][:,7]
 loss_records=[];cause=np.full(len(obs),-1,np.int8)
 safe_cache={}
 def safe_centers(stage,h):
  key=(int(stage),int(h))
  if key not in safe_cache:
   dest=TAB['landing'][stage,h,TAB['legal'][h]].astype(int);F=int(TAB['flight'][stage])
   safe_cache[key]=catch[F,:,dest-4].all(axis=0)
  return safe_cache[key]
 for r in rallies:
  if r['outcome']!=-1:continue
  ids=np.flatnonzero(data['rally_index']==r['index']);p=phase[ids];starts=np.flatnonzero((p==0)&np.r_[True,p[:-1]!=0]);start=int(starts[-1]);lastids=ids[start:]
  flight_walls=int(data['wall_next'][lastids].sum());can_start=bool(exact_can[lastids[0]])
  mistakes=lastids[exact_can[lastids]&~chosen_can[lastids]]
  record=dict(**r,final_flight_start_row=int(lastids[0]),final_flight_ticks=int(ticks[lastids[0]]),final_landing=int(landing[lastids[0]]),paddle_at_final_flight_start=int(own[lastids[0]]),final_flight_initially_interceptable=can_start,final_flight_wall_reflections=flight_walls)
  if len(mistakes):
   first=int(mistakes[0]);record.update(first_incoming_irreversible_row=first,first_incoming_irreversible_tick=int(data['raw'][first,7]),remaining_ticks_at_error=int(ticks[first]),chosen_action=int(data['actions'][first,0]),catch_preserving_actions=np.flatnonzero(after[first]).tolist(),offline_teacher_action=int(data['teacher_actions'][first]))
   priorwalls=lastids[(lastids<first)&data['wall_next'][lastids]]
   record['ticks_since_last_wall_at_error']=int(data['raw'][first,7]-data['raw'][priorwalls[-1],7]-1) if len(priorwalls) else None
  own_hits=ids[(data['contact_side'][ids]==r['side'])&(data['raw'][ids,7]<data['raw'][lastids[0],7])]
  if len(own_hits):
   hitrow=int(own_hits[-1]);j=int(np.flatnonzero(ids==hitrow)[0]);outstart=int(ids[j+1]);outflight=ids[j+1:start]
   stage=int(abs(vx[outstart]));h=int(landing[outstart]);F=int(ticks[outstart]);center=int(own[outstart]);safe=safe_centers(stage,h)
   available=bool((reach[F,center-15]&safe).any())
   record.update(last_own_return_row=hitrow,outgoing_start_row=outstart,outgoing_ticks=F,opponent_contact_height=h,outgoing_speed=stage,own_center_after_return=center,universal_safe_center_count=int(safe.sum()),safe_recovery_reachable_after_own_return=available,actual_center_at_opponent_contact=int(own[lastids[0]]),actual_center_universally_safe=bool(safe[own[lastids[0]]-15]))
   recoverable=[];after_recoverable=[]
   for row in outflight:
    rem=int(ticks[row]);pc=int(own[row]);nxt=int(NEXT[pc-15,data['actions'][row,0]])
    recoverable.append(bool((reach[rem,pc-15]&safe).any()));after_recoverable.append(bool((reach[rem-1,nxt-15]&safe).any()))
   failures=[int(row) for row,a,b in zip(outflight,recoverable,after_recoverable) if a and not b]
   if failures:
    first=failures[0];record.update(first_outgoing_safe_region_loss_row=first,first_outgoing_safe_region_loss_tick=int(data['raw'][first,7]),outgoing_ticks_remaining_at_error=int(ticks[first]),outgoing_chosen_action=int(data['actions'][first,0]),outgoing_teacher_action=int(data['teacher_actions'][first]))
   # One-tick contact alternatives: physical risk diagnostic, no claimed win.
   contact_h=int(reflect(data['raw'][hitrow,1]+data['raw'][hitrow,3]));incoming_speed=int(abs(vx[hitrow]));centers=NEXT[own[hitrow]-15];offset=contact_h-centers;valid=np.abs(offset)<=15
   risk=np.full(3,np.inf);risk[valid]=TAB['risk'][incoming_speed,contact_h,(offset[valid]+15).astype(int)]
   actual_action=int(data['actions'][hitrow,0]);record.update(actual_contact_offset=int(offset[actual_action]),actual_contact_risk=float(risk[actual_action]),best_one_action_contact_risk=float(risk.min()),lower_risk_contact_actions=np.flatnonzero(valid&(risk<risk[actual_action]-1e-9)).tolist())
  if can_start:category=0;name='incoming_interception_error'
  elif record.get('safe_recovery_reachable_after_own_return'):category=1;name='lost_reachable_universal_recovery'
  elif record.get('universal_safe_center_count',0)==0:category=2;name='no_universal_one_reply_safe_center'
  else:category=3;name='safe_region_not_reachable_after_own_shot'
  record['category']=name;record['category_code']=category;cause[ids]=category;loss_records.append(record)
 data['loss_category']=cause
 return loss_records


def main():
 torch.set_num_threads(1);seeds=seed_check();actor=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt')
 graph=onnx.load(ROOT/'public/models/pacer-v4.onnx');tensors={w.name:numpy_helper.to_array(w) for w in graph.graph.initializer}
 for layer,i in [(actor[0],0),(actor[2],2),(actor[4],4)]:
  np.testing.assert_array_equal(layer.weight.detach().numpy(),tensors[f'w{i}']);np.testing.assert_array_equal(layer.bias.detach().numpy(),tensors[f'b{i}'])
 data,summaries,rallies=collect(seeds,actor);reach,catch=reachability();failures=classify(data,rallies,reach,catch)
 np.savez_compressed(OUT/'pacer-failure-states.npz',**data)
 np.save(OUT/'observations.npy',data['observations'])
 result=dict(purpose='Offline diagnosis only; original dense weights untouched; no geometric inference augmentation',runtime='PyTorch original checkpoint, tensors exactly equal original public Pacer ONNX',seeds=seeds,decisions=len(data['observations']),summaries=summaries,rallies=rallies,failures=failures,
  category_codes={'0':'incoming_interception_error','1':'lost_reachable_universal_recovery','2':'no_universal_one_reply_safe_center','3':'safe_region_not_reachable_after_own_shot'},
  array_schema=dict(phase='0incoming/1outgoing',raw=['x','y','vx','vy','left_paddle','right_paddle','returns','tick','candidate_side'],actions=['Pacer','opponent'],rally_index='0..239attack;240..479defend',loss_category='−1 not a loss;0..3 see categories',ticks_to_end='Pacer state tick distance to terminal'),
  interpretation='Incoming catch-loss proof holds ball flight fixed after opponent contact. Universal-recovery proof covers every geometrically legal opponent contact offset after the already-fixed Pacer shot. Empty universal sets do not prove forced losses. Contact-risk alternatives are one-reply physical diagnostics, not guaranteed winning changes.',
  hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'public/models/pacer-v4.onnx',ROOT/'training/results/pace-study/training/candidate/pacer/best.pt',ROOT/'training/scripts/pace_study.py']})
 (OUT/'failure-audit.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
 for opponent in ['attack','defend']:
  losses=[r for r in failures if r['opponent']==opponent];counts={name:sum(r['category']==name for r in losses) for name in result['category_codes'].values()}
  print(opponent,'losses',len(losses),'categories',counts,'finalflightcatchable',sum(r['final_flight_initially_interceptable'] for r in losses),'lastcontactlowerrisk',sum(bool(r.get('lower_risk_contact_actions')) for r in losses),flush=True)
 print('COMPLETE',len(data['observations']),OUT,flush=True)

if __name__=='__main__':main()
