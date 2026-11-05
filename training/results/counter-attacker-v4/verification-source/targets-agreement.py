exec(open('/tmp/rally-attacker-targets/band_policy.py').read().split("if __name__=='__main__':")[0])
from competition_eval import openings
from pace_study import PaceArena
from policy import OnnxPolicy
attack=make_policy(rules,'attack')
frames,sides=openings(seeds);env=PaceArena(len(frames),rules,loop_detection=True);env.load(frames)
active=np.ones(len(frames),bool)
counts=np.zeros((3,3),int);bytime={};examples=[];total=0
for step in range(1600):
 obs=env.obs();otherobs=env.obs(1)
 a,b=attack(obs),attack(otherobs)
 if step%4==0:
  vx,own,opp,t,h=decode(obs);sel=active&(vx<0)
  if sel.any():
   pred=band_policy(obs)
   for aa,bb in zip(a[sel],pred[sel]):counts[aa,bb]+=1
   total+=int(sel.sum())
   for key,mask in [('t1',t==1),('t2to5',(t>=2)&(t<=5)),('t6to15',(t>=6)&(t<=15)),('t16plus',t>=16)]:
    mask=mask&sel;n=int(mask.sum());e=int((a[mask]!=pred[mask]).sum())
    if key not in bytime:bytime[key]=dict(states=0,disagreements=0)
    bytime[key]['states']+=n;bytime[key]['disagreements']+=e
   for idx in np.flatnonzero(sel&(a!=pred))[:max(0,20-len(examples))]:
    examples.append(dict(obs=obs[idx].tolist(),exact=int(a[idx]),band=int(pred[idx]),ticks=int(t[idx]),height=int(h[idx]),own=int(own[idx]),stage=int(abs(vx[idx]))))
 outcome,_=env.step(np.stack([a,b],1),active);active&=outcome<0
 if not active.any():break
result=json.loads(Path('/tmp/rally-attacker-targets/band-results.json').read_text())
result['incoming_agreement']=dict(scope='Active left-player incoming states every4ticks in exact attacker mirrors on development seeds',states=total,disagreements=int(counts.sum()-counts.trace()),agreement=float(counts.trace()/counts.sum()),confusion_exact_rows_band_cols=counts.tolist(),by_time=bytime,examples=examples)
print(json.dumps(result['incoming_agreement'],indent=2),flush=True)
small=OnnxPolicy(ROOT/'training/results/compact-handcrafted-v4/handcrafted-small-v4.onnx')
row=run_pair(rules,band_policy,small,seeds);result['pairs']['craft-small']=row
Path('/tmp/rally-attacker-targets/band-results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
s=row['summary'];print('craft-small',s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
