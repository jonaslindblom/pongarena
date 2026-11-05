"""One predeclared outgoing-gated path group, centered causal gain controls."""
from pathlib import Path
import sys,json,hashlib,argparse
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
import numpy as np,torch
from competition_eval import load_mlp
from pace_study import CONFIGS,decode,tables,make_policy,NEXT
from policy import ROOT
GROUP=np.array([34,62,68]);CENTER=np.array([-1,1,-1],np.float32)
FEATURES=['x','ball_y','vx','vy','own_y','other_y']
OUT=Path('/tmp/rally-causal-features')
RULES=CONFIGS['pace13-v9-scaled']

def describe(obs,out_name,teacher_actions=None):
 torch.set_num_threads(1)
 net=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt')
 st=net.state_dict();w0,b0,w2,b2,w4,b4=[st[k].numpy().copy() for k in ['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']]
 with torch.no_grad():
  inp=torch.from_numpy(obs)
  a1=net[:2](inp).numpy();a2=net[:4](inp).numpy();scores=net(inp).numpy()
 vx,own,other,ticks,height=decode(obs);phase=obs[:,2]<0
 raw=obs.astype(float);y=np.rint(raw[:,1]*80+80).astype(int);vy=np.rint(raw[:,3]*6).astype(int)
 reflected=(y+ticks*vy<4)|(y+ticks*vy>156)
 masks={'incoming':phase,'outgoing':~phase,'incoming_straight':phase&~reflected,'incoming_reflected':phase&reflected,'outgoing_straight':~phase&~reflected,'outgoing_reflected':~phase&reflected}
 baseline=scores.argmax(1);teacher=make_policy(RULES,'attack')(obs) if teacher_actions is None else teacher_actions
 contribution=(a2[:,GROUP]-CENTER)@w4[:,GROUP].T
 raw_contribution=a2[:,GROUP]@w4[:,GROUP].T
 v=w4[2]-w4[0]
 jac=np.empty((len(obs),len(GROUP),6),np.float32)
 for f in range(6):
  jac[:,:,f]=v[GROUP][None,:]*(1-a2[:,GROUP]**2)*(((1-a1*a1)*w0[None,:,f])@w2[GROUP].T)
 table=tables(RULES);low=table['low'][abs(vx),height];high=table['high'][abs(vx),height]
 choices=NEXT[own-15];cost=np.maximum(low[:,None]-choices,choices-high[:,None])
 records={}
 for name,mask in masks.items():
  if not mask.any():continue
  base=scores[mask];bs=baseline[mask];ts=teacher[mask]
  unit_rows=[]
  for n,j in enumerate(GROUP):
   unit_rows.append(dict(neuron=int(j),mean=float(a2[mask,j].mean()),std=float(a2[mask,j].std()),max_abs_deviation_from_incoming_constant=float(np.max(abs(a2[mask,j]-CENTER[n]))),signed_own_path_derivative=float(jac[mask,n,4].mean()),abs_own_path_derivative=float(np.abs(jac[mask,n,4]).mean())))
  records[name]=dict(states=int(mask.sum()),baseline_teacher_agreement=float((bs==ts).mean()),baseline_action_shares=(np.bincount(bs,minlength=3)/len(bs)).tolist(),units=unit_rows,group_signed_feature_derivatives=np.mean(jac[mask].sum(1),axis=0).tolist(),group_mean_abs_feature_derivatives=np.mean(abs(jac[mask].sum(1)),axis=0).tolist(),controls={})
  for label,delta in [('centered_gain_0.9',-.1*contribution),('centered_gain_1.1',.1*contribution),('centered_ablation_0',-contribution),('raw_ablation_0',-raw_contribution)]:
   pred=scores+delta;actual=pred.argmax(1);aa=actual[mask];changed=aa!=bs;rr=np.arange(mask.sum())
   teacher_margin_base=base[rr,ts]-np.max(np.where(np.eye(3,dtype=bool)[ts],-np.inf,base),axis=1)
   now=pred[mask];teacher_margin_now=now[rr,ts]-np.max(np.where(np.eye(3,dtype=bool)[ts],-np.inf,now),axis=1)
   stateids=np.flatnonzero(mask);costchange=cost[stateids,aa]-cost[stateids,bs]
   records[name]['controls'][label]=dict(changed_actions=int(changed.sum()),action_flip_rate=float(changed.mean()),teacher_agreement=float((aa==ts).mean()),teacher_corrected=int(((bs!=ts)&(aa==ts)).sum()),teacher_harmed=int(((bs==ts)&(aa!=ts)).sum()),changed_neither_teacher=int((changed&(bs!=ts)&(aa!=ts)).sum()),mean_teacher_margin_change=float((teacher_margin_now-teacher_margin_base).mean()),median_teacher_margin_change=float(np.median(teacher_margin_now-teacher_margin_base)),mean_abs_score_change=float(abs(delta[mask]).mean()),maximum_abs_score_change=float(abs(delta[mask]).max()),action_shares=(np.bincount(aa,minlength=3)/len(aa)).tolist(),changed_cost_improved=int((changed&(costchange<0)).sum()),changed_cost_worsened=int((changed&(costchange>0)).sum()),changed_cost_equal=int((changed&(costchange==0)).sum()),mean_changed_worst_reply_cost_delta=float(costchange[changed].mean()) if changed.any() else None)
  # Exact bias compensation formula, not a new gate or graph.
 result=dict(purpose='Predeclared group causal intervention on fixed observations; not gameplay fitness or final evaluation',group=GROUP.tolist(),incoming_constants=CENTER.tolist(),gain_formula='W4[:,G]*=lambda; b4+=(1-lambda)*originalW4[:,G]@c; delta=(lambda-1)*(h2_G-c)@originalW4_G.T',feature_names=FEATURES,source_checkpoint_sha256=hashlib.sha256((ROOT/'training/results/pace-study/training/candidate/pacer/best.pt').read_bytes()).hexdigest(),observations=len(obs),phases=records)
 (OUT/f'{out_name}.json').write_text(json.dumps(result,indent=2)+'\n')
 for name in ['incoming','outgoing']:
  r=records[name];print(out_name,name,'states',r['states'],'teacheragree',r['baseline_teacher_agreement'],'signedown',[u['signed_own_path_derivative'] for u in r['units']],flush=True)
  for c,s in r['controls'].items():print(c,s['changed_actions'],'corrected',s['teacher_corrected'],'harmed',s['teacher_harmed'],'worstcostbetter/worse',s['changed_cost_improved'],s['changed_cost_worsened'],'delta',s['mean_changed_worst_reply_cost_delta'],flush=True)
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('observations',type=Path);parser.add_argument('--name',required=True);a=parser.parse_args()
 data=np.load(a.observations);teacher=None
 if hasattr(data,'files'):
  teacher=data['teacher_actions'] if 'teacher_actions' in data.files else None
  data=data['obs'] if 'obs' in data.files else data['observations']
 describe(data,a.name,teacher)
