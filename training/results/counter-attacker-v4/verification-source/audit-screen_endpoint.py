"""Three fixed, physical recovery hypotheses. No fitting, optimization, mutation."""
import json
from pathlib import Path
import sys,time
ROOT=Path('/Users/jonas/.codex-workspaces/worktrees/ios-36683f90-4afb-46b3-b505-f5d7b6d64528/weights')
sys.path.insert(0,str(ROOT/'training/scripts'))
import numpy as np
import torch
from pace_study import CONFIGS,tables,decode,make_policy,run_pair,NEXT
from offense_eval import contact_paths
from policy import OnnxPolicy
from competition_eval import load_mlp
OUT=Path('/tmp/rally-attacker-audit')
RULES=CONFIGS['pace13-v9-scaled'];T=tables(RULES);PATHS=contact_paths()


def recovery_policy(mode,incoming='attack'):
    base=make_policy(RULES,incoming)
    def policy(obs):
        result=base(obs)
        vx,own,other,ticks,h=decode(obs)
        ids=np.flatnonzero(vx>0)
        if not len(ids):return result
        hh=h[ids];p=other[ids];t=ticks[ids];speed=np.abs(vx[ids])
        centers=T['centers'][hh]
        first=PATHS[t[:,None],p[:,None]-15,np.clip(centers-15,0,130),3]
        valid=T['legal'][hh]&(first>=0)
        destinations=T['landing'][speed,hh]
        F=T['flight'][speed]
        if mode=='reachable_mid':
            low=np.maximum(15,np.max(np.where(valid,destinations-15-2*F[:,None],-10000),axis=1))
            high=np.minimum(145,np.min(np.where(valid,destinations+15+2*F[:,None],10000),axis=1))
            target=(low+high)/2
        elif mode=='predicted_endpoint':
            expected=np.clip(T['mid'][speed,hh],own[ids]-2*np.maximum(0,t-1),own[ids]+2*np.maximum(0,t-1))
            lowest=np.min(np.where(valid,destinations,10000),axis=1)
            highest=np.max(np.where(valid,destinations,-10000),axis=1)
            target=np.where(np.abs(highest-expected)>np.abs(lowest-expected),highest,lowest)
        elif mode=='predicted_attack':
            expected=np.clip(T['mid'][speed,hh],own[ids]-2*np.maximum(0,t-1),own[ids]+2*np.maximum(0,t-1))
            margin=np.abs(destinations-expected[:,None])-15-2*F[:,None]-2
            quality=margin-.05*np.maximum(0,T['risk'][speed,hh])-.001*np.abs(centers-p[:,None])
            best=np.argmax(np.where(valid,quality,-1e12),axis=1)
            target=destinations[np.arange(len(ids)),best]
        else:raise ValueError(mode)
        target=np.where(valid.any(1),target,80)
        result[ids]=np.argmin(np.abs(NEXT[own[ids]-15]-target[:,None])+[1e-6,0,1e-6],axis=1)
        return result.astype(np.int64)
    policy.description=dict(incoming=incoming,outgoing=mode,stateless=True)
    return policy


def main():
    torch.set_num_threads(1)
    seeds=(np.random.default_rng(2026110511).choice(50_000_000,24,replace=False)+4_100_000_000).tolist()
    opponents={'attack':make_policy(RULES,'attack'),'defend':make_policy(RULES,'defend'),
      'pacer':load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'),
      'craft_small':OnnxPolicy(ROOT/'training/results/compact-handcrafted-v4/handcrafted-small-v4.onnx')}
    candidates={'attack_predicted_endpoint':recovery_policy('predicted_endpoint','attack')}
    result=dict(method='Fourth predefined recovery hypothesis; endpoint forecast drops per-shot risk and travel; incoming original teacher logic unchanged; no fitting or selection loop',
      seeds=seeds,seed_generation='NumPy2026110511 choice50e6 withoutreplacement +4100000000',
      formulas=dict(reachable_mid='Restrict enemy legal contact centers to exact reachable set in remaining flight; midpoint of resulting landing coverage interval',
      predicted_attack='Predict greedy adaptive attacker chosen contact using its exact current rule and aim recovery at resulting next landing; opponent may later revise'),pairs={})
    for name,candidate in candidates.items():
        for opponent_name,opponent in opponents.items():
            start=time.monotonic();row=run_pair(RULES,candidate,opponent,seeds)
            result['pairs'][name+':'+opponent_name]=row
            (OUT/'endpoint-screen.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
            s=row['summary'];print(name,opponent_name,s['wins'],s['losses'],s['draws'],'CI',s['point_margin_ci95'],'paired',s['paired_match_wins'],s['paired_match_losses'],s['paired_match_draws'],'seconds',time.monotonic()-start,flush=True)
    result['complete']=True
    (OUT/'endpoint-screen.json').write_text(json.dumps(result,separators=(',',':'))+'\n')

if __name__=='__main__':main()
