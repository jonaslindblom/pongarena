"""Matched rule comparison; NN is the frozen v1 Selfplay actor. See balance-v2/README.md."""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch

from env import Arena, scripted
from reference import serve
from policy import Actor, actions, load_builtin, ROOT

torch.set_num_threads(1)
model=Actor();model.load_state_dict(torch.load(ROOT / 'training/results/seed7/ppo.pt',weights_only=True));model.eval()
tracker=load_builtin('tracker')

def run(cap, pair, start, count):
    frames=[];sides=[]
    for side in (0,1):
        for seed in range(start,start+count):
            f=None
            for _ in range(5):
                f=serve(seed,f);frames.append(f);sides.append(side)
    sides=np.array(sides);env=Arena(len(frames), max_vx=cap);env.load(frames)
    initial=env.ball.copy();receiver=(initial[:,2]>0).astype(int)
    outcome=np.full(len(frames),-1);lengths=np.zeros(len(frames),int);duration=lengths.copy()
    histories=[[],[]]
    def decide(kind, obs, side):
        if kind=='A':return scripted(obs,True,max_vx=cap)
        if kind=='D':return scripted(obs,max_vx=cap)
        if kind=='N':return actions(model,obs)
        if kind=='T':return actions(tracker,obs)
        # Reaction-delay proxy: exact predicted interception on a five-tick-old observation.
        if kind=='H':
            histories[side].append(obs.copy())
            return scripted(histories[side][max(0,len(histories[side])-6)], max_vx=cap)
        raise ValueError(kind)
    for _ in range(800):
        left,right=env.obs(),env.obs(1)
        ours=np.where(sides[:,None]==0,left,right);theirs=np.where(sides[:,None]==0,right,left)
        a,b=decide(pair[0],ours,0),decide(pair[1],theirs,1)
        terminal,hit=env.step(np.stack([np.where(sides==0,a,b),np.where(sides==0,b,a)],1))
        done=(outcome<0)&(terminal>=0)
        outcome[done]=terminal[done];lengths[done]=env.returns[done];duration[done]=env.ticks[done]
        if (outcome>=0).all():break
    drawn=outcome==2;decisive=~drawn
    return dict(cap=cap,pair=pair,n=len(frames),wins=int((outcome==sides).sum()),losses=int((outcome==1-sides).sum()),draws=int(drawn.sum()),median_decisive_returns=float(np.median(lengths[decisive])) if decisive.any() else None,
      early_rate=float(((lengths<2)&decisive).mean()),long_scored_rate=float(((lengths>=6)&decisive).mean()),
      first_receiver_win_rate=float((outcome[decisive]==receiver[decisive]).mean()) if decisive.any() else None,mean_ticks=float(duration.mean()))

p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=60);p.add_argument('--start',type=int,default=3_000_000_000);p.add_argument('--caps',type=int,nargs='+',default=[15,14,13,12,11,10]);p.add_argument('--pairs',nargs='+',default=['AA','AD','AN','NN','AT','AH']);p.add_argument('--out',default='training/results/balance-v2/scan.json');args=p.parse_args()
results=[]
for cap in args.caps:
    for pair in args.pairs:
        r=run(cap,pair,args.start,args.count);results.append(r);print(json.dumps(r),flush=True)
Path(args.out).write_text(json.dumps({'seed_start':args.start,'seed_count':args.count,'rows':results},indent=2)+'\n')
