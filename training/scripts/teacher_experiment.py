"""Development-only search for stronger teachers; physics and opponents stay fixed."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from env import Arena, reflect, scripted
from policy import Actor, ROOT, actions
from reference import serve


def teacher_targets(obs, max_offset=15):
    """Physical desired paddle centers for the same minimax teacher decisions."""
    x = np.rint(obs[:, 0].astype(np.float64) * 180 + 180)
    y = np.rint(obs[:, 1].astype(np.float64) * 80 + 80)
    vx = np.rint(obs[:, 2].astype(np.float64) * 6)
    vy = np.rint(obs[:, 3].astype(np.float64) * 6)
    own = np.rint(obs[:, 4].astype(np.float64) * 80 + 80)
    other = np.rint(obs[:, 5].astype(np.float64) * 80 + 80)
    ticks = np.maximum(1, np.ceil((x-15)/np.maximum(1,-vx)))
    landing = reflect(y+ticks*vy)
    moves = np.arange(-65,66)[None,:]
    centers = np.clip(own[:,None]+2*moves,15,145)
    offset = landing[:,None]-centers
    sx = 5+np.floor(8*np.abs(offset)/15+.5)
    sy = np.clip(np.floor(offset/3+.5),-5,5)
    flight = np.ceil(330/sx)
    destination = reflect(landing[:,None]+flight*sy)
    margin = np.abs(destination-other[:,None])-15-2*flight
    valid = (np.abs(moves)<=ticks[:,None])&(np.abs(offset)<=max_offset)
    quality = np.where(valid,margin-.001*np.abs(centers-own[:,None]),-1e9)
    incoming = np.where(valid.any(axis=1),centers[np.arange(len(obs)),quality.argmax(1)],np.clip(landing,15,145))
    ticks = np.maximum(1,np.ceil((345-x)/np.maximum(1,vx)))
    landing = reflect(y+ticks*vy)
    offsets = np.arange(-15,16)[None,:]
    centers = landing[:,None]-offsets
    sx = 5+np.floor(8*np.abs(offsets)/15+.5)
    sy = np.floor(offsets/3+.5)
    flight = np.ceil(330/sx)
    destination = reflect(landing[:,None]+flight*sy)
    valid = (centers>=15)&(centers<=145)&(np.abs(centers-other[:,None])<=2*ticks[:,None])
    low = np.max(np.where(valid,destination-15-2*flight,-1e9),axis=1)
    high = np.min(np.where(valid,destination+15+2*flight,1e9),axis=1)
    outgoing = np.where(valid.any(axis=1),np.clip((low+high)/2,15,145),80)
    return np.where(vx>0,outgoing,incoming)


def target_teacher(obs, max_offset=15):
    target = teacher_targets(obs, max_offset)
    own = np.rint(obs[:,4].astype(np.float64)*80+80)
    error = target-own
    return np.where(error < -1,0,np.where(error > 1,2,1)).astype(np.int64)


def teacher(obs, recovery='minimax'):
    selected = scripted(obs, attack=True)
    if recovery == 'center':
        return selected
    x = np.rint(obs[:, 0].astype(np.float64) * 180 + 180)
    y = np.rint(obs[:, 1].astype(np.float64) * 80 + 80)
    vx = np.rint(obs[:, 2].astype(np.float64) * 6)
    vy = np.rint(obs[:, 3].astype(np.float64) * 6)
    own = np.rint(obs[:, 4].astype(np.float64) * 80 + 80)
    other = np.rint(obs[:, 5].astype(np.float64) * 80 + 80)
    outgoing = vx > 0
    ticks = np.maximum(1, np.ceil((345-x)/np.maximum(1, vx)))
    landing = reflect(y + ticks*vy)
    offsets = np.arange(-15,16)[None,:]
    centers = landing[:,None]-offsets
    sx = 5+np.floor(8*np.abs(offsets)/15+.5)
    sy = np.floor(offsets/3+.5)
    flight = np.ceil(330/sx)
    destination = reflect(landing[:,None]+flight*sy)
    valid = (centers>=15)&(centers<=145)&(np.abs(centers-other[:,None])<=2*ticks[:,None])
    low = np.max(np.where(valid,destination-15-2*flight,-1e9),axis=1)
    high = np.min(np.where(valid,destination+15+2*flight,1e9),axis=1)
    target = np.clip((low+high)/2,15,145)
    if recovery == 'nearest-safe':
        target = np.clip(own, np.minimum(low,high), np.maximum(low,high))
    target = np.where(valid.any(axis=1), target,80)
    error=target-own
    recovering = np.where(error < -1,0,np.where(error>1,2,1))
    return np.where(outgoing,recovering,selected).astype(np.int64)


def evaluate_pair(actor, opponent, seeds):
    frames=[]; sides=[]
    for side in (0,1):
        for seed in seeds:
            f=None
            for _ in range(5):
                f=serve(int(seed),f);frames.append(f);sides.append(side)
    sides=np.array(sides)
    env=Arena(len(frames),loop_detection=True);env.load(frames)
    outcome=np.full(len(frames),-1);reasons=np.zeros(len(frames),int)
    for _ in range(1600):
        active=outcome<0
        left,right=env.obs(),env.obs(1)
        ours=np.where(sides[:,None]==0,left,right);theirs=np.where(sides[:,None]==0,right,left)
        a,b=actor(ours),opponent(theirs)
        terminal,_=env.step(np.stack([np.where(sides==0,a,b),np.where(sides==0,b,a)],1),active)
        done=active&(terminal>=0)
        outcome[done]=terminal[done];reasons[done]=env.draw_reason[done]
        if not (outcome<0).any():break
    won=outcome==sides;lost=outcome==1-sides;drawn=outcome==2
    margins=(won.astype(int)-lost.astype(int)).reshape(2,len(seeds),5).sum(-1)
    paired=margins.sum(0)
    return dict(rallies=len(frames),wins=int(won.sum()),losses=int(lost.sum()),draws=int(drawn.sum()),
        loop_draws=int((reasons==1).sum()),timeout_draws=int((reasons==2).sum()),
        match_wins=int((margins>0).sum()),match_losses=int((margins<0).sum()),match_draws=int((margins==0).sum()),
        paired_wins=int((paired>0).sum()),paired_losses=int((paired<0).sum()),paired_draws=int((paired==0).sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=60);p.add_argument('--seed',type=int,default=1217)
    p.add_argument('--out',type=Path,default=ROOT/'training/runs/teacher-experiment/recovery.json')
    args=p.parse_args();torch.set_num_threads(1)
    seeds=np.random.default_rng(args.seed).choice(100_000_000,args.count,replace=False)+1_200_000_000
    v2=Actor();v2.load_state_dict(torch.load(ROOT/'training/results/v2/ppo.pt',weights_only=True));v2.eval()
    opponents={'attacker':lambda x:scripted(x,True),'v2':lambda x:actions(v2,x)}
    rows=[]
    for recovery in ['center','minimax','nearest-safe']:
        for name,opponent in opponents.items():
            row=dict(recovery=recovery,opponent=name,**evaluate_pair(lambda x:teacher(x,recovery),opponent,seeds))
            rows.append(row);print(json.dumps(row),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(development_only=True,seeds=seeds.tolist(),rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
