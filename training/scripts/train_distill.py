"""Distill the development-tested recovery teacher using learner-visited states.

This changes training only. Pong v3 rules and evaluation remain fixed.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from env import Arena, scripted
from policy import Actor, ROOT, actions, export
from teacher_experiment import teacher, evaluate_pair


def make_actor(widths=(6, 128, 128, 64, 3)):
    layers=[]
    for i,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
        linear=nn.Linear(a,b)
        nn.init.orthogonal_(linear.weight,gain=np.sqrt(2) if i<len(widths)-2 else .01)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if i<len(widths)-2:layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=ROOT/'training/runs/distill-71')
    p.add_argument('--seed',type=int,default=71)
    p.add_argument('--rounds',type=int,default=20)
    p.add_argument('--envs',type=int,default=256)
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--window',type=int,default=4)
    p.add_argument('--batch',type=int,default=2048)
    p.add_argument('--validation-count',type=int,default=32)
    p.add_argument('--validation-seed',type=int,default=1459)
    p.add_argument('--contact-weight',type=float,default=3)
    args=p.parse_args()
    if (args.out/'metrics.jsonl').exists():raise FileExistsError('Use a fresh output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    validation=np.random.default_rng(args.validation_seed).choice(100_000_000,args.validation_count,replace=False)+1_200_000_000
    widths=[6,128,128,64,3]
    actor=make_actor(widths)
    old=Actor();old.load_state_dict(torch.load(ROOT/'training/results/v2/ppo.pt',weights_only=True));old.eval()
    optimizer=torch.optim.Adam(actor.parameters(),lr=.001)
    env=Arena(args.envs,args.seed,loop_detection=False)
    opponents=rng.integers(0,4,args.envs)
    config=dict(**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        architecture=widths,activation='Tanh',rules='pong-v3',training_loop_detection=False,
        teacher='minimax recovery, unchanged greedy incoming attacker',
        validation_seeds=validation.tolist(),selection='maximize minimum (wins + half draws) against v2 and attacker; tie break mean',
        teacher_probability='max(0,1-(round-1)/8)',random_action_probability=.01,
        opponents=['attacker','v2','interceptor','minimax teacher'])
    (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    started=time.monotonic();best=(-1,-1);obs_buffer=[];label_buffer=[];weight_buffer=[]
    def opponent_actions(obs):
        out=np.empty(args.envs,int)
        for code in range(4):
            mask=opponents==code
            if not mask.any():continue
            out[mask]=scripted(obs[mask],True) if code==0 else actions(old,obs[mask]) if code==1 else scripted(obs[mask]) if code==2 else teacher(obs[mask])
        return out
    for iteration in range(1,args.rounds+1):
        inputs=[];labels=[];weights=[]
        beta=max(0.,1-(iteration-1)/8)
        for _ in range(args.steps):
            obs=env.obs();expert=teacher(obs);learner=actions(actor,obs)
            chosen=np.where(rng.random(args.envs)<beta,expert,learner)
            chosen=np.where(rng.random(args.envs)<.01,rng.integers(0,3,args.envs),chosen)
            flight=np.ceil((180+180*obs[:,0]-15)/np.maximum(1,-6*obs[:,2]))
            weight=np.where((obs[:,2]<0)&(flight<=12),args.contact_weight,1).astype(np.float32)
            inputs.append(obs);labels.append(expert);weights.append(weight)
            out,_=env.step(np.stack([chosen,opponent_actions(env.obs(1))],1))
            done=np.flatnonzero(out>=0);env.reset(done);opponents[done]=rng.integers(0,4,len(done))
        obs_buffer.append(np.concatenate(inputs));label_buffer.append(np.concatenate(labels));weight_buffer.append(np.concatenate(weights))
        obs_buffer=obs_buffer[-args.window:];label_buffer=label_buffer[-args.window:];weight_buffer=weight_buffer[-args.window:]
        x=torch.from_numpy(np.concatenate(obs_buffer));y=torch.from_numpy(np.concatenate(label_buffer));w=torch.from_numpy(np.concatenate(weight_buffer))
        optimizer.param_groups[0]['lr']=.001*(1-.7*(iteration-1)/max(1,args.rounds-1))
        for _ in range(args.epochs):
            for ix in torch.randperm(len(x)).split(args.batch):
                per=nn.functional.cross_entropy(actor(x[ix]),y[ix],reduction='none')
                loss=(per*w[ix]).sum()/w[ix].sum()
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
        evals={name:evaluate_pair(lambda o:actions(actor,o),opponent,validation) for name,opponent in
               [('attacker',lambda o:scripted(o,True)),('v2',lambda o:actions(old,o))]}
        scores=[(r['wins']+.5*r['draws'])/r['rallies'] for r in evals.values()]
        key=(min(scores),float(np.mean(scores)))
        with torch.no_grad():
            probe=torch.randperm(len(x))[:min(8192,len(x))]
            accuracy=float((actor(x[probe]).argmax(-1)==y[probe]).float().mean())
        row=dict(round=iteration,env_steps=iteration*args.steps*args.envs,seconds=time.monotonic()-started,
                 teacher_probability=beta,training_accuracy=accuracy,loss=float(loss.detach()),selection_min=key[0],selection_mean=key[1],opponents=evals)
        with (args.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        torch.save(actor.state_dict(),args.out/'last.pt')
        if key>best:
            best=key;torch.save(actor.state_dict(),args.out/'best.pt')
            (args.out/'selection.json').write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    actor.load_state_dict(torch.load(args.out/'best.pt',weights_only=True));export(actor,args.out/'best.onnx')
    print('Finished',args.out,'best',best,'seconds',time.monotonic()-started,flush=True)


if __name__=='__main__':main()
