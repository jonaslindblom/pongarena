"""Train raw six-input Pong v4 MLPs with DAgger, then optional PPO self-play.

The geometric teacher supplies training labels only. Exported inference is three
learned linear layers and two Tanh activations, with no fixed feature transforms.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import torch
from torch import nn

from competition_eval import load_mlp
from pace_study import CONFIGS, PaceArena, make_policy, run_pair
from policy import ROOT, actions, export, load_builtin


def actor(width):
    net = nn.Sequential(nn.Linear(6,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,3))
    for layer in net:
        if isinstance(layer,nn.Linear):
            nn.init.orthogonal_(layer.weight,gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(net[-1].weight,gain=.01)
    return net


def export_v4(net,path):
    export(net,path)
    graph = onnx.load(path)
    graph.graph.name = 'Rally Lab Pong v4 learned actor'
    props = {p.key:p.value for p in graph.metadata_props}
    props.update(rules='pong-v4',training='PyTorch DAgger',fixed_features='none')
    onnx.helper.set_model_props(graph,props)
    onnx.checker.check_model(graph)
    onnx.save(graph,path)


def broad_observations(rng,n):
    # Legal integer-valued components provide broad prediction coverage. These
    # are synthetic supervised states, not claimed reachable game trajectories.
    x = rng.integers(15,346,n)
    y = rng.integers(4,157,n)
    vx = rng.integers(5,14,n)*rng.choice([-1,1],n)
    vy = rng.integers(-9,10,n)
    own,other = rng.integers(15,146,(2,n))
    return np.stack([(x-180)/180,(y-80)/80,vx/6,vy/6,(own-80)/80,(other-80)/80],1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=ROOT/'training/results/pace-study/training/dagger-s411')
    p.add_argument('--seed',type=int,default=411)
    p.add_argument('--rounds',type=int,default=24)
    p.add_argument('--envs',type=int,default=256)
    p.add_argument('--horizon',type=int,default=256)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--eval-every',type=int,default=3)
    p.add_argument('--initial',type=Path)
    p.add_argument('--replay-rounds',type=int,default=4)
    args = p.parse_args()
    if (args.out/'config.json').exists():raise FileExistsError('Use a fresh output directory.')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rules = CONFIGS['pace13-v9-scaled']
    net = actor(args.width)
    if args.initial:
        net.load_state_dict(torch.load(args.initial,weights_only=True))
    teacher = make_policy(rules,'attack')
    defender = make_policy(rules,'defend')
    interceptor = make_policy(rules,'interceptor')
    contender = load_mlp(ROOT/'training/results/fully-learned/candidate/best.pt')
    tracker,anticipator = load_builtin('tracker'),load_builtin('anticipator')
    policies = [teacher,defender,interceptor,contender,tracker,anticipator]
    dev = (np.random.default_rng(20261130).choice(100_000_000,32,replace=False)+1_900_000_000).tolist()
    cfg = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    cfg.update(rules=vars(rules),architecture=[6,args.width,args.width,3],activation='Tanh',
        learning_rate=.001,minibatch=2048,synthetic_states_per_round=16384,
        development_seeds=dev,training_seed_interval=[0,1_000_000_000],
        loss='cross entropy on geometric teacher labels; training-only teacher, plain inference',
        selection='mean point margin versus adaptive attack, adaptive defense, original Contender, Tracker',
        sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(),ROOT/'training/scripts/pace_study.py')})
    (args.out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    env = PaceArena(args.envs,rules,seed=args.seed,loop_detection=False)
    opponent_ids = rng.integers(0,len(policies),args.envs)
    optimizer = torch.optim.Adam(net.parameters(),lr=.001)
    replay_x,replay_y = [],[]
    started,total,best = time.monotonic(),0,-2.

    def evaluate(round_index,diagnostics):
        nonlocal best
        rows = {}
        for name,opponent in [('attack',teacher),('defend',defender),('contender',contender),('tracker',tracker)]:
            rows[name] = run_pair(rules,net,opponent,dev)['summary']
        score = float(np.mean([r['point_margin'] for r in rows.values()]))
        row = dict(round=round_index,seconds=time.monotonic()-started,env_steps=total,
            development_margin=score,opponents=rows,**diagnostics)
        with (args.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        torch.save(net.state_dict(),args.out/'last.pt')
        if score > best:
            best = score
            torch.save(net.state_dict(),args.out/'best.pt')
            export_v4(net,args.out/'best.onnx')
            (args.out/'selection.json').write_text(json.dumps(row,indent=2)+'\n')
        if round_index in (3,6,12,18,24):
            torch.save(net.state_dict(),args.out/f'round-{round_index}.pt')
        print('round',round_index,'steps',total,'margin',round(score,3),
            ' | '.join(f'{name} {r["wins"]}/{r["losses"]}/{r["draws"]}' for name,r in rows.items()),flush=True)

    evaluate(0,dict(cross_entropy=None,training_accuracy=None))
    for iteration in range(1,args.rounds+1):
        xs,ys = [],[]
        beta = max(.05,1-iteration/12)
        for _ in range(args.horizon):
            obs = env.obs()
            labels = teacher(obs)
            learned = actions(net,obs)
            chosen = np.where(rng.random(args.envs)<beta,labels,learned)
            opposing = np.empty(args.envs,int)
            opposite_obs = env.obs(1)
            for code in np.unique(opponent_ids):
                mask = opponent_ids == code
                policy = policies[code]
                opposing[mask] = actions(policy,opposite_obs[mask]) if isinstance(policy,nn.Module) else policy(opposite_obs[mask])
            xs.append(obs);ys.append(labels)
            outcome,_ = env.step(np.stack([chosen,opposing],1))
            ids = np.flatnonzero(outcome>=0)
            env.reset(ids)
            opponent_ids[ids] = rng.integers(0,len(policies),len(ids))
        synthetic = broad_observations(rng,16384)
        xs.append(synthetic);ys.append(teacher(synthetic))
        replay_x.append(np.concatenate(xs));replay_y.append(np.concatenate(ys))
        replay_x,replay_y = replay_x[-args.replay_rounds:],replay_y[-args.replay_rounds:]
        x,y = torch.from_numpy(np.concatenate(replay_x)),torch.from_numpy(np.concatenate(replay_y))
        optimizer.param_groups[0]['lr'] = .001*(.4+.6*(1-(iteration-1)/args.rounds))
        loss_sum,correct,seen = 0.,0,0
        for _ in range(args.epochs):
            for ids in torch.randperm(len(x)).split(2048):
                logits = net(x[ids])
                loss = nn.functional.cross_entropy(logits,y[ids])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                optimizer.step()
                loss_sum += float(loss.detach())*len(ids)
                correct += int((logits.argmax(-1)==y[ids]).sum())
                seen += len(ids)
        total += args.envs*args.horizon
        diagnostics = dict(cross_entropy=loss_sum/seen,training_accuracy=correct/seen,teacher_action_probability=beta)
        if iteration%args.eval_every == 0 or iteration == args.rounds:evaluate(iteration,diagnostics)
        else:print('round',iteration,'CE',round(diagnostics['cross_entropy'],4),'accuracy',round(correct/seen,3),flush=True)
    (args.out/'completed.json').write_text(json.dumps(dict(seconds=time.monotonic()-started,env_steps=total,
        best_development_margin=best,rounds=args.rounds),indent=2)+'\n')


if __name__=='__main__':main()
