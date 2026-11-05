"""Deterministic weight mutation search for the existing, entirely learned MLP.

No feature transform or action feedback is added to the exported actor. Training
fitness uses the unchanged opponents and same serves for every population member.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from competition_eval import evaluate_competition, openings, references
from env import Arena, scripted
from policy import Actor, ROOT, actions, export
from reference import RALLY_TICK_LIMIT


class Population:
    def __init__(self, states):
        self.size = len(states)
        self.layers = [(torch.stack([s[f'{i}.weight'] for s in states]).transpose(1, 2),
                        torch.stack([s[f'{i}.bias'] for s in states])[:, None]) for i in (0, 2, 4)]

    def actions(self, obs):
        # Each policy has both opponents and all openings in its own batch row.
        n = len(obs) // (2 * self.size)
        x = torch.from_numpy(obs.reshape(2, self.size, n, 6).transpose(1, 0, 2, 3).reshape(self.size, 2*n, 6))
        with torch.no_grad():
            for i, (weight, bias) in enumerate(self.layers):
                x = torch.bmm(x, weight) + bias
                if i < 2:
                    x = x.tanh()
        return x.argmax(-1).numpy().reshape(self.size, 2, n).transpose(1, 0, 2).reshape(-1)


def evaluate_population(states, seeds, reference):
    population = Population(states)
    frames, sides = openings(seeds)
    width = len(frames)
    frames = frames * (2 * population.size)
    sides = np.tile(sides, 2 * population.size)
    opponent_id = np.repeat(np.arange(2), population.size * width)
    env = Arena(len(frames), loop_detection=True)
    env.load(frames)
    outcome = np.full(env.n, -1)
    decisions = 0
    for _ in range(RALLY_TICK_LIMIT):
        active = outcome < 0
        left, right = env.obs(0), env.obs(1)
        ours = np.where(sides[:, None] == 0, left, right)
        theirs = np.where(sides[:, None] == 0, right, left)
        a, b = population.actions(ours), np.ones(env.n, dtype=np.int64)
        for code in (0, 1):
            ix = active & (opponent_id == code)
            if ix.any():
                b[ix] = actions(reference, theirs[ix]) if code == 0 else scripted(theirs[ix], True)
        selected = np.stack([np.where(sides == 0, a, b), np.where(sides == 0, b, a)], 1)
        terminal, _ = env.step(selected, active)
        done = active & (terminal >= 0)
        outcome[done] = terminal[done]
        decisions += int(active.sum())
        if (outcome >= 0).all():
            break
    assert (outcome >= 0).all()
    points = np.where(outcome == sides, 1, np.where(outcome == 1 - sides, -1, 0)).reshape(2, population.size, 2, len(seeds), 5)
    rows = []
    for i in range(population.size):
        row = {}
        for j, name in enumerate(['selfplay-v2', 'attacker']):
            q = points[j, i]
            margins = q.sum(-1)
            row[name] = dict(wins=int((q == 1).sum()), losses=int((q == -1).sum()), draws=int((q == 0).sum()),
                             point_margin=float(q.mean()), match_win_rate=float((margins > 0).mean()),
                             paired_match_win_rate=float((margins.sum(0) > 0).mean()))
        rows.append(row)
    return rows, decisions


def fitness(row):
    values = list(row.values())
    # Favor scoring and ordinary match wins against the weaker target. This is
    # training fitness only; the root's reserved protocol decides final success.
    return min(r['point_margin'] + .3*r['match_win_rate'] for r in values), float(np.mean([r['point_margin'] for r in values]))


def dev_key(result):
    rows = list(result['opponents'].values())
    wins = [r['match_wins']/(r['match_wins']+r['match_losses']+r['match_draws']) for r in rows]
    return min(wins), min(r['point_margin'] for r in rows), float(np.mean(wins))


def clone(state):
    return {k: v.clone() for k, v in state.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--initial', type=Path, default=ROOT/'training/results/training-improvement/runs/focus-outcome-stage2-s73/best.pt')
    p.add_argument('--seed', type=int, default=91)
    p.add_argument('--generations', type=int, default=30)
    p.add_argument('--pairs', type=int, default=4)
    p.add_argument('--screen-seeds', type=int, default=8)
    p.add_argument('--confirm-seeds', type=int, default=32)
    p.add_argument('--sigma', type=float, default=.003)
    p.add_argument('--scope', choices=['all', 'output'], default='all')
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    if min(args.generations, args.pairs, args.screen_seeds, args.confirm_seeds, args.eval_every) < 1:
        p.error('generation, population, seed and evaluation counts must be positive')
    if not np.isfinite(args.sigma) or args.sigma <= 0:
        p.error('sigma must be positive and finite')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if (args.out/'config.json').exists():
        raise FileExistsError('Use a fresh output directory.')
    args.out.mkdir(parents=True, exist_ok=True)
    actor, reference = Actor(), Actor()
    actor.load_state_dict(torch.load(args.initial, weights_only=True))
    reference.load_state_dict(torch.load(ROOT/'training/results/v2/ppo.pt', weights_only=True))
    actor.eval(); reference.eval()
    current = clone(actor.state_dict())
    confirmation = rng.choice(1_000_000_000, args.confirm_seeds, replace=False).tolist()
    development = (np.random.default_rng(502).choice(100_000_000, 64, replace=False)+1_300_000_000).tolist()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(rules='pong-v3', architecture=[6,64,64,3], fixed_feature_layers=False,
                  confirmation_seeds=confirmation, development_seeds=development,
                  selection='minimum ordinary five-rally match win rate; then minimum point margin; then mean match win rate')
    (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    opponents = {'selfplay-v2': reference, 'attacker': lambda obs: scripted(obs, True)}
    if args.check:
        seeds = confirmation[:3]
        population, _ = evaluate_population([current, current], seeds, reference)
        expected = evaluate_competition(actor, seeds, opponents=opponents, bootstrap_samples=10)
        for member in population:
            for name, row in member.items():
                for key in ['wins', 'losses', 'draws', 'point_margin', 'paired_match_win_rate']:
                    assert np.isclose(row[key], expected['opponents'][name][key]), (name,key,row,expected['opponents'][name])
        print('PASS: population evaluator matches ordinary evaluator and identical members.', flush=True)
    started, best, total = time.monotonic(), (-1.,-1.,-1.), 0

    def validate(generation, training=None):
        nonlocal best
        actor.load_state_dict(current)
        result = evaluate_competition(actor, development, opponents=opponents, bootstrap_samples=200)
        key = dev_key(result)
        row = dict(generation=generation, seconds=time.monotonic()-started, env_steps=total,
                   minimum_match_win_rate=key[0], minimum_point_margin=key[1], mean_match_win_rate=key[2],
                   opponents=result['opponents'], training=training)
        with (args.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if key > best:
            best=key
            torch.save(current, args.out/'best.pt'); export(actor,args.out/'best.onnx')
            (args.out/'selection.json').write_text(json.dumps(row, indent=2)+'\n')
        print(f'gen {generation} | {total:,} decisions | dev {key} | '+ ' | '.join(f'{n} {v["wins"]}/{v["losses"]}/{v["draws"]}' for n,v in result['opponents'].items()),flush=True)

    validate(0)
    for generation in range(1,args.generations+1):
        states=[current]
        for pair in range(args.pairs):
            # Explore multiple mutation magnitudes within each generation.
            scale=args.sigma * [0.5,1.,2.,4.][pair%4]
            noise={k:torch.randn_like(v)*scale if args.scope=='all' or k.startswith('4.') else torch.zeros_like(v) for k,v in current.items()}
            states += [{k:v+sign*noise[k] for k,v in current.items()} for sign in [-1,1]]
        seeds=rng.choice(1_000_000_000,args.screen_seeds,replace=False).tolist()
        screened, decisions=evaluate_population(states,seeds,reference);total+=decisions
        top=sorted(range(1,len(states)),key=lambda i:fitness(screened[i]),reverse=True)[:2]
        verified,decisions=evaluate_population([current]+[states[i] for i in top],confirmation,reference);total+=decisions
        chosen=max(range(len(verified)),key=lambda i:fitness(verified[i]))
        accepted=chosen>0 and fitness(verified[chosen])>fitness(verified[0])
        if accepted:current=states[top[chosen-1]]
        training=dict(generation=generation,accepted=accepted,confirmation=verified[chosen],fitness=fitness(verified[chosen]))
        with (args.out/'search.jsonl').open('a') as f:f.write(json.dumps(training)+'\n')
        torch.save(current,args.out/'last.pt')
        if generation%args.eval_every==0 or generation==args.generations:validate(generation,training)
        else:print(f'gen {generation} | accepted {accepted} | train {fitness(verified[chosen])}',flush=True)
    print('Finished',args.out,'best',best,flush=True)


if __name__=='__main__':main()
