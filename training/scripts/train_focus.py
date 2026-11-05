"""Focused PPO fine-tuning under frozen Pong v3 rules.

Targets are fixed for entire rallies. The actor retains six inputs; only the
training critic sees opponent identity and remaining time.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from env import Arena, scripted
from policy import Actor, ROOT, actions, export


def checkpoint_actor(path):
    model = Actor()
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=71)
    p.add_argument('--updates', type=int, default=300)
    p.add_argument('--envs', type=int, default=256)
    p.add_argument('--horizon', type=int, default=256)
    p.add_argument('--gamma', type=float, default=1.)
    p.add_argument('--gae-lambda', type=float, default=.97)
    p.add_argument('--return-reward', type=float, default=0.)
    p.add_argument('--logit-scale', type=float, default=.5)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--initial', type=Path, default=ROOT / 'training/results/v2/ppo.pt')
    args = p.parse_args()
    if (args.out / 'metrics.jsonl').exists():
        raise FileExistsError('Use a fresh output directory')
    if min(args.updates, args.envs, args.horizon, args.eval_every) < 1:
        p.error('counts must be positive')
    if not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        p.error('gamma and GAE lambda must be in [0, 1]')
    if not all(math.isfinite(v) for v in [args.logit_scale, args.learning_rate, args.return_reward]) or min(args.logit_scale, args.learning_rate) <= 0:
        p.error('logit scale and learning rate must be positive and all reward/scaling values finite')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    actor = checkpoint_actor(args.initial)
    with torch.no_grad():
        actor[-1].weight.mul_(args.logit_scale)
        actor[-1].bias.mul_(args.logit_scale)
    opponent = checkpoint_actor(ROOT / 'training/results/v2/ppo.pt').eval()
    critic = nn.Sequential(nn.Linear(10, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate, eps=1e-5)
    env = Arena(args.envs, args.seed, loop_detection=False)
    ids = rng.choice(3, args.envs, p=[.4, .4, .2])
    validation_seeds = (np.random.default_rng(402).choice(100_000_000, 32, replace=False) + 1_200_000_000).tolist()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(rules='pong-v3', architecture=[6, 64, 64, 3], opponents={'v2': .4, 'attacker': .4, 'interceptor': .2},
                  critic_inputs='six observations + three opponent one-hot + remaining time',
                  validation_seeds=validation_seeds, loop_detection=False,
                  selection='maximum minimum target score_rate across v2 and attacker, then mean target score')
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    start, total, best = time.monotonic(), 0, (-1., -1.)

    def value_inputs(obs):
        return torch.from_numpy(np.concatenate([obs, np.eye(3, dtype=np.float32)[ids],
            ((1600-env.ticks)/1600).astype(np.float32)[:, None]], axis=1))

    def opposition(obs):
        selected = np.zeros(args.envs, dtype=int)
        for k in range(3):
            mask = ids == k
            if mask.any():
                selected[mask] = actions(opponent, obs[mask]) if k == 0 else scripted(obs[mask], attack=k == 1)
        return selected

    def check(update, diagnostics):
        nonlocal best
        from competition_eval import evaluate_competition
        metrics = evaluate_competition(actor, validation_seeds,
            opponents={'selfplay-v2': opponent, 'attacker': lambda obs: scripted(obs, True)}, bootstrap_samples=200)
        # The helper exposes aggregates under opponents; keep all per-seed evidence.
        targets = metrics['opponents']
        rates = [targets[k]['score_rate'] for k in ['selfplay-v2', 'attacker']]
        key = (min(rates), float(np.mean(rates)))
        row = dict(update=update, env_steps=total, seconds=time.monotonic()-start,
                   minimum_score=key[0], mean_score=key[1], opponents=targets, **diagnostics)
        with (args.out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        if key > best:
            best = key
            torch.save(actor.state_dict(), args.out / 'best.pt')
            export(actor, args.out / 'best.onnx')
            (args.out / 'selection.json').write_text(json.dumps(row, indent=2) + '\n')
        torch.save(actor.state_dict(), args.out / 'last.pt')
        print(f'update {update:4} | {total:,} steps | target min {key[0]:.3f} | ' +
              ' | '.join(f'{k}: {targets[k]["wins"]}/{targets[k]["losses"]}/{targets[k]["draws"]}' for k in targets), flush=True)

    check(0, {})
    n, horizon = args.envs, args.horizon
    for update in range(1, args.updates+1):
        xs, cs, selected, logps, values, rewards, dones = [], [], [], [], [], [], []
        for _ in range(horizon):
            obs = env.obs()
            x, c = torch.from_numpy(obs), value_inputs(obs)
            with torch.no_grad():
                distribution = Categorical(logits=actor(x))
                action = distribution.sample()
                value = critic(c).squeeze(-1)
            outcome, hit = env.step(np.stack([action.numpy(), opposition(env.obs(1))], axis=1))
            reward = (outcome == 0).astype(np.float32) - (outcome == 1).astype(np.float32) + args.return_reward * (hit == 0)
            xs.append(x); cs.append(c); selected.append(action); logps.append(distribution.log_prob(action))
            values.append(value); rewards.append(torch.tensor(reward, dtype=torch.float32)); dones.append(torch.from_numpy(outcome >= 0))
            reset = np.flatnonzero(outcome >= 0)
            env.reset(reset)
            ids[reset] = rng.choice(3, len(reset), p=[.4, .4, .2])
        total += n*horizon
        x, c, action, old_lp, value, reward, done = map(torch.stack, [xs, cs, selected, logps, values, rewards, dones])
        with torch.no_grad():
            next_value = critic(value_inputs(env.obs())).squeeze(-1)
            gae = torch.zeros(n)
            advantage = torch.empty_like(reward)
            for t in reversed(range(horizon)):
                alive = (~done[t]).float()
                delta = reward[t] + args.gamma * next_value * alive - value[t]
                gae = delta + args.gamma * args.gae_lambda * alive * gae
                advantage[t] = gae
                next_value = value[t]
            returns = (advantage + value).flatten()
        x, c, action, old_lp, old_value = x.reshape(-1, 6), c.reshape(-1, 10), action.flatten(), old_lp.flatten(), value.flatten()
        advantage = advantage.flatten()
        advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
        optimizer.param_groups[0]['lr'] = args.learning_rate * (1 - .8*(update-1)/args.updates)
        for _ in range(4):
            kls = []
            for ix in torch.randperm(len(x)).split(2048):
                distribution = Categorical(logits=actor(x[ix]))
                lp = distribution.log_prob(action[ix])
                ratio = (lp-old_lp[ix]).exp()
                policy_loss = torch.maximum(-advantage[ix]*ratio, -advantage[ix]*ratio.clamp(.8, 1.2)).mean()
                predicted = critic(c[ix]).squeeze(-1)
                clipped = old_value[ix] + (predicted-old_value[ix]).clamp(-.2, .2)
                value_loss = .5*torch.maximum((predicted-returns[ix]).square(), (clipped-returns[ix]).square()).mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss + .5*value_loss - .003*entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()), .5)
                optimizer.step()
                kls.append(float(((ratio-1)-(lp-old_lp[ix])).mean().detach()))
            if np.mean(kls) > .03:
                break
        if update % args.eval_every == 0 or update == args.updates:
            check(update, dict(policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                               entropy=float(entropy.detach()), approximate_kl=float(np.mean(kls))))
    print(f'Completed in {time.monotonic()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
