"""DAgger warm start, then clipped PPO with a historical self-play opponent pool."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from env import Arena, scripted
from policy import Actor, actions, load_builtin, export
from evaluate import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('training/runs/v2-seed7'))
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dagger-rounds', type=int, default=10)
    p.add_argument('--updates', type=int, default=250)
    p.add_argument('--envs', type=int, default=256)
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--eval-every', type=int, default=25)
    args = p.parse_args()
    if min(args.dagger_rounds, args.updates, args.envs, args.horizon, args.eval_every) < 1:
        p.error('Round, update, environment, horizon and evaluation counts must be positive.')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'metrics.jsonl').exists():
        raise FileExistsError('Use a fresh --out directory for each run.')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = vars(args).copy()
    config.update(out=str(args.out), torch=torch.__version__, numpy=np.__version__,
                  architecture=[6, 64, 64, 3], activation='Tanh', rules='pong-v2', max_ball_vx=13, gamma=.99, gae_lambda=.95,
                  clip=.2, epochs=4, minibatch=2048, entropy=.01, return_reward=.02,
                  seed_ranges={'train': [0, 999999999], 'validation': [1000000000, 1000000023],
                               'test': [2100000000, 2100000099]})
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    actor = Actor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    baselines = [load_builtin(n) for n in ['tracker', 'anticipator', 'challenger']]
    env = Arena(args.envs, args.seed)
    opponent_ids = rng.integers(0, 5, args.envs)
    pool = []
    started = time.monotonic()
    total_steps = 0
    best = {'imitation': -1., 'ppo': -1.}

    def opposition(obs):
        out = np.empty(len(obs), dtype=int)
        for k in np.unique(opponent_ids):
            mask = opponent_ids == k
            if k < 3:
                out[mask] = actions(baselines[k], obs[mask])
            elif k < 5:
                out[mask] = scripted(obs[mask], attack=k == 4)
            else:
                out[mask] = actions(pool[k - 5], obs[mask])
        return out

    def reset_done(outcomes):
        ids = np.flatnonzero(outcomes >= 0)
        env.reset(ids)
        opponent_ids[ids] = rng.integers(0, 5 + len(pool), len(ids))

    def checkpoint(stage, iteration, diagnostics):
        metrics = evaluate(actor)
        score = float(np.mean([v['score_rate'] for v in metrics.values()]))
        row = dict(stage=stage, iteration=iteration, env_steps=total_steps,
                   seconds=time.monotonic() - started, validation_score=score,
                   opponents=metrics, **diagnostics)
        with (args.out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        torch.save(actor.state_dict(), args.out / 'last.pt')
        if score > best[stage]:
            best[stage] = score
            torch.save(actor.state_dict(), args.out / f'{stage}.pt')
            export(actor, args.out / f'{stage}.onnx')
            (args.out / f'{stage}-selection.json').write_text(json.dumps(row, indent=2) + '\n')
        rates = ', '.join(f'{k}={v["point_rate"]:.0%}' for k, v in metrics.items())
        print(f'{stage} {iteration:3d} | {total_steps:,} steps | validation {score:.3f} | {rates}', flush=True)

    checkpoint('imitation', 0, {'cross_entropy': None})
    replay_obs, replay_labels = [], []
    for iteration in range(1, args.dagger_rounds + 1):
        inputs, labels = [], []
        for _ in range(256):
            obs = env.obs()
            expert = scripted(obs, attack=True)
            learner = actions(actor, obs)
            # DAgger: label learner-visited states, gradually remove teacher control.
            beta = max(0., 1 - iteration / 4)
            act = np.where(rng.random(args.envs) < beta, expert, learner)
            inputs.append(obs)
            labels.append(expert)
            out, _ = env.step(np.stack([act, opposition(env.obs(1))], axis=1))
            reset_done(out)
        total_steps += args.envs * 256
        replay_obs.append(np.concatenate(inputs))
        replay_labels.append(np.concatenate(labels))
        replay_obs, replay_labels = replay_obs[-4:], replay_labels[-4:]
        x = torch.from_numpy(np.concatenate(replay_obs))
        y = torch.from_numpy(np.concatenate(replay_labels))
        for _ in range(6):
            order = torch.randperm(len(x))
            for ix in order.split(2048):
                loss = nn.functional.cross_entropy(actor(x[ix]), y[ix])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.)
                optimizer.step()
        checkpoint('imitation', iteration, {'cross_entropy': float(loss.detach())})

    actor.load_state_dict(torch.load(args.out / 'imitation.pt', weights_only=True))
    teacher = copy.deepcopy(actor).eval()
    pool.append(copy.deepcopy(actor).eval())
    critic = nn.Sequential(nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=2.5e-4, eps=1e-5)
    n, horizon = args.envs, args.horizon
    for update in range(1, args.updates + 1):
        rollout_obs, rollout_a, rollout_lp, rollout_v, rollout_r, rollout_done = [], [], [], [], [], []
        for _ in range(horizon):
            obs = torch.from_numpy(env.obs())
            with torch.no_grad():
                dist = Categorical(logits=actor(obs))
                a = dist.sample()
                v = critic(obs).squeeze(-1)
            out, hit = env.step(np.stack([a.numpy(), opposition(env.obs(1))], axis=1))
            rewards = (out == 0).astype(np.float32) - (out == 1).astype(np.float32) + .02 * (hit == 0)
            rollout_obs.append(obs)
            rollout_a.append(a)
            rollout_lp.append(dist.log_prob(a))
            rollout_v.append(v)
            rollout_r.append(torch.tensor(rewards, dtype=torch.float32))
            rollout_done.append(torch.from_numpy(out >= 0))
            reset_done(out)
        total_steps += n * horizon
        x, a, old_lp, values, rewards, dones = map(torch.stack,
            [rollout_obs, rollout_a, rollout_lp, rollout_v, rollout_r, rollout_done])
        with torch.no_grad():
            next_value = critic(torch.from_numpy(env.obs())).squeeze(-1)
            gae = torch.zeros(n)
            advantages = torch.empty_like(rewards)
            for t in reversed(range(horizon)):
                alive = (~dones[t]).float()
                delta = rewards[t] + .99 * next_value * alive - values[t]
                gae = delta + .99 * .95 * alive * gae
                advantages[t] = gae
                next_value = values[t]
            returns = advantages + values
        x, a, old_lp = x.reshape(-1, 6), a.flatten(), old_lp.flatten()
        advantages, returns, old_values = advantages.flatten(), returns.flatten(), values.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Keep a small, decaying imitation anchor to limit catastrophic forgetting.
        anchor = .05 * max(0., 1 - update / args.updates)
        optimizer.param_groups[0]['lr'] = 2.5e-4 * (1 - .8 * (update - 1) / args.updates)
        kls, clips = [], []
        for epoch in range(4):
            for ix in torch.randperm(len(x)).split(2048):
                logits = actor(x[ix])
                dist = Categorical(logits=logits)
                lp = dist.log_prob(a[ix])
                logratio = lp - old_lp[ix]
                ratio = logratio.exp()
                pg = torch.maximum(-advantages[ix] * ratio,
                                   -advantages[ix] * ratio.clamp(.8, 1.2)).mean()
                predicted = critic(x[ix]).squeeze(-1)
                clipped = old_values[ix] + (predicted - old_values[ix]).clamp(-.2, .2)
                vf = .5 * torch.maximum((predicted - returns[ix]).square(), (clipped - returns[ix]).square()).mean()
                with torch.no_grad():
                    target = teacher(x[ix]).softmax(-1)
                bc = -(target * logits.log_softmax(-1)).sum(-1).mean()
                entropy = dist.entropy().mean()
                loss = pg + .5 * vf - .01 * entropy + anchor * bc
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), .5)
                optimizer.step()
                kls.append(float(((ratio - 1) - logratio).mean().detach()))
                clips.append(float(((ratio - 1).abs() > .2).float().mean()))
            if np.mean(kls[-max(1, len(x) // 2048):]) > .03:
                break
        if update % 25 == 0:
            pool.append(copy.deepcopy(actor).eval())
            # Fixed small league; reset IDs when replacing snapshots.
            if len(pool) > 5:
                pool.pop(0)
                opponent_ids[:] = rng.integers(0, 5 + len(pool), n)
        if update % args.eval_every == 0 or update == args.updates:
            checkpoint('ppo', update, {'policy_loss': float(pg.detach()), 'value_loss': float(vf.detach()),
                                      'entropy': float(entropy.detach()), 'approx_kl': float(np.mean(kls)),
                                      'clip_fraction': float(np.mean(clips))})
    # Training-only state for inspection; inference ONNX files contain actor weights only.
    torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict(),
                'optimizer': optimizer.state_dict(), 'config': config}, args.out / 'training-state.pt')
    print(f'Finished in {time.monotonic() - started:.1f}s. Best validation scores: {best}', flush=True)


if __name__ == '__main__':
    main()
