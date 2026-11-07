"""Outcome-based PPO with script opponents and a small pool of frozen policies.

Only the supplied actor is exported. The critic and opponent snapshots are
training tools; neither changes the six-observation, three-score contract.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from game import Arena
from models import network, decide
from teachers import teacher


DEFAULTS = dict(rounds=128, envs=128, horizon=256, epochs=4, batch_size=2048,
    learning_rate=3e-5, value_learning_rate=3e-4, gamma=.995, gae_lambda=.95,
    clip=.2, entropy=.002, value_coefficient=.5, max_grad_norm=.5,
    target_kl=.02, critic_warmup=4, snapshot_every=16, snapshot_limit=4,
    evaluate_every=8, temperature=1., script_fraction=.75)


def advantages(rewards, values, ended, last_value, gamma, gae_lambda):
    """Bootstrap a rollout cut, but never cross a finished rally into its reset.

The arena's 1,600-tick limit is a scored draw, so it is terminal, not an
external time truncation whose value should be bootstrapped.
"""
    result = torch.zeros_like(rewards)
    following = last_value
    running = torch.zeros_like(last_value)
    for t in range(len(rewards) - 1, -1, -1):
        continues = (~ended[t]).to(rewards.dtype)
        delta = rewards[t] + gamma * following * continues - values[t]
        running = delta + gamma * gae_lambda * continues * running
        result[t] = running
        following = values[t]
    return result, result + values


def finetune(policy, config, seed, on_evaluate=None, on_progress=None):
    """Update an actor in place; callbacks receive (actor, row) or just row.

Selection belongs to the caller: evaluate the deterministic exported policy
on development opponents, and retain the best checkpoint across both phases.
"""
    cfg = DEFAULTS | config
    if set(config) - set(DEFAULTS):
        raise ValueError('Unknown PPO settings: ' + ', '.join(sorted(set(config) - set(DEFAULTS))))
    counts = ('rounds', 'envs', 'horizon', 'epochs', 'batch_size', 'snapshot_every',
              'snapshot_limit', 'evaluate_every')
    if any(not isinstance(cfg[key], int) or cfg[key] < 1 for key in counts):
        raise ValueError('PPO counts and snapshot limits must be positive integers.')
    if any(cfg[key] <= 0 for key in ('learning_rate', 'value_learning_rate', 'temperature',
                                    'max_grad_norm', 'target_kl')):
        raise ValueError('PPO learning rates, temperature and update limits must be positive.')
    if any(not 0 <= cfg[key] <= 1 for key in ('gamma', 'gae_lambda', 'clip', 'script_fraction')):
        raise ValueError('PPO discounts, clipping and opponent fraction must lie between zero and one.')
    if cfg['critic_warmup'] < 0 or cfg['entropy'] < 0 or cfg['value_coefficient'] < 0:
        raise ValueError('PPO warmup and loss coefficients cannot be negative.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    critic = network([6, 128, 128, 1], 'Tanh')
    actor_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg['learning_rate'], eps=1e-5)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg['value_learning_rate'], eps=1e-5)
    scripts = [teacher(name) for name in ('attack', 'defend', 'interceptor')]
    snapshots = {0: copy.deepcopy(policy).eval()}
    available = [0]
    arena = Arena(cfg['envs'], seed=seed, loop_detection=False)
    sides = rng.integers(0, 2, arena.n)
    # Most rallies use a script; the remainder use a past learned policy.
    opponent_ids = (rng.random(arena.n) >= cfg['script_fraction']).astype(int)
    script_ids = rng.integers(0, len(scripts), arena.n)
    snapshot_ids = np.zeros(arena.n, dtype=int)
    shape = (cfg['horizon'], arena.n)
    observations = torch.empty((*shape, 6))
    actions = torch.empty(shape, dtype=torch.long)
    old_log_probs, values, rewards = (torch.empty(shape) for _ in range(3))
    ended = torch.empty(shape, dtype=torch.bool)
    totals = np.zeros(3, dtype=int)
    elapsed_steps = 0
    policy.train()
    for iteration in range(1, cfg['rounds'] + 1):
        outcomes_this_round = np.zeros(3, dtype=int)
        for t in range(cfg['horizon']):
            left, right = arena.obs(0), arena.obs(1)
            ours = np.where(sides[:, None] == 0, left, right)
            theirs = np.where(sides[:, None] == 0, right, left)
            observation = torch.from_numpy(ours)
            with torch.no_grad():
                distribution = Categorical(logits=policy(observation) / cfg['temperature'])
                action = distribution.sample()
                observations[t] = observation
                actions[t] = action
                old_log_probs[t] = distribution.log_prob(action)
                values[t] = critic(observation).squeeze(-1)
            other = np.empty(arena.n, dtype=int)
            for i, opponent in enumerate(scripts):
                mask = (opponent_ids == 0) & (script_ids == i)
                if mask.any():
                    other[mask] = opponent(theirs[mask])
            for i, opponent in snapshots.items():
                mask = (opponent_ids == 1) & (snapshot_ids == i)
                if mask.any():
                    other[mask] = decide(opponent, theirs[mask])
            action = action.numpy()
            outcome, _ = arena.step(np.stack([
                np.where(sides == 0, action, other),
                np.where(sides == 0, other, action)], axis=1))
            done = outcome >= 0
            won = done & (outcome == sides)
            lost = done & (outcome < 2) & ~won
            drawn = outcome == 2
            rewards[t] = torch.from_numpy(won.astype(np.float32) - lost.astype(np.float32))
            ended[t] = torch.from_numpy(done)
            outcomes_this_round += [won.sum(), lost.sum(), drawn.sum()]
            ids = np.flatnonzero(done)
            arena.reset(ids)
            sides[ids] = rng.integers(0, 2, len(ids))
            opponent_ids[ids] = (rng.random(len(ids)) >= cfg['script_fraction']).astype(int)
            script_ids[ids] = rng.integers(0, len(scripts), len(ids))
            snapshot_ids[ids] = rng.choice(available, len(ids))
        last_obs = np.where(sides[:, None] == 0, arena.obs(0), arena.obs(1))
        with torch.no_grad():
            last_value = critic(torch.from_numpy(last_obs)).squeeze(-1)
            advantage, returns = advantages(rewards, values, ended, last_value,
                cfg['gamma'], cfg['gae_lambda'])
        x, a = observations.reshape(-1, 6), actions.flatten()
        log_before, target = old_log_probs.flatten(), returns.flatten()
        adv = advantage.flatten()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        fraction = 1 - (iteration - 1) / cfg['rounds']
        actor_optimizer.param_groups[0]['lr'] = cfg['learning_rate'] * fraction
        critic_optimizer.param_groups[0]['lr'] = cfg['value_learning_rate'] * fraction
        losses, kls, entropies = [], [], []
        update_actor = iteration > cfg['critic_warmup']
        for _ in range(cfg['epochs']):
            epoch_kls = []
            for ids in torch.randperm(len(x)).split(cfg['batch_size']):
                prediction = critic(x[ids]).squeeze(-1)
                value_loss = .5 * nn.functional.mse_loss(prediction, target[ids])
                if not torch.isfinite(value_loss):
                    raise ValueError('PPO critic loss is not finite.')
                critic_optimizer.zero_grad()
                (cfg['value_coefficient'] * value_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg['max_grad_norm'])
                critic_optimizer.step()
                if update_actor:
                    distribution = Categorical(logits=policy(x[ids]) / cfg['temperature'])
                    log_ratio = distribution.log_prob(a[ids]) - log_before[ids]
                    ratio = log_ratio.exp()
                    policy_loss = -torch.minimum(adv[ids] * ratio,
                        adv[ids] * ratio.clamp(1 - cfg['clip'], 1 + cfg['clip'])).mean()
                    entropy = distribution.entropy().mean()
                    loss = policy_loss - cfg['entropy'] * entropy
                    if not torch.isfinite(loss):
                        raise ValueError('PPO actor loss is not finite.')
                    actor_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), cfg['max_grad_norm'])
                    actor_optimizer.step()
                    kl = float(((ratio - 1) - log_ratio).mean().detach())
                    epoch_kls.append(kl)
                    losses.append(float(policy_loss.detach()))
                    entropies.append(float(entropy.detach()))
            if epoch_kls:
                kls.extend(epoch_kls)
                if np.mean(epoch_kls) > cfg['target_kl']:
                    break
        elapsed_steps += cfg['envs'] * cfg['horizon']
        totals += outcomes_this_round
        row = dict(phase='ppo', round=iteration, environment_steps=elapsed_steps,
            wins=int(outcomes_this_round[0]), losses=int(outcomes_this_round[1]),
            draws=int(outcomes_this_round[2]), policy_loss=float(np.mean(losses)) if losses else 0.,
            approximate_kl=float(np.mean(kls)) if kls else 0.,
            entropy=float(np.mean(entropies)) if entropies else 0.)
        if on_evaluate and (iteration % cfg['evaluate_every'] == 0 or iteration == cfg['rounds']):
            on_evaluate(policy, row)
        if on_progress:
            on_progress(row)
        if iteration % cfg['snapshot_every'] == 0:
            snapshots[iteration] = copy.deepcopy(policy).eval()
            available = (available + [iteration])[-cfg['snapshot_limit']:]
        # Retired snapshots remain pinned until their active rallies finish.
        # An opponent never changes midway through a rally.
        needed = set(available) | set(snapshot_ids[opponent_ids == 1])
        snapshots = {key: value for key, value in snapshots.items() if key in needed}
    return dict(environment_steps=elapsed_steps, wins=int(totals[0]),
                losses=int(totals[1]), draws=int(totals[2]))
