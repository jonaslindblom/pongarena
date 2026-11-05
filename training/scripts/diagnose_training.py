"""Development-only diagnostics of frozen policies; does not alter rules or weights."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from env import Arena, scripted
from policy import Actor, ROOT, actions
from reference import serve


def load(path):
    actor = Actor()
    actor.load_state_dict(torch.load(path, weights_only=True))
    return actor.eval()


def openings(seeds):
    frames, sides = [], []
    for side in (0, 1):
        for seed in seeds:
            f = None
            for _ in range(5):
                f = serve(int(seed), f)
                frames.append(f)
                sides.append(side)
    return frames, np.array(sides)


def run(actor, opponent, frames, sides, sample, loops, random_seed):
    env = Arena(len(frames), loop_detection=loops)
    env.load(frames)
    generator = torch.Generator().manual_seed(random_seed)
    n = len(frames)
    outcome = np.full(n, -1)
    duration = np.zeros(n, int)
    lengths = np.zeros(n, int)
    reason = np.zeros(n, int)
    undiscounted = np.zeros(n)
    discounted = np.zeros(n)
    terminal_discounted = np.zeros(n)
    return_discounted = np.zeros(n)
    entropy_sum = 0.
    action_count = 0
    for tick in range(1600):
        active = outcome < 0
        left, right = env.obs(0), env.obs(1)
        ours = np.where(sides[:, None] == 0, left, right)
        theirs = np.where(sides[:, None] == 0, right, left)
        with torch.no_grad():
            logits = actor(torch.from_numpy(ours))
            probabilities = logits.softmax(-1)
            a = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1).numpy() if sample else logits.argmax(-1).numpy()
            entropy = -(probabilities * logits.log_softmax(-1)).sum(-1).numpy()
        b = actions(opponent, theirs) if isinstance(opponent, torch.nn.Module) else opponent(theirs)
        pair = np.stack([np.where(sides == 0, a, b), np.where(sides == 0, b, a)], 1)
        terminal, hit = env.step(pair, active)
        terminal_reward = ((terminal == sides) & active).astype(float) - ((terminal == 1 - sides) & active).astype(float)
        return_reward = .02 * ((hit == sides) & active)
        undiscounted += terminal_reward + return_reward
        discounted += .99 ** tick * (terminal_reward + return_reward)
        terminal_discounted += .99 ** tick * terminal_reward
        return_discounted += .99 ** tick * return_reward
        entropy_sum += float(entropy[active].sum())
        action_count += int(active.sum())
        done = active & (terminal >= 0)
        outcome[done] = terminal[done]
        duration[done] = env.ticks[done]
        lengths[done] = env.returns[done]
        reason[done] = env.draw_reason[done]
        if (outcome >= 0).all():
            break
    won, lost, drawn = outcome == sides, outcome == 1 - sides, outcome == 2
    margin = (won.astype(int) - lost.astype(int)).reshape(2, -1, 5).sum(-1)
    paired = margin.sum(0)
    return dict(rallies=n, wins=int(won.sum()), losses=int(lost.sum()), draws=int(drawn.sum()),
                loop_draws=int((reason == 1).sum()), timeout_draws=int((reason == 2).sum()),
                mean_net_points=float((won.astype(int) - lost.astype(int)).mean()),
                score_rate=float((won + .5 * drawn).mean()),
                paired_wins=int((paired > 0).sum()), paired_losses=int((paired < 0).sum()), paired_draws=int((paired == 0).sum()),
                mean_ticks=float(duration.mean()),
                median_decisive_returns=float(np.median(lengths[~drawn])) if (~drawn).any() else None,
                mean_training_return=float(undiscounted.mean()),
                mean_discounted_training_return=float(discounted.mean()),
                mean_discounted_point_reward=float(terminal_discounted.mean()),
                mean_discounted_return_bonus=float(return_discounted.mean()),
                mean_action_entropy=entropy_sum / action_count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--sampler-seed', type=int, default=1847)
    p.add_argument('--out', type=Path, default=ROOT / 'training/results/training-diagnosis/diagnosis.json')
    args = p.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.sampler_seed)
    seeds = rng.choice(100_000_000, args.count, replace=False).astype(np.int64) + 1_200_000_000
    frames, sides = openings(seeds)
    paths = {'v2': ROOT / 'training/results/v2/ppo.pt'}
    paths.update({f'v3-seed{seed}': ROOT / f'training/results/v3/seed{seed}/ppo.pt' for seed in [7, 17, 29]})
    actors = {name: load(path) for name, path in paths.items()}
    opponents = {'attacker': lambda x: scripted(x, True), 'interceptor': lambda x: scripted(x), 'v2': actors['v2']}
    result = dict(purpose='Development diagnostics, not final model testing', seeds=seeds.tolist(),
                  seed_count=args.count, sampler_seed=args.sampler_seed, rules='pong-v3',
                  policy_hashes={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
                  discounts={str(t): .99 ** t for t in [100, 200, 400, 800, 1600]},
                  rows=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, actor in actors.items():
        for opponent_name, opponent in opponents.items():
            for mode, sample, loops in [('greedy', False, True), ('greedy-full', False, False), ('sampled-full', True, False)]:
                row = dict(model=name, opponent=opponent_name, mode=mode,
                           **run(actor, opponent, frames, sides, sample, loops, args.sampler_seed))
                result['rows'].append(row)
                args.out.write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps(row), flush=True)
    for name in actors:
        for opponent in opponents:
            a, b = [next(r for r in result['rows'] if r['model'] == name and r['opponent'] == opponent and r['mode'] == mode) for mode in ['greedy', 'greedy-full']]
            assert [a[k] for k in ['wins', 'losses', 'draws']] == [b[k] for k in ['wins', 'losses', 'draws']]
    print('PASS: greedy loop exits preserve all outcomes compared with full 1,600-tick continuations.')


if __name__ == '__main__':
    main()
