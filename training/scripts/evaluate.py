"""Deterministic first-argmax evaluation, paired serves, both sides, five-rally matches."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from env import Arena, scripted
from reference import serve, RALLY_TICK_LIMIT
from policy import Actor, actions, load_builtin, OnnxPolicy


def run_policy(policy, obs, active=None):
    if isinstance(policy, OnnxPolicy) and active is not None:
        result = np.ones(len(obs), dtype=int)
        result[active] = policy(obs[active])
        return result
    return actions(policy, obs) if isinstance(policy, torch.nn.Module) else policy(obs)


def evaluate(actor, seed_start=1_100_000_000, count=24, opponents=None):
    """All supplied policies must be fixed, deterministic, and stateless."""
    if opponents is None:
        opponents = {name: load_builtin(name) for name in ['tracker', 'anticipator', 'challenger']}
        opponents.update(interceptor=lambda x: scripted(x), attacker=lambda x: scripted(x, True))
    frames, sides = [], []
    for side in (0, 1):
        for seed in range(seed_start, seed_start + count):
            f = None
            for _ in range(5):
                f = serve(seed, f)
                frames.append(f)
                sides.append(side)
    sides = np.array(sides)
    results = {}
    for name, opponent in opponents.items():
        env = Arena(len(frames), loop_detection=True)
        env.load(frames)
        outcome = np.full(len(frames), -1)
        lengths = np.zeros(len(frames), dtype=int)
        durations = np.zeros(len(frames), dtype=int)
        reasons = np.zeros(len(frames), dtype=int)
        for _ in range(RALLY_TICK_LIMIT):
            active = outcome == -1
            left, right = env.obs(0), env.obs(1)
            ours = np.where(sides[:, None] == 0, left, right)
            theirs = np.where(sides[:, None] == 0, right, left)
            a, b = run_policy(actor, ours, active), run_policy(opponent, theirs, active)
            pair = np.stack([np.where(sides == 0, a, b), np.where(sides == 0, b, a)], axis=1)
            terminal, _ = env.step(pair, active)
            done = (outcome == -1) & (terminal >= 0)
            outcome[done] = terminal[done]
            lengths[done] = env.returns[done]
            durations[done] = env.ticks[done]
            reasons[done] = env.draw_reason[done]
            if (outcome >= 0).all():
                break
        won, lost, drawn = outcome == sides, outcome == 1 - sides, outcome == 2
        match_margin = (won.astype(int) - lost.astype(int)).reshape(2, count, 5).sum(-1)
        paired_margin = match_margin.sum(0)
        decisive = lengths[~drawn]
        results[name] = dict(rallies=len(frames), wins=int(won.sum()), losses=int(lost.sum()), draws=int(drawn.sum()),
                             loop_draws=int((reasons == 1).sum()), timeout_draws=int((reasons == 2).sum()),
                             point_rate=float(won.mean()), score_rate=float((won + .5 * drawn).mean()),
                             match_wins=int((match_margin > 0).sum()), match_losses=int((match_margin < 0).sum()),
                             match_draws=int((match_margin == 0).sum()),
                             paired_match_wins=int((paired_margin > 0).sum()), paired_match_losses=int((paired_margin < 0).sum()),
                             paired_match_draws=int((paired_margin == 0).sum()),
                             median_decisive_returns=float(np.median(decisive)) if len(decisive) else None,
                             mean_returns=float(lengths.mean()), mean_ticks=float(durations.mean()),
                             early_finish_rate=float(((lengths < 2) & ~drawn).mean()))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', type=Path)
    p.add_argument('--seeds', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=2_200_000_000)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.model.suffix == '.onnx':
        actor = OnnxPolicy(args.model)
    else:
        actor = Actor()
        actor.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
        actor.eval()
    result = {'rules': 'pong-v3', 'seed_start': args.seed_start, 'seed_count': args.seeds, 'model': str(args.model),
              'opponents': evaluate(actor, args.seed_start, args.seeds)}
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
