"""Experimental rules only: matched serves, deadline snapshots and exact cycles.

The live engine, exported models, and frozen training contracts are never modified.
"""
import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from env import Arena, reflect
from policy import Actor, ROOT, actions, load_builtin
from reference import serve


@dataclass(frozen=True)
class Rules:
    paddle_speed: float = 2
    min_vx: int = 5
    max_vx: int = 13
    radius: int = 15
    symmetric_rounding: bool = False


CONFIGS = {
    'v2': Rules(),
    'cap12': Rules(max_vx=12),
    'cap14': Rules(max_vx=14),
    'paddle2.25': Rules(paddle_speed=2.25),
    'paddle2.5': Rules(paddle_speed=2.5),
    'paddle2.25-symmetric': Rules(paddle_speed=2.25, symmetric_rounding=True),
    'paddle2.5-symmetric': Rules(paddle_speed=2.5, symmetric_rounding=True),
    'paddle2.25-cap14': Rules(paddle_speed=2.25, max_vx=14, symmetric_rounding=True),
    'center6': Rules(min_vx=6),
    'larger-paddle': Rules(radius=17),
}


def shot(offset, rules):
    vx = rules.min_vx + np.floor((rules.max_vx - rules.min_vx) * np.abs(offset) / rules.radius + .5)
    value = 5 * offset / rules.radius
    vy = np.sign(value) * np.floor(np.abs(value) + .5) if rules.symmetric_rounding else np.floor(value + .5)
    return vx, np.clip(vy, -5, 5)


class StudyArena(Arena):
    """V2 tick order with explicit experimental parameters and no time termination."""
    def __init__(self, n, rules):
        super().__init__(n)
        self.rules = rules

    def step(self, selected):
        r = self.rules
        self.paddles[:] = np.clip(self.paddles + r.paddle_speed * (selected - 1), r.radius, 160 - r.radius)
        self.ball[:, :2] += self.ball[:, 2:]
        self.ticks += 1
        x, y, vx, vy = self.ball.T
        wall = (y < 4) | (y > 156)
        y[:] = np.where(y < 4, 8 - y, np.where(y > 156, 312 - y, y))
        vy[wall] *= -1
        contact = ((vx < 0) & (x <= 15)) | ((vx > 0) & (x >= 345))
        side = (vx > 0).astype(int)
        offset = y - self.paddles[np.arange(self.n), side]
        miss = contact & (np.abs(offset) > r.radius)
        hit = contact & ~miss
        speed_x, speed_y = shot(offset, r)
        x[hit] = np.where(side[hit] == 0, 15, 345)
        vx[hit] = np.where(side[hit] == 0, 1, -1) * speed_x[hit]
        vy[hit] = speed_y[hit]
        self.returns[hit] += 1
        return np.where(miss, 1 - side, -1), np.where(hit, side, -1)


def script(obs, rules, attack=False):
    """Same v2 heuristic, with candidate physics and quarter-grid paddle decoding."""
    x = np.rint(obs[:, 0].astype(np.float64) * 180 + 180)
    y = np.rint(obs[:, 1].astype(np.float64) * 80 + 80)
    vx = np.rint(obs[:, 2].astype(np.float64) * 6)
    vy = np.rint(obs[:, 3].astype(np.float64) * 6)
    own = np.rint((obs[:, 4].astype(np.float64) * 80 + 80) * 4) / 4
    other = np.rint((obs[:, 5].astype(np.float64) * 80 + 80) * 4) / 4
    ticks = np.maximum(1, np.ceil((x - 15) / np.maximum(1, -vx)))
    landing = reflect(y + ticks * vy)
    target = np.clip(landing, rules.radius, 160 - rules.radius)
    if attack:
        moves = np.arange(-65, 66)[None, :]
        centers = np.clip(own[:, None] + rules.paddle_speed * moves, rules.radius, 160 - rules.radius)
        offset = landing[:, None] - centers
        sx, sy = shot(offset, rules)
        flight = np.ceil(330 / sx)
        destination = reflect(landing[:, None] + flight * sy)
        margin = np.abs(destination - other[:, None]) - rules.radius - rules.paddle_speed * flight
        valid = (np.abs(moves) <= ticks[:, None]) & (np.abs(offset) <= rules.radius)
        quality = np.where(valid, margin - .001 * np.abs(centers - own[:, None]), -1e9)
        best = np.argmax(quality, axis=1)
        target = np.where(valid.any(axis=1), centers[np.arange(len(obs)), best], target)
    error = np.where(vx < 0, target, 80) - own
    threshold = rules.paddle_speed / 2
    return np.where(error < -threshold, 0, np.where(error > threshold, 2, 1)).astype(np.int64)


def load_policies():
    policies = {'T': load_builtin('tracker')}
    for code, name in [('N', 'ppo'), ('P', 'imitation')]:
        actor = Actor()
        actor.load_state_dict(torch.load(ROOT / f'training/results/v2/{name}.pt', weights_only=True))
        policies[code] = actor.eval()
    return policies


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


def summarize(record, limit, sides, receiver):
    winner = np.array(record['winner'])
    end = np.array(record['point_tick'])
    decisive = (winner >= 0) & (end <= limit)
    lengths = np.array(record['point_returns'])
    cycles = np.array(record['cycle_tick'])
    won, lost = decisive & (winner == sides), decisive & (winner != sides)
    margin = (won.astype(int) - lost.astype(int)).reshape(2, -1, 5).sum(-1)
    paired = margin.sum(0)
    return dict(limit=limit, n=len(sides), wins=int(won.sum()), losses=int(lost.sum()), draws=int((~decisive).sum()),
                cycles_by_limit=int(((cycles > 0) & (cycles <= limit)).sum()),
                draws_proven_cyclic_by_end=int(((cycles > 0) & ~decisive).sum()),
                draws_scoring_later=int(((winner >= 0) & ~decisive).sum()),
                draws_unresolved_at_end=int(((cycles == 0) & (winner < 0)).sum()),
                median_decisive_returns=float(np.median(lengths[decisive])) if decisive.any() else None,
                early_rate=float(((lengths < 2) & decisive).mean()),
                long_scored_rate=float(((lengths >= 6) & decisive).mean()),
                first_receiver_win_rate=float((winner[decisive] == receiver[decisive]).mean()) if decisive.any() else None,
                mean_ticks=float(np.where(decisive, end, limit).mean()),
                match_wins=int((margin > 0).sum()), match_losses=int((margin < 0).sum()), match_draws=int((margin == 0).sum()),
                paired_match_wins=int((paired > 0).sum()), paired_match_losses=int((paired < 0).sum()),
                paired_match_draws=int((paired == 0).sum()))


def run(rules, pair, seeds, policies, limits, detect_cycles=True):
    frames, sides = openings(seeds)
    n = len(frames)
    env = StudyArena(n, rules)
    env.load(frames)
    receiver = (env.ball[:, 2] > 0).astype(int)
    winner = np.full(n, -1)
    end = np.zeros(n, int)
    lengths = np.zeros(n, int)
    cycle_at = np.zeros(n, int)
    cycle_start = np.zeros(n, int)
    cycle_period = np.zeros(n, int)
    seen = [{} for _ in range(n)]
    histories = [deque(maxlen=6), deque(maxlen=6)]
    active = np.ones(n, bool)

    def decide(kind, obs, role):
        if kind == 'H':
            histories[role].append(obs.copy())
            return script(histories[role][0], rules)
        if kind in 'AD':
            return script(obs, rules, kind == 'A')
        return actions(policies[kind], obs)

    for tick in range(1, max(limits) + 1):
        left, right = env.obs(), env.obs(1)
        ours = np.where(sides[:, None] == 0, left, right)
        theirs = np.where(sides[:, None] == 0, right, left)
        a, b = decide(pair[0], ours, 0), decide(pair[1], theirs, 1)
        selected = np.stack([np.where(sides == 0, a, b), np.where(sides == 0, b, a)], axis=1)
        # Keep finished lanes physically bounded while preserving a fixed inference batch.
        saved_ball, saved_paddles = env.ball[~active].copy(), env.paddles[~active].copy()
        terminal, hit = env.step(selected)
        env.ball[~active], env.paddles[~active] = saved_ball, saved_paddles
        done = active & (terminal >= 0)
        winner[done], end[done], lengths[done] = terminal[done], tick, env.returns[done]
        active[done] = False
        if detect_cycles:
            for i in np.flatnonzero(active & (hit >= 0)):
                # Complete future-determining state at a contact. Include delayed policy memory.
                key = env.ball[i].tobytes() + env.paddles[i].tobytes()
                for role, kind in enumerate(pair):
                    if kind == 'H':
                        key += b''.join(obs[i].tobytes() for obs in histories[role])
                if key in seen[i]:
                    cycle_at[i], cycle_start[i] = tick, seen[i][key]
                    cycle_period[i] = tick - seen[i][key]
                    active[i] = False
                else:
                    seen[i][key] = tick
        if not active.any():
            break
    record = dict(winner=winner.tolist(), point_tick=end.tolist(), point_returns=lengths.tolist(),
                  cycle_tick=cycle_at.tolist(), cycle_start=cycle_start.tolist(), cycle_period=cycle_period.tolist())
    rows = [summarize(record, limit, sides, receiver) for limit in limits]
    return rows, record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=60)
    p.add_argument('--seed', type=int, default=117)
    p.add_argument('--configs', nargs='+', choices=CONFIGS, default=list(CONFIGS))
    p.add_argument('--pairs', nargs='+', default=['AA', 'AD', 'DD', 'NN', 'NA', 'NP', 'NT', 'AH'])
    p.add_argument('--limits', type=int, nargs='+', default=[800, 1600, 3200])
    p.add_argument('--out', type=Path, default=ROOT / 'training/results/rule-study/scan.json')
    args = p.parse_args()
    if args.count < 1 or any(v < 1 for v in args.limits):
        p.error('count and limits must be positive')
    if any(len(pair) != 2 or any(code not in 'ADHNPT' for code in pair) for pair in args.pairs):
        p.error('pairs must contain two policy codes from ADHNPT')
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    seeds = rng.choice(1_000_000_000, args.count, replace=False).astype(np.int64) + 3_200_000_000
    policies = load_policies()
    sources = ['src/game.ts', 'training/scripts/rule_study.py', 'training/results/v2/ppo.pt', 'training/results/v2/imitation.pt']
    result = dict(rng_seed=args.seed, seeds=seeds.tolist(), seed_count=args.count,
                  configs={k: asdict(CONFIGS[k]) for k in args.configs}, limits=sorted(set(args.limits)),
                  inference='Frozen v2 PyTorch checkpoint, deterministic first argmax, one CPU thread',
                  ordering='side, seed, five serves; N=Selfplay, P=Apprentice, A=attacker, D=interceptor, T=Tracker, H=five-tick-delayed interceptor',
                  hashes={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
                  rows=[], records=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for name in args.configs:
        for pair in args.pairs:
            rows, record = run(CONFIGS[name], pair, seeds, policies, result['limits'])
            result['rows'].extend(dict(config=name, pair=pair, **row) for row in rows)
            result['records'].append(dict(config=name, pair=pair, **record))
            print(json.dumps(dict(config=name, pair=pair, seconds=round(time.monotonic() - started, 1), rows=rows)), flush=True)
            args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
