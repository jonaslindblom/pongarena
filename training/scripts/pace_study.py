"""Isolated classic-inspired speed progression; no released rules are modified.

Horizontal speed increases on each return independently of placement. All game
state needed for the next return remains in the existing six observations.
"""
import argparse
from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from competition_eval import decisions, load_mlp, openings, summarize, validate_seeds
from env import Arena, reflect
from offense_eval import contact_paths, make_policy as old_policy, run_pair as old_pair
from offense_rules import CONFIGS as OLD_CONFIGS
from policy import ROOT


@dataclass(frozen=True)
class Config:
    name: str
    cap: int = 13
    vertical_max: int = 5
    baseline: bool = False
    scaled_angle: bool = False


CONFIGS = {'baseline': Config('baseline', baseline=True)}
CONFIGS.update({f'pace{cap}-v{vertical}': Config(f'pace{cap}-v{vertical}', cap, vertical)
                for vertical in (5, 6, 7, 8, 9) for cap in (11, 12, 13, 14)})
CONFIGS.update({f'pace13-v{vertical}-scaled': Config(f'pace13-v{vertical}-scaled', 13, vertical, scaled_angle=True)
                for vertical in (8, 9)})
CENTERS = np.arange(15, 146)
NEXT = np.clip(CENTERS[:, None] + [-2, 0, 2], 15, 145).astype(int)
OFFSETS = np.arange(-15, 16)
OUT = ROOT / 'training/results/pace-study'


def shot(offset, incoming_speed, config):
    offset = np.asarray(offset)
    speed = (5 + np.floor(8 * np.abs(offset) / 15 + .5) if config.baseline
             else np.minimum(config.cap, np.abs(incoming_speed) + 1))
    vertical = config.vertical_max * offset / 15
    if config.scaled_angle:
        vertical = config.vertical_max * speed * offset / (15 * config.cap)
    return speed, np.floor(vertical + .5)


class PaceArena(Arena):
    def __init__(self, n, config, seed=1, loop_detection=False):
        super().__init__(n, seed=seed, loop_detection=loop_detection)
        self.config = config

    def step(self, actions, active=None):
        if active is None:
            active = np.ones(self.n, bool)
        self.paddles[:] = np.where(active[:, None], np.clip(self.paddles + 2 * (actions - 1), 15, 145), self.paddles)
        self.ball[:, :2] += self.ball[:, 2:] * active[:, None]
        self.ticks += active
        x, y, vx, vy = self.ball.T
        wall = active & ((y < 4) | (y > 156))
        y[wall] = np.where(y[wall] < 4, 8 - y[wall], 312 - y[wall])
        vy[wall] *= -1
        contact = active & (((vx < 0) & (x <= 15)) | ((vx > 0) & (x >= 345)))
        side = (vx > 0).astype(int)
        offset = y - self.paddles[np.arange(self.n), side]
        miss = contact & (np.abs(offset) > 15)
        hit = contact & ~miss
        speed_x, speed_y = shot(offset[hit], np.abs(vx[hit]), self.config)
        x[hit] = np.where(side[hit] == 0, 15, 345)
        vx[hit] = np.where(side[hit] == 0, 1, -1) * speed_x
        vy[hit] = speed_y
        self.returns[hit] += 1
        loop = np.zeros(self.n, bool)
        if self.loop_detection:
            for i in np.flatnonzero(hit):
                key = tuple(self.ball[i]) + tuple(self.paddles[i])
                loop[i] = key in self.seen[i]
                self.seen[i].add(key)
        timeout = active & (self.ticks >= 1600) & ~miss & ~loop
        self.draw_reason[:] = np.where(loop, 1, np.where(timeout, 2, 0))
        return np.where(miss, 1-side, np.where(loop | timeout, 2, -1)), np.where(hit, side, -1)


@lru_cache(maxsize=16)
def tables(config):
    height = np.arange(157)[:, None]
    centers = height - OFFSETS
    legal = (centers >= 15) & (centers <= 145) & (height >= 4)
    speeds = np.minimum(config.cap, np.arange(config.cap + 1) + 1)
    sy = (np.floor(config.vertical_max * speeds[:, None] * OFFSETS / (15 * config.cap) + .5)
          if config.scaled_angle else np.broadcast_to(np.floor(config.vertical_max * OFFSETS / 15 + .5), (len(speeds), 31)))
    flights = np.ceil(330 / speeds).astype(int)
    landing = reflect(height[None] + flights[:, None, None] * sy[:, None, :]).astype(int)
    low = np.maximum(15, np.max(np.where(legal, landing - 15 - 2 * flights[:, None, None], -10000), axis=2))
    high = np.minimum(145, np.min(np.where(legal, landing + 15 + 2 * flights[:, None, None], 10000), axis=2))
    mid = (low + high) / 2
    risks = np.zeros_like(landing, float)
    for incoming in range(5, config.cap + 1):
        outgoing = min(config.cap, incoming + 1)
        destinations = landing[incoming]
        recover = np.clip(mid[outgoing, destinations], np.maximum(15, centers - 2*flights[incoming]),
                          np.minimum(145, centers + 2*flights[incoming]))
        risks[incoming] = np.maximum(low[outgoing, destinations] - recover, recover - high[outgoing, destinations])
    return dict(centers=centers, legal=legal, flight=flights, landing=landing, low=low, high=high, mid=mid, risk=risks)


def decode(obs):
    raw = obs.astype(np.float64)
    x = np.rint(raw[:, 0]*180 + 180).astype(int)
    y = np.rint(raw[:, 1]*80 + 80).astype(int)
    vx = np.rint(raw[:, 2]*6).astype(int)
    vy = np.rint(raw[:, 3]*6).astype(int)
    own = np.rint(raw[:, 4]*80 + 80).astype(int)
    other = np.rint(raw[:, 5]*80 + 80).astype(int)
    distance = np.where(vx < 0, x-15, 345-x)
    ticks = np.clip(np.ceil(distance/np.maximum(1, np.abs(vx))).astype(int), 1, 66)
    landing = reflect(y + ticks*vy).astype(int)
    return vx, own, other, ticks, landing


@lru_cache(maxsize=20)
def invariant(config):
    """Conservative safety fixed point at cap, then backward through speed stages.

    G[s,y,p] considers our paddle p just after an opponent contact whose incoming
    horizontal speed was s and contact height y. The opponent may choose every
    geometrically legal offset regardless of their earlier position. A surviving
    set is sufficient for defense; an empty one is not proof of winning attack.
    """
    if config.baseline:
        raise ValueError('Use the existing v3 invariant proof for baseline.')
    tab = tables(config)
    reachable = contact_paths()[:, :, :, 3] >= 0
    reachable[0] = np.eye(131, dtype=bool)
    fixed = np.zeros((config.cap+1, 157, 131), bool)
    legal = tab['legal'][4:]
    center = np.clip(tab['centers'][4:]-15, 0, 130)
    counts = {}

    def good_returns(incoming):
        outgoing = min(config.cap, incoming+1)
        flight = int(tab['flight'][incoming])
        recovery = fixed[outgoing] @ reachable[flight].T
        safe = legal & recovery[tab['landing'][incoming, 4:], center]
        good = np.zeros((157, 131), bool)
        for y in range(153):
            good[y+4, center[y, safe[y]]] = True
        return good

    for stage in range(config.cap, 4, -1):
        fixed[stage, 4:] = True
        sizes = [int(fixed[stage].sum())]
        while True:
            incoming = min(config.cap, stage+1)
            good = good_returns(incoming)
            interception = good @ reachable[int(tab['flight'][stage])].T
            following = np.ones((157, 131), bool)
            following[:4] = False
            for y in range(4, 157):
                destinations = np.unique(tab['landing'][stage, y, tab['legal'][y]])
                following[y] = interception[destinations].all(axis=0)
            assert not (following & ~fixed[stage]).any()
            sizes.append(int(following.sum()))
            same = np.array_equal(following, fixed[stage])
            fixed[stage] = following
            if same:
                break
        counts[str(stage)] = sizes
    good_serve_return = good_returns(5)
    serve_catch = good_serve_return @ reachable[33].T
    failures = [[], []]
    for center_y in (32, 56, 80, 104, 128):
        for y in range(center_y-10, center_y+11):
            for vy in (*range(-5, 0), *range(1, 6)):
                landing = int(reflect(y+33*vy))
                if not serve_catch[landing, 65]:
                    failures[0].append([y, vy])
                if not (reachable[33, 65] & fixed[5, landing]).any():
                    failures[1].append([y, vy])
    return dict(config=asdict(config), proves_never_concede_from_standard_serves=not any(failures),
                cap_safe_heights=int(fixed[config.cap].any(axis=1).sum()),
                stage_safe_heights={str(s): int(fixed[s].any(axis=1).sum()) for s in range(5, config.cap+1)},
                retained_state_counts=counts, serve_cases_per_role=1050,
                receiver_failures=len(failures[0]), nonreceiver_failures=len(failures[1]),
                failure_examples=[x[:10] for x in failures],
                interpretation='Sufficient controlled-invariant defense test; failure does not establish forced winning attack.')


def make_policy(config, kind='attack', delay=0):
    if config.baseline:
        return old_policy(OLD_CONFIGS['baseline'], kind, delay)
    if kind not in ('attack', 'defend', 'interceptor') or delay < 0:
        raise ValueError('Invalid policy or delay')
    tab, paths = tables(config), contact_paths()
    history = deque(maxlen=delay+1)

    def policy(obs):
        if delay:
            history.append(obs.copy())
            delayed = history[0].copy()
            delayed[:, 4] = obs[:, 4]
            obs = delayed
        vx, own, other, ticks, height = decode(obs)
        stage = np.minimum(config.cap, np.abs(vx))
        target = tab['mid'][stage, height]
        result = np.argmin(np.abs(NEXT[own-15] - target[:, None]) + [1e-6, 0, 1e-6], axis=1)
        ids = np.flatnonzero(vx < 0)
        if not len(ids):
            return result.astype(np.int64)
        h, p, t, speed = height[ids], own[ids], ticks[ids], stage[ids]
        fallback = np.argmin(np.abs(NEXT[p-15] - np.clip(h, 15, 145)[:, None]) + [1e-6, 0, 1e-6], axis=1)
        if kind == 'interceptor':
            result[ids] = fallback
            return result.astype(np.int64)
        centers = tab['centers'][h]
        first = paths[t[:, None], p[:, None]-15, np.clip(centers-15, 0, 130), 3]
        valid = tab['legal'][h] & (first >= 0)
        travel = np.abs(centers-p[:, None])
        if kind == 'defend':
            quality = -tab['risk'][speed, h] - .001*travel
        else:
            expected = np.clip(tab['mid'][speed, h], other[ids]-2*np.maximum(0, t-1), other[ids]+2*np.maximum(0, t-1))
            margin = np.abs(tab['landing'][speed, h] - expected[:, None]) - 15 - 2*tab['flight'][speed, None] - 2
            quality = margin - .05*np.maximum(0, tab['risk'][speed, h]) - .001*travel
        best = np.argmax(np.where(valid, quality, -1e12), axis=1)
        result[ids] = np.where(valid.any(axis=1), first[np.arange(len(ids)), best], fallback)
        return result.astype(np.int64)

    policy.has_memory = bool(delay)
    policy.description = dict(kind=kind, observation_delay_ticks=delay,
        recovery='stage-aware universal reply coverage midpoint',
        contact='exact discrete feasible contact positions; placement and next recovery',
        limitation='finite-horizon heuristic, not solved minimax play')
    return policy


def run_pair(config, policy_a, policy_b, seeds, capture_replays=0):
    if config.baseline:
        return old_pair(OLD_CONFIGS['baseline'], policy_a, policy_b, seeds, capture_replays)
    seeds = validate_seeds(seeds)
    frames, sides = openings(seeds)
    loops = not (getattr(policy_a, 'has_memory', False) or getattr(policy_b, 'has_memory', False))
    env = PaceArena(len(frames), config, loop_detection=loops)
    env.load(frames)
    n = env.n
    receiver = (env.ball[:, 2] > 0).astype(int)
    outcome = np.full(n, -1)
    lengths, durations, reasons = (np.zeros(n, int) for _ in range(3))
    forced = np.zeros(n, bool)
    last_forced = np.zeros(n, bool)
    last_side = np.full(n, -1)
    replays = [dict(index=i, side=int(sides[i]), seed=seeds[(i//5)%len(seeds)], rally=i%5+1, frames=[])
               for i in range(min(capture_replays, n))]
    for tick in range(1600):
        active = outcome < 0
        left, right = env.obs(), env.obs(1)
        ours = np.where(sides[:, None] == 0, left, right)
        theirs = np.where(sides[:, None] == 0, right, left)
        a, b = decisions(policy_a, ours, active), decisions(policy_b, theirs, active)
        selected = np.stack([np.where(sides == 0, a, b), np.where(sides == 0, b, a)], 1)
        for replay in replays:
            i = replay['index']
            if active[i]:
                replay['frames'].append(dict(tick=tick, ball=env.ball[i].astype(int).tolist(),
                    paddles=env.paddles[i].astype(int).tolist(), actions=selected[i].tolist(), returns=int(env.returns[i])))
        terminal, hit = env.step(selected, active)
        hits = np.flatnonzero(active & (hit >= 0))
        if len(hits):
            flight = np.ceil(330 / np.abs(env.ball[hits, 2])).astype(int)
            landing = reflect(env.ball[hits, 1] + flight*env.ball[hits, 3])
            last_forced[hits] = np.abs(landing-env.paddles[hits, 1-hit[hits]]) > 15+2*flight
            last_side[hits] = hit[hits]
        done = active & (terminal >= 0)
        outcome[done], lengths[done], durations[done] = terminal[done], env.returns[done], env.ticks[done]
        reasons[done] = env.draw_reason[done]
        forced[done] = last_forced[done] & (last_side[done] == terminal[done])
        for replay in replays:
            i = replay['index']
            if done[i]:
                replay['frames'].append(dict(tick=int(env.ticks[i]), ball=env.ball[i].astype(int).tolist(),
                    paddles=env.paddles[i].astype(int).tolist(), actions=None, returns=int(env.returns[i]),
                    outcome=int(outcome[i]), draw_reason=int(reasons[i])))
        if (outcome >= 0).all():
            break
    bootstrap = np.random.default_rng(20261122).integers(0, len(seeds), (2000, len(seeds)))
    summary = summarize(outcome, sides, lengths, durations, reasons, seeds, bootstrap)
    decisive = outcome < 2
    summary.update(first_receiver_win_rate=float((outcome[decisive] == receiver[decisive]).mean()) if decisive.any() else None,
        long_scored_rate=float((decisive & (lengths >= 6)).mean()),
        locally_unreachable_scores=int(forced.sum()),
        locally_unreachable_share_of_scores=float(forced[decisive].mean()) if decisive.any() else None,
        loop_detection_enabled=loops,
        reaction_window_ticks=dict(first_return=55, minimum=int(np.ceil(330/config.cap))))
    records = dict(outcome=outcome.tolist(), candidate_side=sides.tolist(), first_receiver=receiver.tolist(),
        returns=lengths.tolist(), ticks=durations.tolist(), draw_reason=reasons.tolist(), last_shot_locally_unreachable=forced.tolist())
    for replay in replays:
        i = replay['index']
        replay.update(outcome=int(outcome[i]), draw_reason=int(reasons[i]), returns=int(lengths[i]), ticks=int(durations[i]))
    return dict(rules=asdict(config), seeds=seeds, seed_count=len(seeds), summary=summary, records=records, replays=replays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--pairs', nargs='+', default=['attack:attack', 'attack:defend', 'defend:defend', 'attack:interceptor'])
    parser.add_argument('--out', type=Path, default=OUT/'screen.json')
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng_seed = 20261121 if args.confirmation else 20261120
    low = 2_400_000_000 if args.confirmation else 2_300_000_000
    seeds = (np.random.default_rng(rng_seed).choice(100_000_000, args.count, replace=False)+low).tolist()
    contender = load_mlp(ROOT/'training/results/fully-learned/candidate/best.pt')
    data = dict(purpose='Gradual speed progression tuning', seeds=seeds, rng_seed=rng_seed,
                phase='confirmation' if args.confirmation else 'screen', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), configs={})
    for name in args.configs:
        config = CONFIGS[name]
        data['configs'][name] = {}
        for pair in args.pairs:
            def make(kind):
                return contender if kind == 'contender' else make_policy(config, 'interceptor', 5) if kind == 'delayed' else make_policy(config, kind)
            a, b = pair.split(':')
            started = time.monotonic()
            result = run_pair(config, make(a), make(b), seeds, capture_replays=5 if args.confirmation and pair == 'attack:defend' else 0)
            result['seconds'] = time.monotonic()-started
            data['configs'][name][pair] = result
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(data, separators=(',', ':'))+'\n')
            s = result['summary']
            print(name, pair, s['wins'], s['losses'], s['draws'], 'median', s['median_decisive_returns'],
                  'early', s['early_finish_rate'], 'receiver', s['first_receiver_win_rate'], flush=True)


if __name__ == '__main__':
    main()
