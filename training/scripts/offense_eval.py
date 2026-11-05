"""Isolated matched-serve offensive-rule comparison; no live arena mutations.

The scripts are finite-horizon geometric baselines, not optimal game solvers or
learned submissions. They use only the six observations. Contact planning uses
exact discrete movement reachability including clamp/reverse and final motion.
"""
import argparse
from collections import deque
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from competition_eval import decisions, load_mlp, openings, summarize, validate_seeds
from env import reflect
from offense_rules import CONFIGS, OffenseArena, shot
from policy import ROOT

CENTERS = np.arange(15, 146)
NEXT = np.clip(CENTERS[:, None] + [-2, 0, 2], 15, 145).astype(int)
MAX_FLIGHT = 66


@lru_cache(maxsize=1)
def contact_paths():
    """First action reaching (final center, actual final movement) in t ticks.

    Last axis -1/0/+1 uses indices 0/1/2; index3 permits any final movement.
    -1 denotes unreachable. Every path has exactly t actions (stays allowed).
    """
    first = np.full((MAX_FLIGHT + 1, 131, 131, 4), -1, dtype=np.int8)
    p, q = CENTERS[:, None], CENTERS[None, :]
    preference = np.stack([np.where(q < p, 0, np.where(q > p, 2, 1)),
                           np.ones((131, 131), int),
                           np.where(q < p, 2, 0)], axis=-1)
    preference[q == p] = [1, 0, 2]
    for action in range(3):
        dst = NEXT[:, action] - 15
        motion = np.sign(NEXT[:, action] - CENTERS).astype(int) + 1
        first[1, np.arange(131), dst, motion] = action
        first[1, np.arange(131), dst, 3] = action
    for ticks in range(2, MAX_FLIGHT + 1):
        for rank in range(3):
            action = preference[:, :, rank]
            following = NEXT[np.arange(131)[:, None], action] - 15
            can = first[ticks - 1, following, np.arange(131)[None, :], :] >= 0
            first[ticks] = np.where((first[ticks] < 0) & can, action[:, :, None], first[ticks])
    return first


@lru_cache(maxsize=4)
def tables(rules):
    """All legal shot envelopes and exact current-contact option metadata."""
    motions = np.array([-1, 0, 1]) if rules.deflection else np.array([0])
    offsets = np.repeat(np.arange(-15, 16), len(motions))
    motion = np.tile(motions, 31)
    height = np.arange(157)[:, None]
    centers = height - offsets
    vx, vy = shot(offsets, motion, rules)
    flight = np.ceil(330 / vx).astype(int)
    landing = reflect(height + flight * vy).astype(int)
    legal = (centers >= 15) & (centers <= 145) & (height >= 4)
    # Actual nonzero movement must have a legal predecessor. Partial clamping
    # introduces no additional shot beyond the full-step predecessor here.
    legal &= (centers - 2 * motion >= 15) & (centers - 2 * motion <= 145)
    low = np.max(np.where(legal, landing - 15 - 2 * flight, -10000), axis=1)
    high = np.min(np.where(legal, landing + 15 + 2 * flight, 10000), axis=1)
    low, high = np.maximum(low, 15), np.minimum(high, 145)
    midpoint = (low + high) / 2
    # At our next opponent contact, choose the best reachable recovery point.
    recovery = np.clip(midpoint[landing], np.maximum(15, centers - 2 * flight),
                       np.minimum(145, centers + 2 * flight))
    risk = np.maximum(low[landing] - recovery, recovery - high[landing])
    return dict(offsets=offsets, motion=motion, centers=centers, flight=flight,
                landing=landing, legal=legal, low=low, high=high, midpoint=midpoint,
                risk=risk, motion_index=motion + 1 if rules.deflection else np.full(len(offsets), 3))


def decode(obs):
    raw = obs.astype(np.float64)
    x = np.rint(raw[:, 0] * 180 + 180).astype(int)
    y = np.rint(raw[:, 1] * 80 + 80).astype(int)
    vx = np.rint(raw[:, 2] * 6).astype(int)
    vy = np.rint(raw[:, 3] * 6).astype(int)
    own = np.rint(raw[:, 4] * 80 + 80).astype(int)
    other = np.rint(raw[:, 5] * 80 + 80).astype(int)
    distance = np.where(vx < 0, x - 15, 345 - x)
    ticks = np.maximum(1, np.ceil(distance / np.maximum(1, np.abs(vx)))).astype(int)
    landing = reflect(y + ticks * vy).astype(int)
    return vx, own, other, np.minimum(ticks, MAX_FLIGHT), landing


def make_policy(rules, kind='attack', delay=0):
    """Fixed geometric policy; kinds attack, defend, interceptor.

    While recovering, choose the midpoint of the minimax coverage interval for
    every legal reply at the opponent's next contact height. Empty intervals use
    the midpoint of their conflicting bounds; this minimizes the largest miss
    margin, but cannot guarantee survival. Incoming attack maximizes one-flight
    placement margin against projected geometric recovery, with a small own-risk
    penalty. Defender instead maximizes its subsequent worst-case recovery slack.
    Delay applies to ball/opponent observations only; own paddle remains current.
    Delayed policy memory requires deadline-only evaluation (no physical loops).
    """
    if kind not in ('attack', 'defend', 'interceptor') or delay < 0:
        raise ValueError('Unknown policy or negative observation delay')
    tab, paths = tables(rules), contact_paths()
    history = deque(maxlen=delay + 1)

    def policy(obs):
        if delay:
            history.append(obs.copy())
            delayed = history[0].copy()
            delayed[:, 4] = obs[:, 4]
            obs = delayed
        vx, own, other, ticks, landing = decode(obs)
        target = tab['midpoint'][landing]
        # Tie toward stay. Nonempty baseline bands are always approached safely.
        nxt = NEXT[own - 15]
        cost = np.abs(nxt - target[:, None]) + np.array([1e-6, 0, 1e-6])
        result = np.argmin(cost, axis=1).astype(np.int64)
        ids = np.flatnonzero(vx < 0)
        if not len(ids):
            return result
        h, p, t = landing[ids], own[ids], ticks[ids]
        if kind == 'interceptor':
            target = np.clip(h, 15, 145)
            result[ids] = np.argmin(np.abs(NEXT[p - 15] - target[:, None]) + [1e-6, 0, 1e-6], axis=1)
            return result
        centers = tab['centers'][h]
        first = paths[t[:, None], p[:, None] - 15,
                      np.clip(centers - 15, 0, 130), tab['motion_index'][None, :]]
        valid = tab['legal'][h] & (first >= 0)
        travel = np.abs(centers - p[:, None])
        if kind == 'defend':
            quality = -tab['risk'][h] - .001 * travel
        else:
            # Anticipate the opposing maximin recovery during the approach. At
            # collision, conservatively allow their simultaneous ±2-unit move.
            expected = np.clip(tab['midpoint'][h], other[ids] - 2 * np.maximum(0, t - 1),
                               other[ids] + 2 * np.maximum(0, t - 1))
            destination = tab['landing'][h]
            margin = np.abs(destination - expected[:, None]) - 15 - 2 * tab['flight'] - 2
            quality = margin - .05 * np.maximum(0, tab['risk'][h]) - .001 * travel
        best = np.argmax(np.where(valid, quality, -1e12), axis=1)
        possible = valid.any(axis=1)
        fallback = np.argmin(np.abs(NEXT[p - 15] - np.clip(h, 15, 145)[:, None]) + [1e-6, 0, 1e-6], axis=1)
        result[ids] = np.where(possible, first[np.arange(len(ids)), best], fallback)
        return result

    policy.has_memory = bool(delay)
    policy.description = dict(kind=kind, observation_delay_ticks=delay,
                              recovery='universal one-reply maximin coverage midpoint',
                              contact='exact discrete feasible center/final motion enumeration',
                              scope='finite-horizon heuristic, not a minimax game solution')
    return policy


def run_pair(rules, policy_a, policy_b, seeds, capture_replays=0):
    """Paired all-five-serve competition. No checkpoint selection or seed creation."""
    seeds = validate_seeds(seeds)
    frames, sides = openings(seeds)
    n = len(frames)
    loops = not (getattr(policy_a, 'has_memory', False) or getattr(policy_b, 'has_memory', False))
    env = OffenseArena(n, rules, loop_detection=loops)
    env.load(frames)
    receiver = (env.ball[:, 2] > 0).astype(int)
    outcome = np.full(n, -1)
    lengths, durations, reasons = (np.zeros(n, int) for _ in range(3))
    last_return_forced = np.zeros(n, bool)
    last_return_side = np.full(n, -1)
    forced_score = np.zeros(n, bool)
    replays = [dict(index=i, side=int(sides[i]), seed=seeds[(i // 5) % len(seeds)],
                    rally=i % 5 + 1, frames=[]) for i in range(min(capture_replays, n))]
    for tick in range(1600):
        active = outcome < 0
        left, right = env.obs(0), env.obs(1)
        ours = np.where(sides[:, None] == 0, left, right)
        theirs = np.where(sides[:, None] == 0, right, left)
        a, b = decisions(policy_a, ours, active), decisions(policy_b, theirs, active)
        chosen = np.stack([np.where(sides == 0, a, b), np.where(sides == 0, b, a)], axis=1)
        for replay in replays:
            i = replay['index']
            if active[i]:
                replay['frames'].append(dict(tick=tick, ball=env.ball[i].astype(int).tolist(),
                    paddles=env.paddles[i].astype(int).tolist(), actions=chosen[i].tolist(), returns=int(env.returns[i])))
        terminal, hit = env.step(chosen, active)
        hits = np.flatnonzero(active & (hit >= 0))
        if len(hits):
            flight = np.ceil(330 / np.abs(env.ball[hits, 2])).astype(int)
            landing = reflect(env.ball[hits, 1] + flight * env.ball[hits, 3])
            other = env.paddles[hits, 1 - hit[hits]]
            # Contact extents are broad enough that interval reachability is
            # exact for catching; integer clamp/reverse parity cannot remove all.
            last_return_forced[hits] = np.abs(landing - other) > 15 + 2 * flight
            last_return_side[hits] = hit[hits]
        done = active & (terminal >= 0)
        outcome[done], lengths[done], durations[done] = terminal[done], env.returns[done], env.ticks[done]
        reasons[done] = env.draw_reason[done]
        forced_score[done] = last_return_forced[done] & (last_return_side[done] == terminal[done])
        for replay in replays:
            i = replay['index']
            if done[i]:
                replay['frames'].append(dict(tick=int(env.ticks[i]), ball=env.ball[i].astype(int).tolist(),
                    paddles=env.paddles[i].astype(int).tolist(), actions=None, returns=int(env.returns[i]),
                    outcome=int(outcome[i]), draw_reason=int(reasons[i])))
        if (outcome >= 0).all():
            break
    bootstrap = np.random.default_rng(20261109).integers(0, len(seeds), (2000, len(seeds)))
    summary = summarize(outcome, sides, lengths, durations, reasons, seeds, bootstrap)
    decisive = outcome < 2
    summary.update(first_receiver_win_rate=float((outcome[decisive] == receiver[decisive]).mean()) if decisive.any() else None,
                   long_scored_rate=float((decisive & (lengths >= 6)).mean()),
                   scored_after_return_rate=float((decisive & (lengths > 0)).mean()),
                   locally_unreachable_scores=int(forced_score.sum()),
                   locally_unreachable_share_of_scores=float(forced_score[decisive].mean()) if decisive.any() else None,
                   loop_detection_enabled=loops,
                   reaction_window_ticks=dict(minimum=26, maximum=66))
    records = dict(outcome=outcome.tolist(), candidate_side=sides.tolist(), first_receiver=receiver.tolist(),
                   returns=lengths.tolist(), ticks=durations.tolist(), draw_reason=reasons.tolist(),
                   last_shot_locally_unreachable=forced_score.tolist())
    for replay in replays:
        i = replay['index']
        replay.update(outcome=int(outcome[i]), draw_reason=int(reasons[i]), returns=int(lengths[i]), ticks=int(durations[i]))
    return dict(rules=asdict(rules), seeds=seeds, seed_count=len(seeds), summary=summary, records=records, replays=replays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--configs', nargs='+', choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument('--pairs', nargs='+', default=['attack:attack', 'attack:defend', 'defend:defend', 'attack:delayed', 'contender:attack', 'contender:defend'])
    parser.add_argument('--out', type=Path, default=ROOT / 'training/results/offense-study/evaluation/screen.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    seeds = (np.random.default_rng(20261108).choice(100_000_000, args.count, replace=False) + 1_500_000_000).tolist()
    contender = load_mlp(ROOT / 'training/results/fully-learned/candidate/best.pt')
    result = dict(purpose='Diagnostic rule selection, no final-test claim', seeds=seeds, seed_count=len(seeds),
                  sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__).resolve(), ROOT / 'training/scripts/offense_rules.py']}, configs={})
    for name in args.configs:
        rules = CONFIGS[name]
        result['configs'][name] = {}
        for pair in args.pairs:
            def make(kind):
                return contender if kind == 'contender' else make_policy(rules, 'defend', delay=5) if kind == 'delayed' else make_policy(rules, kind)
            a, b = pair.split(':')
            started = time.monotonic()
            row = run_pair(rules, make(a), make(b), seeds)
            row['seconds'] = time.monotonic() - started
            result['configs'][name][pair] = row
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(dict(config=name, pair=pair, seconds=row['seconds'], **{k: row['summary'][k] for k in
                ['wins', 'losses', 'draws', 'median_decisive_returns', 'early_finish_rate', 'first_receiver_win_rate', 'locally_unreachable_scores']})), flush=True)


if __name__ == '__main__':
    main()
