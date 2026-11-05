"""Diagnose scoring mechanisms without changing rules or deployed controllers.

Local reachability is measured at the previous shot's release. A reachable miss
is not automatically an unforced strategic error: earlier choices and the risk
of attempting a particular return are outside this one-flight counterfactual.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from competition_eval import decisions, load_mlp, openings, references, summarize
from env import Arena, reflect, scripted
from policy import ROOT
from reference import RALLY_TICK_LIMIT, serve


def flight(ball, paddles):
    x, y, vx, vy = ball.T
    receiver = (vx > 0).astype(int)
    distance = np.where(receiver == 0, x - 15, 345 - x)
    ticks = np.maximum(1, np.ceil(distance / np.abs(vx))).astype(int)
    landing = reflect(y + ticks * vy)
    own = paddles[np.arange(len(ball)), receiver]
    low, high = np.clip(own - 2*ticks, 15, 145), np.clip(own + 2*ticks, 15, 145)
    margin = np.maximum(low - landing, landing - high) - 15
    return receiver, ticks, landing, margin


def check_reachability():
    rng = np.random.default_rng(2202)
    n = 25000
    frames = [serve(7) for _ in range(n)]
    env = Arena(n)
    env.load(frames)
    env.ball[:, 0] = rng.choice([15, 180, 345], n)
    env.ball[:, 1] = rng.integers(4, 157, n)
    env.ball[:, 2] = rng.integers(5, 14, n) * np.where(env.ball[:, 0] == 345, -1, 1)
    env.ball[:, 3] = rng.integers(-5, 6, n)
    env.paddles[:] = rng.integers(15, 146, (n, 2))
    receiver, ticks, landing, margin = flight(env.ball, env.paddles)
    active = np.ones(n, bool)
    for _ in range(int(ticks.max())):
        action = np.ones((n, 2), int)
        error = np.clip(landing, 15, 145) - env.paddles[np.arange(n), receiver]
        action[np.arange(n), receiver] = np.where(error < -1, 0, np.where(error > 1, 2, 1))
        outcome, hit = env.step(action, active)
        contact = active & ((hit >= 0) | (outcome >= 0))
        np.testing.assert_array_equal(env.ball[contact, 1], landing[contact])
        np.testing.assert_array_equal(env.ticks[contact], ticks[contact])
        np.testing.assert_array_equal(hit[contact] < 0, margin[contact] > 0)
        active[contact] = False
    assert not active.any()
    return dict(probes=n, forced=int((margin > 0).sum()), reachable=int((margin <= 0).sum()),
                method='Exact vector engine, ideal immediate chase until the first contact; predicted landing, tick and catchability must match.')


def distribution(values):
    if not len(values):
        return dict(count=0, mean=None, median=None, p10=None, p90=None)
    return dict(count=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)))


def analyze(actor, opponent, seeds):
    frames, sides = openings(seeds)
    env = Arena(len(frames), loop_detection=True)
    env.load(frames)
    n = env.n
    outcome = np.full(n, -1)
    lengths, durations, reasons = (np.zeros(n, int) for _ in range(3))
    receiver, pending_ticks, landing, margin = flight(env.ball, env.paddles)
    launch_tick = np.zeros(n, int)
    from_serve = np.ones(n, bool)
    points = {name: dict(forced_after_return=0, reachable_after_return=0, serve=0) for name in ['candidate', 'opponent']}
    shots = {name: dict(offset=[], speed=[], margin=[], completed_margin=[], scorable_margin=[]) for name in points}
    point_records = []
    for _ in range(RALLY_TICK_LIMIT):
        active = outcome < 0
        left, right = env.obs(), env.obs(1)
        ours = np.where(sides[:, None] == 0, left, right)
        theirs = np.where(sides[:, None] == 0, right, left)
        a, b = decisions(actor, ours, active), decisions(opponent, theirs, active)
        selected = np.stack([np.where(sides == 0, a, b), np.where(sides == 0, b, a)], axis=1)
        terminal, hit = env.step(selected, active)
        miss = active & ((terminal == 0) | (terminal == 1))
        contact = active & ((hit >= 0) | miss)
        np.testing.assert_array_equal(env.ball[contact, 1], landing[contact])
        np.testing.assert_array_equal(env.ticks[contact] - launch_tick[contact], pending_ticks[contact])
        assert not np.any((hit >= 0) & active & (margin > 0)), 'A mechanically unreachable flight was caught.'
        for role, belongs in [('candidate', receiver != sides), ('opponent', receiver == sides)]:
            ended_shot = contact & ~from_serve & belongs
            shots[role]['completed_margin'].extend(margin[ended_shot].tolist())
            scored = miss & belongs
            points[role]['serve'] += int((scored & from_serve).sum())
            points[role]['forced_after_return'] += int((scored & ~from_serve & (margin > 0)).sum())
            points[role]['reachable_after_return'] += int((scored & ~from_serve & (margin <= 0)).sum())
        for i in np.flatnonzero(miss):
            point_records.append(dict(seed=int(seeds[(i % (n//2))//5]), rally=int(i%5+1),
                candidate_side=int(sides[i]), winner='candidate' if terminal[i] == sides[i] else 'opponent',
                returns=int(env.returns[i]), tick=int(env.ticks[i]), preceding_serve=bool(from_serve[i]),
                preceding_flight_ticks=int(pending_ticks[i]), landing=float(landing[i]),
                receiver_release_margin=float(margin[i]),
                classification='serve' if from_serve[i] else 'forced_after_return' if margin[i] > 0 else 'reachable_after_return'))
        returned = active & (hit >= 0)
        if returned.any():
            ids = np.flatnonzero(returned)
            r, t, land, gap = flight(env.ball[ids], env.paddles[ids])
            receiver[ids], pending_ticks[ids], landing[ids], margin[ids] = r, t, land, gap
            launch_tick[ids] = env.ticks[ids]
            from_serve[ids] = False
            for role, belongs in [('candidate', hit[ids] == sides[ids]), ('opponent', hit[ids] != sides[ids])]:
                ix = ids[belongs]
                shots[role]['offset'].extend((env.ball[ix, 1] - env.paddles[ix, hit[ix]]).tolist())
                shots[role]['speed'].extend(np.abs(env.ball[ix, 2]).tolist())
                shots[role]['margin'].extend(margin[ix].tolist())
                scorable = (env.ticks[ix] + pending_ticks[ix] <= RALLY_TICK_LIMIT) & (terminal[ix] < 0)
                shots[role]['scorable_margin'].extend(margin[ix[scorable]].tolist())
        done = active & (terminal >= 0)
        outcome[done], lengths[done], durations[done], reasons[done] = terminal[done], env.returns[done], env.ticks[done], env.draw_reason[done]
        if not (outcome < 0).any():
            break
    bootstrap = np.random.default_rng(2203).integers(0, len(seeds), (1000, len(seeds)))
    result = summarize(outcome, sides, lengths, durations, reasons, seeds, bootstrap)
    assert sum(points['candidate'].values()) == result['wins']
    assert sum(points['opponent'].values()) == result['losses']
    result['point_mechanism'] = points
    result['shot_mechanism'] = {}
    for role, data in shots.items():
        result['shot_mechanism'][role] = dict(
            returns=len(data['offset']), absolute_offset=distribution(np.abs(data['offset'])),
            horizontal_speed=distribution(data['speed']), receiver_release_margin=distribution(data['margin']),
            forced_at_release=int((np.array(data['margin']) > 0).sum()),
            scorable_shots=len(data['scorable_margin']), scorable_forced_at_release=int((np.array(data['scorable_margin']) > 0).sum()),
            completed_return_flights=len(data['completed_margin']), completed_forced_at_release=int((np.array(data['completed_margin']) > 0).sum()))
    result['points'] = point_records
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2201)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--safe-defender', action='store_true', help='Stress-test against the constructed universal-coverage defender')
    args = parser.parse_args()
    torch.set_num_threads(1)
    seeds = (np.random.default_rng(args.seed).choice(100_000_000, args.count, replace=False) + 1_400_000_000).tolist()
    verification = check_reachability()
    paths = {'contender': ROOT/'training/results/fully-learned/candidate/best.pt',
             'prior-ppo': ROOT/'training/results/training-improvement/runs/focus-outcome-stage2-s73/best.pt',
             'selfplay-v2': ROOT/'training/results/v2/ppo.pt'}
    opponents = references()
    policies = {name: load_mlp(path) for name, path in paths.items()}
    sources = [ROOT/'src/game.ts', ROOT/'training/scripts/env.py', Path(__file__), *paths.values()]
    if args.safe_defender:
        from attack_geometry import safe_defender
        opponents = {'safe-defender': safe_defender}
        policies['attacker'] = lambda obs: scripted(obs, True)
        sources.append(ROOT/'training/scripts/attack_geometry.py')
    if args.output is None:
        args.output = ROOT/'training/results/attack-audit'/('safe-defender.json' if args.safe_defender else 'points.json')
    rows = []
    for name, actor in policies.items():
        for opponent, policy in opponents.items():
            result = analyze(actor, policy, seeds)
            rows.append(dict(candidate=name, opponent=opponent, **result))
            print(name, opponent, result['wins'], result['losses'], result['draws'], result['point_mechanism'], flush=True)
    evidence = dict(rules='pong-v3',purpose='Development-only point-mechanism diagnosis; no rule or weight changes',
        seeds=seeds, seed_sampler=args.seed, inference='PyTorch first argmax; exact current vector engine and model loop detection',
        source_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sources},
        interpretation='Forced means impossible to catch from the receiver position at the preceding shot release, even with immediate ideal movement. It can still result from avoidable earlier recovery. Reachable does not assert an unforced strategic error or identify intent. Draws and initial serves are separate.',
        verification=verification, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
