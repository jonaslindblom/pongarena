"""Audit whether current integer Pong physics permit an invariant safe defense.

This is a diagnostic script, not an arena controller or a submitted model.
No game rule, reference opponent, or model weight is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from competition_eval import evaluate_competition, load_mlp
from env import reflect, scripted
from policy import ROOT
from reference import advance


def legal_shots():
    rows = []
    for height in range(4, 157):
        for offset in range(-15, 16):
            paddle = height - offset
            if not 15 <= paddle <= 145:
                continue
            vx = 5 + int(np.floor(8 * abs(offset) / 15 + .5))
            vy = int(np.floor(offset / 3 + .5))
            ticks = int(np.ceil(330 / vx))
            landing = int(reflect(height + ticks * vy))
            rows.append(dict(height=height, offset=offset, paddle=paddle, vx=vx,
                             vy=vy, ticks=ticks, landing=landing))
    return rows


SHOTS = legal_shots()
SAFE = np.zeros((157, 2), dtype=int)
for height in range(4, 157):
    rows = [row for row in SHOTS if row['height'] == height]
    SAFE[height] = [max(15, max(row['landing'] - 15 - 2 * row['ticks'] for row in rows)),
                    min(145, min(row['landing'] + 15 + 2 * row['ticks'] for row in rows))]


def safe_defender(obs):
    """Stateless six-input defense: intercept, then recover to the safe band."""
    x = np.rint(obs[:, 0].astype(np.float64) * 180 + 180)
    y = np.rint(obs[:, 1].astype(np.float64) * 80 + 80)
    vx = np.rint(obs[:, 2].astype(np.float64) * 6)
    vy = np.rint(obs[:, 3].astype(np.float64) * 6)
    own = np.rint(obs[:, 4].astype(np.float64) * 80 + 80)
    distance = np.where(vx < 0, x - 15, 345 - x)
    ticks = np.maximum(1, np.ceil(distance / np.maximum(1, np.abs(vx))))
    landing = reflect(y + ticks * vy).astype(int)
    low, high = SAFE[landing].T
    target = np.where(vx < 0, np.clip(landing, 15, 145), np.clip(own, low, high))
    error = target - own
    tolerance = np.where(vx < 0, 1, 0)
    return np.where(error < -tolerance, 0, np.where(error > tolerance, 2, 1)).astype(np.int64)


def verify_geometry():
    centers = np.arange(15, 146)
    # Exact reachable sets, including staying still and clamp-then-reverse paths.
    reachable = np.zeros((67, 131, 131), dtype=bool)
    reachable[0] = np.eye(131, dtype=bool)
    for ticks in range(1, 67):
        for start in range(131):
            previous = centers[reachable[ticks - 1, start]]
            following = np.clip(previous[:, None] + [-2, 0, 2], 15, 145)
            reachable[ticks, start, np.unique(following).astype(int) - 15] = True
    exact_safe = np.ones((157, 131), dtype=bool)
    received_cases = []
    recovery_slack = []
    for row in SHOTS:
        height, landing, ticks = row['height'], row['landing'], row['ticks']
        catch_positions = np.abs(centers - landing) <= 15
        can_catch = reachable[ticks, :, catch_positions].any(axis=0)
        exact_safe[height] &= can_catch
        low, high = SAFE[height]
        received_cases.extend((int(p), landing, ticks) for p in centers[(centers >= low) & (centers <= high)])
        low, high = SAFE[landing]
        distance = max(low - row['paddle'], row['paddle'] - high, 0)
        recovery_slack.append(2 * ticks - distance)
        safe_positions = (centers >= low) & (centers <= high)
        assert reachable[ticks, row['paddle'] - 15, safe_positions].any(), row
        # Independently replay every outgoing flight through the scalar engine.
        for side in (0, 1):
            frame = dict(tick=0, rally=1, rallyTick=0, x=15 if side == 0 else 345,
                         y=height, vx=row['vx'] * (1 if side == 0 else -1), vy=row['vy'],
                         paddles=[80, 80], points=[0, 0], returns=0, event='return', scores=[None, None])
            for tick in range(1, ticks + 1):
                frame = advance(frame, [[0, 1, 0], [0, 1, 0]])
                assert (frame['event'] in ['point', 'return']) == (tick == ticks), (row, tick)
            assert frame['y'] == landing, row
    for height in range(4, 157):
        low, high = SAFE[height]
        assert np.array_equal(exact_safe[height], (centers >= low) & (centers <= high)), height
    # Verify the actual greedy movement rule on every safe integer start for
    # every legal incoming shot, not merely a continuous speed-distance bound.
    cases = np.array(received_cases)
    paddle, landing, ticks = cases.T.copy()
    target = np.clip(landing, 15, 145)
    for step in range(1, 67):
        error = target - paddle
        movement = np.where(error < -1, -2, np.where(error > 1, 2, 0))
        paddle = np.where(step <= ticks, np.clip(paddle + movement, 15, 145), paddle)
    assert np.all(np.abs(paddle - landing) <= 15)
    # The same nearest-band movement used by safe_defender must recover after
    # every possible legal return, regardless of how its contact was chosen.
    paddle = np.array([row['paddle'] for row in SHOTS])
    ticks = np.array([row['ticks'] for row in SHOTS])
    destination = np.array([row['landing'] for row in SHOTS])
    low, high = SAFE[destination].T
    for step in range(1, 67):
        target = np.clip(paddle, low, high)
        error = target - paddle
        movement = np.where(error < 0, -2, np.where(error > 0, 2, 0))
        paddle = np.where(step <= ticks, np.clip(paddle + movement, 15, 145), paddle)
    assert np.all((paddle >= low) & (paddle <= high))
    serve_cases = []
    for center in [32, 56, 80, 104, 128]:
        for y in range(center - 10, center + 11):
            for vy in [*range(-5, 0), *range(1, 6)]:
                landing = int(reflect(y + 33 * vy))
                assert reachable[33, 80 - 15, np.abs(centers - landing) <= 15].any()
                low, high = SAFE[landing]
                assert reachable[33, 80 - 15, (centers >= low) & (centers <= high)].any()
                serve_cases.append(dict(y=y, vy=vy, landing=landing))
    landings = np.array([row['landing'] for row in serve_cases])
    receiver = np.full(len(landings), 80)
    nonreceiver = np.full(len(landings), 80)
    low, high = SAFE[landings].T
    for _ in range(33):
        error = np.clip(landings, 15, 145) - receiver
        receiver = np.clip(receiver + np.where(error < -1, -2, np.where(error > 1, 2, 0)), 15, 145)
        error = np.clip(nonreceiver, low, high) - nonreceiver
        nonreceiver = np.clip(nonreceiver + np.where(error < 0, -2, np.where(error > 0, 2, 0)), 15, 145)
    assert np.all(np.abs(receiver - landings) <= 15)
    assert np.all((nonreceiver >= low) & (nonreceiver <= high))
    worst = int(np.argmin(recovery_slack))
    return dict(legal_returns=len(SHOTS), integer_contact_heights=153,
                safe_band_empty_heights=0, minimum_safe_band_width=int(np.min(SAFE[4:, 1] - SAFE[4:, 0])),
                universal_fixed_position_interval=[int(SAFE[4:, 0].max()), int(SAFE[4:, 1].min())],
                all_safe_bands_intersect_75_to_85=bool(np.all((SAFE[4:, 0] <= 85) & (SAFE[4:, 1] >= 75))),
                exact_reachable_set_checks=len(SHOTS) * 131,
                scalar_flights_checked_both_sides=2 * len(SHOTS),
                greedy_incoming_cases=len(received_cases), greedy_recovery_cases=len(SHOTS),
                serve_height_slope_cases=len(serve_cases), serve_ticks=33,
                greedy_serve_catching_and_nonreceiver_recovery=True,
                minimum_recovery_spare_movement_units=int(min(recovery_slack)),
                minimum_recovery_spare_movement_witness=SHOTS[worst],
                intervals=[dict(height=y, low=int(SAFE[y, 0]), high=int(SAFE[y, 1])) for y in range(4, 157)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2201)
    parser.add_argument('--out', type=Path, default=ROOT / 'training/results/attack-audit')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    geometry = verify_geometry()
    geometry['rules_sha256'] = hashlib.sha256((ROOT / 'src/game.ts').read_bytes()).hexdigest()
    geometry['scope'] = ('All integer contact heights 4..156 and all legal contact offsets, paddle bounds 15..145, '
                         'speed 2, exact rounding/wall/contact order; initial standard serves. Safe defense is a '
                         'diagnostic scripted construction, not a learned competition submission.')
    geometry['inductive_argument'] = [
        'Standard serves and all future legal physics states have integer coordinates and velocities; '
        'the enumeration covers a superset of all reachable legal contact heights and offsets.',
        'For each opponent contact height, the universal safe band contains exactly the starting paddle '
        'centers from which every legal returned shot can be caught. Exact movement reachability includes '
        'staying, two-unit motion, clamping, and clamp-then-reverse paths.',
        'The incoming greedy chase catches every legal shot from every safe-band integer start. '
        'Its landing prediction is invariant during that flight.',
        'After any legal return, the hitter can move into the safe band corresponding to the known next '
        'contact height before the opponent hits. The strict nearest-band movement rule achieves this '
        'for every enumerated return, with both players moving on each intervening tick.',
        'Both first-serve roles satisfy the base case: the receiver catches the serve, while the other '
        'player reaches the safe band before the first return.',
        'By induction, either player can avoid every miss against any sequence of legal opponent actions '
        'from a standard serve. Timeout and valid loop exits may draw, but cannot create a conceded point.',
    ]
    (args.out / 'geometry.json').write_text(json.dumps(geometry, indent=2) + '\n')
    print(json.dumps({k: v for k, v in geometry.items() if k != 'intervals'}), flush=True)
    seeds = (np.random.default_rng(args.seed).choice(100_000_000, args.count, replace=False) + 1_400_000_000).tolist()
    contenders = {'contender': load_mlp(ROOT / 'training/results/fully-learned/candidate/best.pt'),
                  'selfplay-v2': load_mlp(ROOT / 'training/results/v2/ppo.pt'),
                  'attacker': lambda obs: scripted(obs, True),
                  'interceptor': lambda obs: scripted(obs), 'safe-defender': safe_defender}
    result = evaluate_competition(safe_defender, seeds, opponents=contenders)
    result['purpose'] = 'Diagnostic defense ceiling, not trained-model validation or a new rules release.'
    (args.out / 'geometry-matches.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {key: row[key] for key in ['wins', 'losses', 'draws', 'loop_draws', 'timeout_draws']}
                      for name, row in result['opponents'].items()}), flush=True)


if __name__ == '__main__':
    main()
