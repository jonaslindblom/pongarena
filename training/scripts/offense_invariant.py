"""Conservative controlled-invariant proof for the isolated rule candidates.

G[y,p] means a defender at p can survive every geometrically legal return from
opponent contact height y, then select a return and recover into G again.
The adversary may choose any legal shot regardless of its previous position.
Surviving sets therefore prove defense; empty sets do not prove winning attack.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from offense_rules import CONFIGS, shot

ROOT = Path(__file__).resolve().parents[2]
HEIGHTS = np.arange(4, 157)
CENTERS = np.arange(15, 146)
MOVEMENTS = np.arange(-1, 2)
FLIGHTS = np.array(sorted({math.ceil(330 / vx) for vx in range(5, 14)}))


def reachability():
    """Exact discrete reachability, including waiting and clamp-then-reverse."""
    reachable = np.zeros((67, 131, 131), bool)
    reachable[0] = np.eye(131, dtype=bool)
    last_predecessors = [[[] for _ in MOVEMENTS] for _ in CENTERS]
    for old, p in enumerate(CENTERS):
        for action in (-1, 0, 1):
            q = int(np.clip(p + 2 * action, 15, 145))
            m = int(np.sign(q - p))
            last_predecessors[q - 15][m + 1].append(old)
    for t in range(1, 67):
        for q in range(131):
            predecessors = set(sum(last_predecessors[q], []))
            reachable[t, :, q] = reachable[t - 1, :, list(predecessors)].any(axis=0)
    # Separate set representation: shifts, waits and saturated boundary bits.
    mask = (1 << 131) - 1
    for origin in range(131):
        bits = 1 << origin
        for t in range(67):
            assert bits == sum(1 << int(q) for q in np.flatnonzero(reachable[t, origin]))
            bits = (bits | (bits >> 2) | ((bits << 2) & mask)
                    | (1 if bits & 3 else 0)
                    | (1 << 130 if bits & (3 << 129) else 0))
    final = np.zeros((len(FLIGHTS), 131, 131 * 3), bool)
    for ti, t in enumerate(FLIGHTS):
        for q in range(131):
            for mi in range(3):
                pred = last_predecessors[q][mi]
                if pred:
                    final[ti, :, q * 3 + mi] = reachable[t - 1, :, pred].any(axis=0)
    legal_motion = np.array([[bool(p) for p in row] for row in last_predecessors])
    return reachable, final, legal_motion


def candidates(rules, legal_motion):
    """For every contact y, return q and last movement, map outgoing flight."""
    q = np.repeat(CENTERS, 3)
    m = np.tile(MOVEMENTS, 131)
    d = HEIGHTS[:, None] - q
    legal = (np.abs(d) <= 15) & legal_motion.reshape(1, -1)
    vx, vy = shot(d, m, rules)
    flight = np.ceil(330 / vx).astype(int)
    z = (HEIGHTS[:, None] + flight * vy - 4) % 304
    destination = (4 + np.where(z <= 152, z, 304 - z)).astype(int)
    times = np.searchsorted(FLIGHTS, flight)
    # Invalid contacts can produce non-candidate speeds; use valid values only.
    times[~legal] = 0
    destination[~legal] = 4
    return legal, times, destination - 4


def solve(rules, reachable, final, legal_motion):
    legal, times, destinations = candidates(rules, legal_motion)
    fixed = np.ones((153, 131), bool)
    counts = [int(fixed.sum())]
    own_center = np.repeat(np.arange(131), 3)
    opponent_choices = [np.unique(np.stack([times[y, legal[y]], destinations[y, legal[y]]], axis=1), axis=0)
                        for y in range(153)]
    while True:
        # recovery[t,z,p]: can p reach any currently safe center for contact z?
        recovery = np.stack([fixed @ reachable[t].T for t in FLIGHTS])
        good_return = legal & recovery[times, destinations, own_center[None, :]]
        # interception[t,z,p]: can p produce any of z's good returns in t ticks?
        interception = np.stack([(final[ti] @ good_return.T).T for ti in range(len(FLIGHTS))])
        next_fixed = np.array([interception[choices[:, 0], choices[:, 1], :].all(axis=0)
                               for choices in opponent_choices])
        assert not (next_fixed & ~fixed).any(), 'The safety iteration must be monotonically decreasing.'
        counts.append(int(next_fixed.sum()))
        if np.array_equal(fixed, next_fixed):
            break
        fixed = next_fixed
    # A final recomputation establishes closure, not just an iteration cutoff.
    assert np.array_equal(fixed, next_fixed)
    receiver_failures, nonreceiver_failures = [], []
    initial_y = [center + delta for center in (32, 56, 80, 104, 128) for delta in range(-10, 11)]
    serve_t = int(np.flatnonzero(FLIGHTS == 33)[0])
    for y in initial_y:
        for vy in list(range(-5, 0)) + list(range(1, 6)):
            z = (y + 33 * vy - 4) % 304
            landing = 4 + (z if z <= 152 else 304 - z)
            if not interception[serve_t, landing - 4, 80 - 15]:
                receiver_failures.append([y, vy])
            if not (reachable[33, 80 - 15] & fixed[landing - 4]).any():
                nonreceiver_failures.append([y, vy])
    proven = bool(fixed.any(axis=1).all() and not receiver_failures and not nonreceiver_failures)
    return dict(rules=asdict(rules), proves_never_concede_from_all_standard_serves=proven,
                fixed_point_iterations=len(counts) - 1, retained_state_counts=counts,
                contact_heights_with_safe_center=int(fixed.any(axis=1).sum()),
                minimum_safe_centers_per_height=int(fixed.sum(axis=1).min()),
                legal_contact_motion_options=int(legal.sum()),
                distinct_incoming_landing_time_options=sum(len(x) for x in opponent_choices),
                serve_cases_per_role=len(initial_y) * 10,
                receiver_failure_count=len(receiver_failures), nonreceiver_failure_count=len(nonreceiver_failures),
                receiver_failure_examples=receiver_failures[:10], nonreceiver_failure_examples=nonreceiver_failures[:10],
                safe_centers={str(y): CENTERS[row].tolist() for y, row in zip(HEIGHTS, fixed)})


class InvariantDefender:
    """Stateless diagnostic controller, not a learned submission.

    Dynamic-programming action tables realize the proved invariant. Every
    action uses only the six current observations, including time-to-contact
    inferred from ball state. No chosen target persists across ticks.
    """
    def __init__(self, rules):
        reachable, final, legal_motion = reachability()
        self.proof = solve(rules, reachable, final, legal_motion)
        if not self.proof['proves_never_concede_from_all_standard_serves']:
            raise ValueError(f'No complete standard-serve guarantee for {rules.name}')
        fixed = np.zeros((153, 131), bool)
        for y, centers in self.proof['safe_centers'].items():
            fixed[int(y) - 4, np.array(centers) - 15] = True
        legal, times, destinations = candidates(rules, legal_motion)
        recovery = np.stack([fixed @ reachable[t].T for t in FLIGHTS])
        good_return = legal & recovery[times, destinations, np.repeat(np.arange(131), 3)[None, :]]
        next_center = np.clip(np.arange(131)[:, None] + 2 * np.arange(-1, 2), 0, 130)
        movement = np.sign(next_center - np.arange(131)[:, None])
        order = np.array([1, 0, 2])  # Stable preference: stay, up, down.
        self.incoming = np.ones((67, 153, 131), dtype=np.int8)
        self.outgoing = np.ones_like(self.incoming)
        incoming_safe = np.zeros_like(self.incoming, dtype=bool)
        outgoing_safe = np.zeros_like(incoming_safe)
        outgoing_safe[0] = fixed
        for t in range(1, 67):
            choices = good_return[:, next_center * 3 + movement + 1] if t == 1 else incoming_safe[t - 1][:, next_center]
            incoming_safe[t] = choices.any(axis=2)
            self.incoming[t] = order[choices[:, :, order].argmax(axis=2)]
            # Outside the proved region, gracefully chase rather than index an
            # invalid action; this fallback is never needed by the proof.
            fallback = np.where(HEIGHTS[:, None] < CENTERS - 1, 0,
                                np.where(HEIGHTS[:, None] > CENTERS + 1, 2, 1))
            self.incoming[t, ~incoming_safe[t]] = fallback[~incoming_safe[t]]
            choices = outgoing_safe[t - 1][:, next_center]
            outgoing_safe[t] = choices.any(axis=2)
            self.outgoing[t] = order[choices[:, :, order].argmax(axis=2)]
        # Tables must realize every existential choice used by the proof.
        checked = 0
        for y in range(153):
            for ti, destination in np.unique(np.stack([times[y, legal[y]], destinations[y, legal[y]]], axis=1), axis=0):
                assert incoming_safe[FLIGHTS[ti], destination, fixed[y]].all()
                checked += int(fixed[y].sum())
        for y, option in zip(*np.nonzero(good_return)):
            assert outgoing_safe[FLIGHTS[times[y, option]], destinations[y, option], option // 3]
        self.proof['constructive_interception_checks'] = checked
        self.proof['constructive_return_recovery_checks'] = int(good_return.sum())

    def __call__(self, obs):
        x = np.rint(obs[:, 0].astype(np.float64) * 180 + 180).astype(int)
        y = np.rint(obs[:, 1].astype(np.float64) * 80 + 80).astype(int)
        vx = np.rint(obs[:, 2].astype(np.float64) * 6).astype(int)
        vy = np.rint(obs[:, 3].astype(np.float64) * 6).astype(int)
        own = np.rint(obs[:, 4].astype(np.float64) * 80 + 80).astype(int)
        incoming = vx < 0
        distance = np.where(incoming, x - 15, 345 - x)
        ticks = np.clip(np.ceil(distance / np.maximum(1, np.abs(vx))).astype(int), 1, 66)
        z = (y + ticks * vy - 4) % 304
        landing = np.where(z <= 152, z, 304 - z)
        return np.where(incoming, self.incoming[ticks, landing, own - 15],
                        self.outgoing[ticks, landing, own - 15]).astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matches', action='store_true', help='Also corroborate constructive defenders in played diagnostic rallies.')
    args = parser.parse_args()
    reachable, final, legal_motion = reachability()
    result = dict(method='greatest fixed point of conservative contact-height/own-center safety game',
                  interpretation='Nonempty closed sets plus serve base cases prove defense. Empty sets do not prove forceable offense.',
                  reachability_checks=67 * 131,
                  source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                 for p in ('training/scripts/offense_rules.py', 'training/scripts/offense_invariant.py')},
                  configs={})
    for name, rules in CONFIGS.items():
        row = solve(rules, reachable, final, legal_motion)
        if row['proves_never_concede_from_all_standard_serves']:
            row = InvariantDefender(rules).proof
        result['configs'][name] = row
        print(name, json.dumps({k: v for k, v in row.items() if k not in ('safe_centers', 'rules')}), flush=True)
    assert result['configs']['baseline']['proves_never_concede_from_all_standard_serves']
    out = ROOT / 'training/results/offense-study/verification/invariant.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    if args.matches:
        import torch
        from competition_eval import load_mlp
        from offense_eval import make_policy, run_pair
        torch.set_num_threads(1)
        seeds = (np.random.default_rng(6713).choice(100_000_000, 32, replace=False) + 1_700_000_000).tolist()
        rows = dict(purpose='Diagnostic corroboration of invariant, not an untouched model test', seeds=seeds, configs={})
        contender = load_mlp(ROOT / 'training/results/fully-learned/candidate/best.pt')
        v2 = load_mlp(ROOT / 'training/results/v2/ppo.pt')
        for name in ('baseline', 'angle8'):
            rules = CONFIGS[name]
            defender = InvariantDefender(rules)
            rows['configs'][name] = {}
            opponents = dict(attack=make_policy(rules, 'attack'), defend=make_policy(rules, 'defend'),
                             contender=contender, v2=v2, self=defender)
            for kind, opponent in opponents.items():
                match = run_pair(rules, defender, opponent, seeds)
                assert match['summary']['losses'] == 0, (name, kind, match['summary'])
                rows['configs'][name][kind] = match
                print(name, kind, {k: match['summary'][k] for k in ('wins', 'losses', 'draws')}, flush=True)
        (out.parent / 'invariant-matches.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
