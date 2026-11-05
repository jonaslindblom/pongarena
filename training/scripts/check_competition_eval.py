"""Verify paired outcomes and seed-level uncertainty without using fresh holdouts."""
import numpy as np
import torch

from competition_eval import evaluate_competition, load_mlp, summarize, validate_seeds
from env import scripted
from evaluate import evaluate
from policy import ROOT


def main():
    torch.set_num_threads(1)
    for invalid in [[], [1, 1], [-1], [2**32], [1.5], [True]]:
        try:
            validate_seeds(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted invalid or duplicated seed clusters: {invalid}')

    actor = load_mlp(ROOT / 'training/results/v2/ppo.pt')
    opponents = {'selfplay-v2': actor, 'attacker': lambda obs: scripted(obs, True)}
    # Existing published test seeds are used only to check evaluator equivalence.
    seeds = list(range(2_200_000_000, 2_200_000_005))
    old = evaluate(actor, seeds[0], len(seeds), opponents)
    current = evaluate_competition(actor, seeds, opponents, bootstrap_samples=500)
    for name, previous in old.items():
        row = current['opponents'][name]
        for key, expected in previous.items():
            assert row[key] == expected, (name, key, row[key], expected)
        assert sum(x['paired_point_margin'] for x in row['per_seed']) == row['wins'] - row['losses']
        for side in ('0', '1'):
            for key in ['wins', 'losses', 'draws', 'loop_draws', 'timeout_draws']:
                assert sum(x['sides'][side][key] for x in row['per_seed']) == row['sides'][side][key]
    mirror = current['opponents']['selfplay-v2']
    assert mirror['paired_match_draws'] == len(seeds)
    assert mirror['paired_match_wins'] == 0
    assert mirror['point_margin_ci95'] == [0., 0.]
    assert mirror['score_rate_ci95'] == [.5, .5]
    assert not current['primary_targets_pass_this_block']
    print('PASS: 100 rally outcomes reproduce the established evaluator; identical policies tie all paired matches.')

    # Perfectly correlated outcomes within each seed must stay together. One seed
    # wins all ten rallies, the other loses all ten: two samples, not twenty.
    sides = np.repeat([0, 1], 10)
    won = np.tile(np.repeat([True, False], 5), 2)
    outcome = np.where(won, sides, 1 - sides)
    rng = np.random.default_rng(1741)
    indices = rng.integers(0, 2, (2000, 2))
    row = summarize(outcome, sides, np.full(20, 4), np.full(20, 100), np.zeros(20, int), [3, 7], indices)
    assert row['point_margin_ci95'] == [-1., 1.]
    assert row['score_rate_ci95'] == [0., 1.]
    assert [x['paired_point_margin'] for x in row['per_seed']] == [10, -10]
    assert row['paired_match_win_rate'] == .5
    assert not row['acceptance']['passed']
    again = summarize(outcome, sides, np.full(20, 4), np.full(20, 100), np.zeros(20, int), [3, 7], indices)
    assert row == again
    print('PASS: bootstrap preserves whole-seed dependence; draw-inclusive majority and interval criteria reject insufficient evidence.')


if __name__ == '__main__':
    main()
