"""Majority paired contests must not substitute for ordinary-game wins."""
import copy

from evaluate_frozen import apply_protocol


def main():
    row = dict(match_wins=10, match_losses=0, match_draws=10,
               acceptance=dict(passed=True, criteria=dict(majority_paired_matches=True,
                   positive_margin_ci=True, wins_exceed_losses_on_both_sides=True)))
    result = dict(opponents={name: copy.deepcopy(row) for name in ['selfplay-v2', 'attacker']})
    protocol = dict(acceptance=dict(ordinary_match_win_rate='> .5 including draws'),
                    primary_targets=['selfplay-v2', 'attacker'])
    apply_protocol(result, protocol)
    assert not result['primary_targets_pass_this_block']
    assert result['opponents']['attacker']['ordinary_match_win_rate'] == .5
    for value in result['opponents'].values():
        value.update(match_wins=11, match_draws=9)
    apply_protocol(result, protocol)
    assert result['primary_targets_pass_this_block']
    result['opponents']['attacker']['acceptance']['criteria']['positive_margin_ci'] = False
    apply_protocol(result, protocol)
    assert not result['primary_targets_pass_this_block']
    print('PASS: paired wins cannot hide ordinary-game draws; all earlier criteria still apply.')


if __name__ == '__main__':
    main()
