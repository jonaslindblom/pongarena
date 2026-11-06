"""Shared Classic evaluation summaries on explicit, paired serve seeds.

Policies must be fixed, deterministic and stateless. Bootstrap units are whole
serve seeds, retaining both sides and all five rallies together. This module
never chooses a checkpoint or generates final-test seeds.
"""
import numpy as np
from evaluate import run_policy
from env import serve


def validate_seeds(seeds):
    values = list(seeds)
    if not values or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                         or not 0 <= int(v) < 2**32 for v in values):
        raise ValueError('Supply a nonempty explicit list of uint32 integer seeds.')
    values = [int(v) for v in values]
    if len(set(values)) != len(values):
        raise ValueError('Seed clusters must be unique; duplicate seeds overstate evidence.')
    return values


def openings(seeds):
    frames, sides = [], []
    for side in (0, 1):
        for seed in seeds:
            f = None
            for _ in range(5):
                f = serve(seed, f)
                frames.append(f)
                sides.append(side)
    return frames, np.array(sides)


def decisions(policy, obs, active):
    selected = np.asarray(run_policy(policy, obs, active))
    if selected.shape != (len(obs),) or not np.isin(selected, [0, 1, 2]).all():
        raise ValueError('Policy must return one legal up/stay/down action per observation.')
    return selected.astype(np.int64)




def counts(won, lost, drawn, reasons):
    n = len(won)
    return dict(rallies=n, wins=int(won.sum()), losses=int(lost.sum()), draws=int(drawn.sum()),
                loop_draws=int((reasons == 1).sum()), timeout_draws=int((reasons == 2).sum()),
                point_rate=float(won.mean()), score_rate=float((won + .5 * drawn).mean()),
                point_margin=float((won.astype(int) - lost.astype(int)).mean()))


def summarize(outcome, sides, lengths, durations, reasons, seeds, bootstrap_indices):
    if np.any(outcome < 0):
        raise AssertionError('Every Classic rally must terminate by its fixed deadline.')
    won, lost, drawn = outcome == sides, outcome == 1 - sides, outcome == 2
    if int(drawn.sum()) != int((reasons > 0).sum()):
        raise AssertionError('Every draw must have exactly one recorded reason.')
    count = len(seeds)
    match_margin = (won.astype(int) - lost.astype(int)).reshape(2, count, 5).sum(-1)
    paired_margin = match_margin.sum(0)
    decisive = lengths[~drawn]
    result = counts(won, lost, drawn, reasons)
    result.update(match_wins=int((match_margin > 0).sum()), match_losses=int((match_margin < 0).sum()),
                  match_draws=int((match_margin == 0).sum()),
                  paired_match_wins=int((paired_margin > 0).sum()), paired_match_losses=int((paired_margin < 0).sum()),
                  paired_match_draws=int((paired_margin == 0).sum()),
                  paired_match_win_rate=float((paired_margin > 0).mean()),
                  median_decisive_returns=float(np.median(decisive)) if len(decisive) else None,
                  mean_returns=float(lengths.mean()), mean_ticks=float(durations.mean()),
                  early_finish_rate=float(((lengths < 2) & ~drawn).mean()))
    result['sides'] = {str(side): counts(won[sides == side], lost[sides == side], drawn[sides == side],
                                       reasons[sides == side]) for side in (0, 1)}
    # Each seed contributes ten rallies, not ten independent observations.
    bootstrap_margin = (paired_margin[bootstrap_indices] / 10).mean(axis=1)
    interval = np.quantile(bootstrap_margin, [.025, .975])
    result['point_margin_ci95'] = interval.tolist()
    result['score_rate_ci95'] = ((interval + 1) / 2).tolist()
    records = []
    for i, seed in enumerate(seeds):
        record = dict(seed=seed, paired_point_margin=int(paired_margin[i]), sides={})
        for side in (0, 1):
            ids = slice((side * count + i) * 5, (side * count + i + 1) * 5)
            record['sides'][str(side)] = dict(
                **counts(won[ids], lost[ids], drawn[ids], reasons[ids]),
                point_margin_total=int(match_margin[side, i]),
                rally_outcomes=np.where(won[ids], 1, np.where(lost[ids], -1, 0)).tolist(),
                draw_reasons=reasons[ids].tolist(), ticks=durations[ids].tolist())
        records.append(record)
    result['per_seed'] = records
    criteria = dict(majority_paired_matches=result['paired_match_win_rate'] > .5,
                    positive_margin_ci=result['point_margin_ci95'][0] > 0,
                    wins_exceed_losses_on_both_sides=all(v['wins'] > v['losses'] for v in result['sides'].values()))
    result['acceptance'] = dict(passed=all(criteria.values()), criteria=criteria)
    return result
