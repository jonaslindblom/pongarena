"""Fixed-v3 competition evaluation on explicit, paired serve seeds.

Policies must be fixed, deterministic and stateless. Bootstrap units are whole
serve seeds, retaining both sides and all five rallies together. This module
never chooses a checkpoint or generates final-test seeds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from env import Arena, scripted
from evaluate import run_policy
from policy import Actor, OnnxPolicy, ROOT
from reference import RALLY_TICK_LIMIT, serve


PRIMARY_TARGETS = ('selfplay-v2', 'attacker')
ACCEPTANCE = {
    'paired_match_win_rate': '> 0.5, including draws in the denominator',
    'point_margin_ci95_lower': '> 0, bootstrap whole seeds with both sides',
    'each_side': 'rally wins > rally losses on both physical sides',
    'consistency': 'all criteria against both primary targets in each of three fresh seed blocks',
}


def references(native=False, include_v3=False):
    """Use exported reference weights in final tests; Torch is faster for development."""
    policies = {}
    for version in (['v2', 'v3'] if include_v3 else ['v2']):
        if native:
            actor = OnnxPolicy(ROOT / f'public/models/selfplay-{version}.onnx')
        else:
            actor = Actor()
            actor.load_state_dict(torch.load(ROOT / f'training/results/{version}/ppo.pt', weights_only=True))
            actor.eval()
        policies[f'selfplay-{version}'] = actor
    policies['attacker'] = lambda obs: scripted(obs, attack=True)
    policies['interceptor'] = lambda obs: scripted(obs)
    return policies


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


def runtime(policy):
    return 'native_onnxruntime' if isinstance(policy, OnnxPolicy) else 'pytorch' if isinstance(policy, nn.Module) else 'python_callable'


def counts(won, lost, drawn, reasons):
    n = len(won)
    return dict(rallies=n, wins=int(won.sum()), losses=int(lost.sum()), draws=int(drawn.sum()),
                loop_draws=int((reasons == 1).sum()), timeout_draws=int((reasons == 2).sum()),
                point_rate=float(won.mean()), score_rate=float((won + .5 * drawn).mean()),
                point_margin=float((won.astype(int) - lost.astype(int)).mean()))


def summarize(outcome, sides, lengths, durations, reasons, seeds, bootstrap_indices):
    if np.any(outcome < 0):
        raise AssertionError('Every v3 rally must terminate by its fixed deadline.')
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


def evaluate_competition(actor, seeds, opponents=None, bootstrap_seed=1741, bootstrap_samples=2000,
                         native_reference=False, include_v3=False):
    """Evaluate one frozen policy; returned pass flag concerns this seed block only.

    Pass any deterministic nn.Module or callable returning integer actions. Native
    final evaluation passes OnnxPolicy for the candidate and native_reference=True.
    Training/validation must not tune to final block outcomes.
    """
    seeds = validate_seeds(seeds)
    if bootstrap_samples < 1:
        raise ValueError('bootstrap_samples must be positive.')
    if opponents is None:
        opponents = references(native=native_reference, include_v3=include_v3)
    if not opponents:
        raise ValueError('At least one opponent is required.')
    frames, sides = openings(seeds)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_indices = rng.integers(0, len(seeds), (bootstrap_samples, len(seeds)))
    results = {}
    for name, opponent in opponents.items():
        env = Arena(len(frames), loop_detection=True)
        env.load(frames)
        outcome = np.full(len(frames), -1)
        lengths, durations, reasons = (np.zeros(len(frames), int) for _ in range(3))
        for _ in range(RALLY_TICK_LIMIT):
            active = outcome < 0
            left, right = env.obs(0), env.obs(1)
            ours = np.where(sides[:, None] == 0, left, right)
            theirs = np.where(sides[:, None] == 0, right, left)
            a, b = decisions(actor, ours, active), decisions(opponent, theirs, active)
            selected = np.stack([np.where(sides == 0, a, b), np.where(sides == 0, b, a)], axis=1)
            terminal, _ = env.step(selected, active)
            done = active & (terminal >= 0)
            outcome[done], lengths[done], durations[done] = terminal[done], env.returns[done], env.ticks[done]
            reasons[done] = env.draw_reason[done]
            if (outcome >= 0).all():
                break
        results[name] = summarize(outcome, sides, lengths, durations, reasons, seeds, bootstrap_indices)
    targets_present = all(name in results for name in PRIMARY_TARGETS)
    return dict(rules='pong-v3', seeds=seeds, seed_count=len(seeds),
                inference='Deterministic first argmax; exact contact-state loop draws; both sides/all five serves',
                candidate_runtime=runtime(actor), opponent_runtimes={name: runtime(opponent) for name, opponent in opponents.items()},
                bootstrap=dict(unit='whole seed including both sides and five serves', seed=bootstrap_seed,
                               samples=bootstrap_samples, confidence=.95, method='percentile'),
                acceptance_protocol=ACCEPTANCE, primary_targets=list(PRIMARY_TARGETS),
                primary_targets_pass_this_block=all(results[name]['acceptance']['passed'] for name in PRIMARY_TARGETS)
                    if targets_present else None,
                opponents=results)


def load_mlp(path, activation='tanh'):
    """Plain sequential Linear/activation state dictionaries, with no pickled code."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    weights = sorted((k for k in state if k.endswith('.weight')), key=lambda k: int(k.split('.')[0]))
    layers = []
    for i, key in enumerate(weights):
        if key != f'{2 * i}.weight':
            raise ValueError('CLI expects a plain sequential MLP checkpoint; pass custom modules to the Python API.')
        out_size, in_size = state[key].shape
        layers.append(nn.Linear(in_size, out_size))
        if i < len(weights) - 1:
            layers.append(nn.Tanh() if activation == 'tanh' else nn.ReLU())
    if not layers or layers[0].in_features != 6 or layers[-1].out_features != 3:
        raise ValueError('Expected the fixed six-input/three-output model interface.')
    actor = nn.Sequential(*layers)
    actor.load_state_dict(state)
    return actor.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--seeds-file', type=Path, required=True, help='JSON array of unique uint32 seeds; no implicit test set')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
    parser.add_argument('--include-v3', action='store_true')
    parser.add_argument('--native-reference', action='store_true', help='Always enabled for an ONNX candidate')
    parser.add_argument('--bootstrap-seed', type=int, default=1741)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    actor = OnnxPolicy(args.model) if args.model.suffix == '.onnx' else load_mlp(args.model, args.activation)
    result = evaluate_competition(actor, json.loads(args.seeds_file.read_text()),
                                  native_reference=args.native_reference or isinstance(actor, OnnxPolicy),
                                  include_v3=args.include_v3, bootstrap_seed=args.bootstrap_seed,
                                  bootstrap_samples=args.bootstrap_samples)
    result['model'] = str(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {k: r[k] for k in ['wins', 'losses', 'draws', 'point_margin_ci95',
                      'paired_match_win_rate', 'acceptance']} for name, r in result['opponents'].items()}, indent=2))


if __name__ == '__main__':
    main()
