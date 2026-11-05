"""Matched-budget learned-weight search for isolated offensive-rule candidates.

All inference parameters belong to an ordinary six-input Tanh MLP. Geometry is
used by training opponents only. Nothing here modifies the released arena.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import torch

from competition_eval import openings
from offense_rules import CONFIGS, OffenseArena
from policy import Actor, ROOT, actions, export


class Population:
    def __init__(self, states, opponent_count):
        self.size, self.opponent_count = len(states), opponent_count
        self.layers = [(torch.stack([s[f'{i}.weight'] for s in states]).transpose(1, 2),
                        torch.stack([s[f'{i}.bias'] for s in states])[:, None]) for i in (0, 2, 4)]

    def actions(self, obs):
        n = len(obs) // (self.opponent_count * self.size)
        x = torch.from_numpy(obs.reshape(self.opponent_count, self.size, n, 6)
                             .transpose(1, 0, 2, 3).reshape(self.size, self.opponent_count*n, 6))
        with torch.no_grad():
            for i, (weight, bias) in enumerate(self.layers):
                x = torch.bmm(x, weight) + bias
                if i < 2:
                    x = x.tanh()
        return x.argmax(-1).numpy().reshape(self.size, self.opponent_count, n).transpose(1, 0, 2).reshape(-1)


def evaluate_population(states, seeds, rules, opponents):
    population = Population(states, len(opponents))
    frames, sides = openings(seeds)
    width = len(frames)
    frames = frames * (len(opponents) * population.size)
    sides = np.tile(sides, len(opponents) * population.size)
    opponent_id = np.repeat(np.arange(len(opponents)), population.size * width)
    env = OffenseArena(len(frames), rules, loop_detection=True)
    env.load(frames)
    outcome = np.full(env.n, -1)
    decisions = 0
    for _ in range(1600):
        active = outcome < 0
        left, right = env.obs(0), env.obs(1)
        ours = np.where(sides[:, None] == 0, left, right)
        theirs = np.where(sides[:, None] == 0, right, left)
        a, b = population.actions(ours), np.ones(env.n, dtype=np.int64)
        for code, opponent in enumerate(opponents.values()):
            ix = active & (opponent_id == code)
            if ix.any():
                b[ix] = opponent(theirs[ix])
        selected = np.stack([np.where(sides == 0, a, b), np.where(sides == 0, b, a)], 1)
        terminal, _ = env.step(selected, active)
        done = active & (terminal >= 0)
        outcome[done] = terminal[done]
        decisions += int(active.sum())
        if (outcome >= 0).all():
            break
    assert (outcome >= 0).all()
    points = np.where(outcome == sides, 1, np.where(outcome == 1-sides, -1, 0))
    points = points.reshape(len(opponents), population.size, 2, len(seeds), 5)
    rows = []
    for i in range(population.size):
        row = {}
        for j, name in enumerate(opponents):
            q = points[j, i]
            margins = q.sum(-1)
            row[name] = dict(wins=int((q == 1).sum()), losses=int((q == -1).sum()), draws=int((q == 0).sum()),
                             point_margin=float(q.mean()), score_rate=float((q == 1).mean()),
                             match_win_rate=float((margins > 0).mean()),
                             paired_match_win_rate=float((margins.sum(0) > 0).mean()),
                             seed_margins=q.sum(axis=(0, 2)).tolist())
        rows.append(row)
    return rows, decisions


def fitness(row):
    # Symmetric point rewards with an additional ordinary-game win term. An
    # equally weighted league avoids the unbreakable baseline defender making
    # all attacking improvements invisible to a minimum-opponent objective.
    return (float(np.mean([r['point_margin'] + .3*r['match_win_rate'] for r in row.values()])),
            float(np.mean([r['score_rate'] for r in row.values()])))


def clone(state):
    return {k: v.clone() for k, v in state.items()}


def experimental_export(actor, path, rules):
    export(actor, path)
    model = onnx.load(path)
    model.graph.name = f'Rally Lab experimental {rules.name} actor'
    onnx.helper.set_model_props(model, {
        'rules': f'offense-study-{rules.name}', 'architecture': '6-64-64-3 Tanh',
        'training': 'PyTorch weights; antithetic evolutionary fine-tuning',
        'status': 'experimental; not a released rule version',
    })
    onnx.checker.check_model(model)
    onnx.save(model, path)


def verify_export(directory, rules):
    import onnxruntime as ort

    actor = Actor().eval()
    actor.load_state_dict(torch.load(directory/'best.pt', weights_only=True))
    model = onnx.load(directory/'best.onnx')
    assert [n.op_type for n in model.graph.node] == ['Gemm', 'Tanh', 'Gemm', 'Tanh', 'Gemm']
    weights = {w.name: onnx.numpy_helper.to_array(w) for w in model.graph.initializer}
    for i in (0, 2, 4):
        assert np.array_equal(weights[f'w{i}'], actor[i].weight.detach().numpy())
        assert np.array_equal(weights[f'b{i}'], actor[i].bias.detach().numpy())
    props = {p.key: p.value for p in model.metadata_props}
    assert props['rules'] == f'offense-study-{rules.name}'
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(directory/'best.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    obs = np.random.default_rng(22061105).uniform(-2, 2, (8192, 6)).astype(np.float32)
    with torch.no_grad():
        expected = actor(torch.from_numpy(obs)).numpy()
    actual = np.concatenate([session.run(None, {'observation': row[None]})[0] for row in obs])
    mismatch = int((expected.argmax(1) != actual.argmax(1)).sum())
    result = dict(probes=len(obs), action_mismatches=mismatch,
                  maximum_score_error=float(np.abs(expected-actual).max()),
                  checkpoint_parameters_exactly_match_graph=True, graph_ops=[n.op_type for n in model.graph.node],
                  rules=props['rules'], onnx_sha256=hashlib.sha256((directory/'best.onnx').read_bytes()).hexdigest())
    (directory/'export-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    assert mismatch == 0, result


def training_seeds(seed, confirmation_count, development_count):
    rng = np.random.default_rng(seed)
    confirmation = rng.choice(1_000_000_000, confirmation_count, replace=False).tolist()
    development = (np.random.default_rng(seed+1).choice(100_000_000, development_count, replace=False)
                   + 1_600_000_000).tolist()
    return rng, confirmation, development


def train(args, name):
    from offense_eval import make_policy

    out = args.out/name
    if (out/'config.json').exists():
        raise FileExistsError(f'Use a fresh directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng, confirmation, development = training_seeds(args.seed, args.confirm_seeds, args.dev_seeds)
    rules = CONFIGS[name]
    actor, anchor = Actor(), Actor()
    state = torch.load(args.initial, weights_only=True)
    actor.load_state_dict(state)
    anchor.load_state_dict(state)
    actor.eval(); anchor.eval()
    current = clone(state)
    opponents = {
        'adaptive-attacker': make_policy(rules, 'attack'),
        'adaptive-defender': make_policy(rules, 'defend'),
        'frozen-contender': lambda obs: actions(anchor, obs),
    }
    sources = ['training/scripts/offense_rules.py', 'training/scripts/offense_eval.py',
               'training/scripts/train_offense.py', 'training/scripts/env.py']
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(rules=vars(rules), architecture=[6, 64, 64, 3], fixed_inference_features=False,
                  initial_sha256=hashlib.sha256(args.initial.read_bytes()).hexdigest(),
                  sources={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                  confirmation_seeds=confirmation, development_seeds=development,
                  opponents=list(opponents),
                  selection='mean(point margin + 0.3 ordinary five-rally game win rate); tie mean rally score rate',
                  final_test_used=False,
                  comparison='identical initial weights, RNG, mutation count, seed sets and generations; active decisions vary by rule')
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    torch.save(current, out/'initial.pt')
    experimental_export(actor, out/'initial.onnx', rules)
    started, best, total = time.monotonic(), (-10., -10.), 0

    def validate(generation, training=None):
        nonlocal best
        actor.load_state_dict(current)
        rows, validation_decisions = evaluate_population([current], development, rules, opponents)
        result, key = rows[0], fitness(rows[0])
        row = dict(generation=generation, seconds=time.monotonic()-started,
                   training_decisions=total, validation_decisions=validation_decisions,
                   fitness=list(key), opponents=result, training=training)
        with (out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if key > best:
            best = key
            torch.save(current, out/'best.pt')
            experimental_export(actor, out/'best.onnx', rules)
            (out/'selection.json').write_text(json.dumps(row, indent=2)+'\n')
        print(f'{name} gen {generation} | {total:,} decisions | dev {key} | '
              + ' | '.join(f'{n} {v["wins"]}/{v["losses"]}/{v["draws"]}' for n, v in result.items()), flush=True)

    if args.check:
        obs = np.random.default_rng(17).uniform(-2, 2, (len(opponents)*2*30, 6)).astype(np.float32)
        stacked = Population([current, current], len(opponents)).actions(obs)
        assert np.array_equal(stacked, actions(actor, obs)), 'Population inference differs from ordinary MLP.'
        from offense_eval import run_pair
        result, _ = evaluate_population([current, current], confirmation[:2], rules, opponents)
        for opponent_name, opponent in opponents.items():
            expected = run_pair(rules, lambda obs: actions(actor, obs), opponent, confirmation[:2])['summary']
            for member in result:
                for key in ('wins', 'losses', 'draws'):
                    assert member[opponent_name][key] == expected[key], (opponent_name, key, member, expected)
        print(f'{name}: population inference and identical members match independent pair evaluator.', flush=True)

    validate(0)
    for generation in range(1, args.generations+1):
        states = [current]
        for pair in range(args.pairs):
            scale = args.sigma*[.5, 1., 2., 4.][pair % 4]
            noise = {k: torch.randn_like(v)*scale for k, v in current.items()}
            states += [{k: v+sign*noise[k] for k, v in current.items()} for sign in (-1, 1)]
        seeds = rng.choice(1_000_000_000, args.screen_seeds, replace=False).tolist()
        screened, decisions = evaluate_population(states, seeds, rules, opponents)
        total += decisions
        top = sorted(range(1, len(states)), key=lambda i: fitness(screened[i]), reverse=True)[:2]
        verified, decisions = evaluate_population([current]+[states[i] for i in top], confirmation, rules, opponents)
        total += decisions
        chosen = max(range(len(verified)), key=lambda i: fitness(verified[i]))
        accepted = chosen > 0 and fitness(verified[chosen]) > fitness(verified[0])
        if accepted:
            current = states[top[chosen-1]]
        training = dict(generation=generation, accepted=accepted, screen_seeds=seeds,
                        confirmation=verified[chosen], fitness=fitness(verified[chosen]),
                        decisions=total, seconds=time.monotonic()-started)
        with (out/'search.jsonl').open('a') as f:
            f.write(json.dumps(training)+'\n')
        torch.save(current, out/'last.pt')
        if generation % args.eval_every == 0 or generation == args.generations:
            validate(generation, training)
        else:
            print(f'{name} gen {generation} | accepted {accepted} | train {fitness(verified[chosen])}', flush=True)
    (out/'completed.json').write_text(json.dumps(dict(seconds=time.monotonic()-started,
        training_decisions=total, best_fitness=best, generations=args.generations), indent=2)+'\n')
    verify_export(out, rules)


def plot(root, configs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in configs:
        path = root/name/'metrics.jsonl'
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        xs = [r['generation'] for r in rows]
        axes[0].plot(xs, [r['fitness'][0] for r in rows], marker='o', label=name)
        axes[1].plot(xs, [np.mean([v['score_rate'] for v in r['opponents'].values()]) for r in rows], marker='o', label=name)
    axes[0].set_ylabel('Mean point margin + 0.3 × game win rate')
    axes[1].set_ylabel('Mean rally scoring rate (draws included)')
    for ax in axes:
        ax.set_xlabel('Search generation')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Rule comparison: learned-weight fine-tuning on fixed development serves')
    fig.tight_layout()
    fig.savefig(root/'learning-curves.png', dpi=160)
    fig.savefig(root/'learning-curves.svg')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT/'training/results/offense-study/training/search')
    p.add_argument('--initial', type=Path, default=ROOT/'training/results/fully-learned/candidate/best.pt')
    p.add_argument('--configs', nargs='+', choices=CONFIGS, default=list(CONFIGS))
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--generations', type=int, default=20)
    p.add_argument('--pairs', type=int, default=4)
    p.add_argument('--screen-seeds', type=int, default=8)
    p.add_argument('--confirm-seeds', type=int, default=16)
    p.add_argument('--dev-seeds', type=int, default=32)
    p.add_argument('--sigma', type=float, default=.006)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--check', action='store_true')
    p.add_argument('--plot-only', action='store_true')
    args = p.parse_args()
    if min(args.generations, args.pairs, args.screen_seeds, args.confirm_seeds, args.dev_seeds, args.eval_every) < 1:
        p.error('all counts must be positive')
    if not np.isfinite(args.sigma) or args.sigma <= 0:
        p.error('sigma must be positive and finite')
    torch.set_num_threads(1)
    if not args.plot_only:
        for name in args.configs:
            train(args, name)
    plot(args.out, args.configs)


if __name__ == '__main__':
    main()
