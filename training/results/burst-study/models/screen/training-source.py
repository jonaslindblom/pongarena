"""Compare raw-input model sizes on an isolated burst-Pong prototype.

All actors receive identical pooled DAgger data, minibatch orders and epoch
budgets. The geometric teacher is used for labels only; exports are plain MLPs.
This is a bounded learnability study, not an optimal-capacity result.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = {'micro': [16], 'tiny': [16, 16], 'small': [32, 32],
                 'medium': [64, 64], 'large': [128, 128]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor(hidden):
    layers = []
    widths = [8, *hidden, 5]
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layer = nn.Linear(a, b)
        nn.init.orthogonal_(layer.weight, gain=.01 if i == len(widths)-2 else np.sqrt(2))
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if i < len(widths)-2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def model_stats(net):
    layers = [layer for layer in net if isinstance(layer, nn.Linear)]
    return dict(architecture=[layers[0].in_features]+[layer.out_features for layer in layers],
                params=sum(p.numel() for p in net.parameters()),
                macs=sum(layer.in_features*layer.out_features for layer in layers))


def actions(net, observations):
    with torch.no_grad():
        return net(torch.from_numpy(observations)).argmax(-1).numpy()


def export(net, path, rules_name):
    nodes, weights, current = [], [], 'observation'
    for i, layer in enumerate(net):
        output = 'action_scores' if i == len(net)-1 else f'h{i}'
        if isinstance(layer, nn.Linear):
            weights.extend([numpy_helper.from_array(layer.weight.detach().numpy(), f'w{i}'),
                            numpy_helper.from_array(layer.bias.detach().numpy(), f'b{i}')])
            nodes.append(helper.make_node('Gemm', [current, f'w{i}', f'b{i}'], [output], transB=1))
        elif isinstance(layer, nn.Tanh):
            nodes.append(helper.make_node('Tanh', [current], [output]))
        else:
            raise ValueError(f'Unsupported learned layer {layer}')
        current = output
    graph = helper.make_graph(nodes, 'Rally Lab burst prototype learned actor',
        [helper.make_tensor_value_info('observation', 1, [1, 8])],
        [helper.make_tensor_value_info('action_scores', 1, [1, 5])], weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    helper.set_model_props(model, dict(rules=rules_name, training='Pooled PyTorch DAgger size study',
        fixed_features='none', architecture='-'.join(map(str, model_stats(net)['architecture']))+' Tanh'))
    onnx.checker.check_model(model)
    onnx.save(model, path)


def tracker(obs):
    error = obs[:, 1]-obs[:, 4]
    return np.where(error < -.015, 0, np.where(error > .015, 2, 1)).astype(np.int64)


def synthetic(rng, n, energy_max):
    x = rng.integers(30, 691, n)/2
    y = rng.integers(8, 313, n)/2
    vx = rng.integers(10, 27, n)/2*rng.choice([-1, 1], n)
    vy = rng.integers(-9, 10, n)
    own, other = rng.integers(15, 146, (2, n))
    energy = rng.integers(0, energy_max+1, (2, n))/energy_max
    return np.stack([(x-180)/180, (y-80)/80, vx/6, vy/6,
                     (own-80)/80, (other-80)/80, energy[0], energy[1]], 1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rules', required=True)
    p.add_argument('--module', default='burst_rules')
    p.add_argument('--architectures', nargs='+', choices=ARCHITECTURES, default=list(ARCHITECTURES))
    p.add_argument('--init-seeds', type=int, nargs='+', default=[611, 617])
    p.add_argument('--seed', type=int, default=619)
    p.add_argument('--rounds', type=int, default=24)
    p.add_argument('--envs', type=int, default=128)
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--replay-rounds', type=int, default=4)
    p.add_argument('--synthetic', type=int, default=8192)
    p.add_argument('--eval-every', type=int, default=6)
    p.add_argument('--dev-seeds', type=int, default=8)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory; retained results are immutable.')
    args.out.mkdir(parents=True)
    rules_module = importlib.import_module(args.module)
    rules = rules_module.CONFIGS[args.rules]
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    teacher = rules_module.make_policy(rules, 'attack')
    defender = rules_module.make_policy(rules, 'defend')
    interceptor = rules_module.make_policy(rules, 'interceptor')
    opponents = [('tracker', tracker), ('interceptor', interceptor), ('attack', teacher)]
    collection_opponents = [teacher, defender, interceptor, tracker]
    development = (np.random.default_rng(20261211).choice(10_000_000, args.dev_seeds, replace=False)+2_810_000_000).tolist()
    models = []
    for name in args.architectures:
        for seed in args.init_seeds:
            torch.manual_seed(seed)
            net = actor(ARCHITECTURES[name])
            path = args.out/f'{name}-s{seed}'
            path.mkdir()
            models.append(dict(name=name, seed=seed, net=net, path=path,
                optimizer=torch.optim.Adam(net.parameters(), lr=.001), best=-float('inf')))
    cfg = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    cfg.update(rules=asdict(rules), development_seeds=development,
        architectures={name: model_stats(actor(ARCHITECTURES[name])) for name in args.architectures},
        method='Identical pooled DAgger data and minibatch orders for every actor; cross entropy; plain inference',
        sources={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
                 str(Path(rules_module.__file__).relative_to(ROOT)):digest(Path(rules_module.__file__))})
    (args.out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    (args.out/'training-source.py').write_bytes(Path(__file__).read_bytes())
    env = rules_module.BurstArena(args.envs, rules, seed=args.seed, loop_detection=False)
    model_ids = rng.integers(0, len(models), args.envs)
    opponent_ids = rng.integers(0, len(collection_opponents), args.envs)
    replay_x, replay_y = [], []
    started, total, labels_total = time.monotonic(), 0, 0
    energy_max = int(getattr(rules, 'energy_max', getattr(rules, 'capacity', 96)))

    def evaluate(round_index, diagnostics):
        for m, diagnostic in zip(models, diagnostics):
            rows = {name:rules_module.run_pair(rules, m['net'], opponent, development)['summary']
                    for name, opponent in opponents}
            score = float(np.mean([row['point_margin'] for row in rows.values()]))
            row = dict(round=round_index, env_steps=total, labels=labels_total,
                seconds=time.monotonic()-started, development_margin=score, opponents=rows, **diagnostic)
            with (m['path']/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            torch.save(m['net'].state_dict(), m['path']/'last.pt')
            if score > m['best']:
                m['best'] = score
                torch.save(m['net'].state_dict(), m['path']/'best.pt')
                export(m['net'], m['path']/'best.onnx', args.rules)
                (m['path']/'selection.json').write_text(json.dumps(row, indent=2)+'\n')
            print(m['name'], m['seed'], 'round', round_index, 'margin', round(score, 3),
                ' | '.join(f'{name} {r["wins"]}/{r["losses"]}/{r["draws"]}' for name, r in rows.items()), flush=True)

    evaluate(0, [dict(cross_entropy=None, training_accuracy=None) for _ in models])
    for iteration in range(1, args.rounds+1):
        xs, ys = [], []
        beta = max(.05, 1-iteration/12)
        for _ in range(args.horizon):
            obs = env.obs()
            labels = teacher(obs)
            learned = np.empty(args.envs, int)
            for code in np.unique(model_ids):
                mask = model_ids == code
                learned[mask] = actions(models[code]['net'], obs[mask])
            chosen = np.where(rng.random(args.envs) < beta, labels, learned)
            opposing_obs = env.obs(1)
            opposing = np.empty(args.envs, int)
            for code in np.unique(opponent_ids):
                mask = opponent_ids == code
                opposing[mask] = collection_opponents[code](opposing_obs[mask])
            # Both perspective labels enter one common dataset, independent of
            # which network generated the trajectory or action.
            xs.extend([obs, opposing_obs])
            ys.extend([labels, teacher(opposing_obs)])
            outcomes, _ = env.step(np.stack([chosen, opposing], 1))
            ids = np.flatnonzero(outcomes >= 0)
            env.reset(ids)
            model_ids[ids] = rng.integers(0, len(models), len(ids))
            opponent_ids[ids] = rng.integers(0, len(collection_opponents), len(ids))
        broad = synthetic(rng, args.synthetic, energy_max)
        xs.append(broad)
        ys.append(teacher(broad))
        new_x, new_y = np.concatenate(xs), np.concatenate(ys)
        replay_x.append(new_x)
        replay_y.append(new_y)
        replay_x, replay_y = replay_x[-args.replay_rounds:], replay_y[-args.replay_rounds:]
        x, y = torch.from_numpy(np.concatenate(replay_x)), torch.from_numpy(np.concatenate(replay_y))
        diagnostics = [dict(loss=0., correct=0, seen=0) for _ in models]
        for m in models:
            m['optimizer'].param_groups[0]['lr'] = .001*(.4+.6*(1-(iteration-1)/args.rounds))
        for _ in range(args.epochs):
            # Same actual minibatches, not merely the same count.
            for ids in torch.from_numpy(rng.permutation(len(x))).split(2048):
                for m, d in zip(models, diagnostics):
                    logits = m['net'](x[ids])
                    loss = nn.functional.cross_entropy(logits, y[ids])
                    m['optimizer'].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(m['net'].parameters(), 1.)
                    m['optimizer'].step()
                    d['loss'] += float(loss.detach())*len(ids)
                    d['correct'] += int((logits.argmax(-1) == y[ids]).sum())
                    d['seen'] += len(ids)
        total += args.envs*args.horizon
        labels_total += len(new_x)
        rows = [dict(cross_entropy=d['loss']/d['seen'], training_accuracy=d['correct']/d['seen'],
                     teacher_action_probability=beta, optimization_examples=d['seen']) for d in diagnostics]
        print('round', iteration, 'shared_labels', labels_total,
              'losses', [round(d['cross_entropy'], 3) for d in rows], flush=True)
        if iteration % args.eval_every == 0 or iteration == args.rounds:
            evaluate(iteration, rows)
    completed = dict(seconds=time.monotonic()-started, env_steps=total, labels=labels_total,
        models={m['path'].name:dict(**model_stats(m['net']), best_development_margin=m['best'],
            onnx_sha256=digest(m['path']/'best.onnx'), checkpoint_sha256=digest(m['path']/'best.pt')) for m in models})
    (args.out/'completed.json').write_text(json.dumps(completed, indent=2)+'\n')
    print('Completed', args.out, flush=True)


if __name__ == '__main__':
    main()
