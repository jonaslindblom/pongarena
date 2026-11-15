"""Train Quill's architecture independently, using the frozen learned Large teacher.

Programmed policies are never loaded here. Selection uses scripts and Large;
evaluate the frozen result against programmed models separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn

from evaluate import DEVELOPMENT_SEEDS, pair
from export import export_model
from game import Arena, RULES_ID
from models import network, decide
from ppo import finetune
from teachers import teacher
from train import synthetic

HERE = Path(__file__).resolve().parent
WIDTHS = [6, 12, 4, 3, 3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def learned_teacher(path):
    """Only the independently trained, recorded Large model may supply labels."""
    expected = json.loads((HERE / 'results.json').read_text())['models']['large']['sha256']
    if digest(path) != expected:
        raise ValueError('Teacher must be the recorded independently trained Large model.')
    model = onnx.load(path)
    arrays = {t.name: numpy_helper.to_array(t).copy() for t in model.graph.initializer}
    affine = [n for n in model.graph.node if n.op_type == 'Gemm']
    widths = [6] + [len(arrays[n.input[2]]) for n in affine]
    policy = network(widths, 'Relu')
    with torch.no_grad():
        for layer, node in zip([l for l in policy if isinstance(l, nn.Linear)], affine):
            layer.weight.copy_(torch.from_numpy(arrays[node.input[1]]))
            layer.bias.copy_(torch.from_numpy(arrays[node.input[2]]))
    return policy.eval().requires_grad_(False)


def distillation_loss(logits, targets, importance, temperature):
    # Soft preferences preserve ambiguity; all labels use only the six inputs.
    loss = nn.functional.kl_div(nn.functional.log_softmax(logits / temperature, -1),
                                targets, reduction='none').sum(-1)
    return (loss * importance).sum() / importance.sum() * temperature ** 2


def development(policy, opponents, seeds=DEVELOPMENT_SEEDS):
    if not seeds or any(not 1_000_000_000 <= s < 2_000_000_000 for s in seeds):
        raise ValueError('Selection requires development seeds.')
    results = {}
    for name, opponent in opponents.items():
        row = pair(policy, opponent, seeds)
        margins = np.asarray(row['paired_point_margin'])
        row['league_score'] = float(np.mean((margins > 0) + .5 * (margins == 0)))
        results[name] = {k: v for k, v in row.items() if not k.startswith('paired_')}
    # Match the actual paired-contest objective; rally score breaks exact ties.
    score = tuple(float(np.mean([r[key] for r in results.values()]))
                  for key in ('league_score', 'rally_score'))
    return score, results


def train(config, output, large_path, label_source='large', resume=None):
    if output.exists():
        raise FileExistsError('Choose a fresh output directory.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    large = learned_teacher(large_path)
    # Loading the teacher must not change the student's initialization stream.
    torch.manual_seed(config['seed'])
    policy = network(WIDTHS, 'Relu')
    if resume:
        saved = torch.load(resume / 'model.pt', map_location='cpu', weights_only=True)
        if saved['widths'] != WIDTHS or saved['activation'] != 'Relu':
            raise ValueError('The student architecture must remain fixed.')
        policy.load_state_dict(saved['state_dict'])
    opponents = {kind: teacher(kind) for kind in ('attack', 'defend', 'interceptor')}
    opponents['large'] = large
    output.mkdir(parents=True)
    record = config | {'widths': WIDTHS, 'activation': 'Relu', 'labels': label_source,
                       'teacher_sha256': digest(large_path), 'rules_id': RULES_ID,
                       'resume_sha256': digest(resume / 'model.onnx') if resume else None}
    (output / 'config.json').write_text(json.dumps(record, indent=2) + '\n')
    best = (-1., -1.)
    started = time.monotonic()

    def select(candidate, row):
        nonlocal best
        score, results = development(candidate, opponents)
        row.update(development_score=score, opponents=results)
        if score > best:
            best = score
            torch.save({'widths': WIDTHS, 'activation': 'Relu',
                        'state_dict': candidate.state_dict()}, output / 'model.pt')
            export_model(candidate, output / 'model.onnx', 'Relu', 'Quill-sized student',
                         'Random initialization; independent teacher imitation and outcome learning')
            (output / 'selection.json').write_text(json.dumps(row, indent=2) + '\n')

    def progress(row):
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'opponents'}), flush=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config['learning_rate'])
    arena = Arena(config['envs'], seed=config['seed'])
    opponent_list = list(opponents.values())
    opponent_ids = rng.integers(0, len(opponent_list), arena.n)
    sides = rng.integers(0, 2, arena.n)
    replay = []
    temperature = config['temperature'] if label_source == 'large' else 1.

    def targets_for(obs):
        with torch.no_grad():
            if label_source == 'large':
                return (large(torch.from_numpy(obs)) / temperature).softmax(-1).numpy()
        return np.eye(3, dtype=np.float32)[opponents['attack'](obs)]

    if resume:
        row = {'phase': 'resume', 'round': 0}
        select(policy, row)
        progress(row)
    for iteration in range(1, config['rounds'] + 1):
        samples, targets = [], []
        beta = max(.05, 1 - iteration / config['teacher_rounds'])
        for _ in range(config['horizon']):
            left, right = arena.obs(0), arena.obs(1)
            obs = np.where(sides[:, None] == 0, left, right)
            opposite = np.where(sides[:, None] == 0, right, left)
            labels = targets_for(obs)
            actions = np.where(rng.random(arena.n) < beta, labels.argmax(-1), decide(policy, obs))
            other = np.empty(arena.n, int)
            for i, opponent in enumerate(opponent_list):
                mask = opponent_ids == i
                if mask.any():
                    other[mask] = decide(opponent, opposite[mask])
            samples.append(obs)
            targets.append(labels)
            outcome, _ = arena.step(np.stack([np.where(sides == 0, actions, other),
                                               np.where(sides == 0, other, actions)], 1))
            ended = np.flatnonzero(outcome >= 0)
            arena.reset(ended)
            opponent_ids[ended] = rng.integers(0, len(opponent_list), len(ended))
            sides[ended] = rng.integers(0, 2, len(ended))
        broad = synthetic(rng, config['synthetic'], True)
        samples.append(broad)
        targets.append(targets_for(broad))
        replay.append((np.concatenate(samples), np.concatenate(targets)))
        replay = replay[-config['replay_rounds']:]
        x = torch.from_numpy(np.concatenate([r[0] for r in replay]))
        y = torch.from_numpy(np.concatenate([r[1] for r in replay]))
        critical = (x[:, 2] < 0) & (x[:, 0] < -.45)
        importance = torch.where(critical, float(config['contact_weight']), 1.)
        optimizer.param_groups[0]['lr'] = config['learning_rate'] * (.15 + .85 * (1 - (iteration - 1) / config['rounds']))
        losses, correct, seen = [], 0, 0
        for _ in range(config['epochs']):
            for ids in torch.randperm(len(x)).split(config['batch_size']):
                logits = policy(x[ids])
                loss = distillation_loss(logits, y[ids], importance[ids], temperature)
                if not torch.isfinite(loss):
                    raise ValueError('Distillation produced a non-finite loss.')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
                correct += int((logits.argmax(-1) == y[ids].argmax(-1)).sum())
                seen += len(ids)
        row = {'phase': 'distillation', 'round': iteration,
               'loss': float(np.mean(losses)), 'agreement': correct / seen}
        if iteration % config['evaluate_every'] == 0 or iteration == config['rounds']:
            select(policy, row)
        progress(row)
    if not (output / 'model.pt').exists():
        raise ValueError('No student checkpoint was selected.')
    reinforcement = None
    if config.get('ppo', {}).get('rounds', 0):
        saved = torch.load(output / 'model.pt', map_location='cpu', weights_only=True)
        policy.load_state_dict(saved['state_dict'])
        reinforcement = finetune(policy, config['ppo'], config['seed'] + 1100,
                                 on_evaluate=select, on_progress=progress)
    (output / 'run.json').write_text(json.dumps({
        'seconds': time.monotonic() - started, 'best_development_score': best,
        'parameters': sum(p.numel() for p in policy.parameters()),
        'imitation_environment_steps': config['rounds'] * config['envs'] * config['horizon'],
        'reinforcement': reinforcement, 'sha256': digest(output / 'model.onnx')}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'configs/quill-distill.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--labels', choices=['large', 'script'], default='large')
    parser.add_argument('--large', type=Path, default=HERE.parent / 'public/models/large.onnx')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config['seed'] = args.seed
    if args.smoke:
        config.update(rounds=1, envs=8, horizon=8, epochs=1, synthetic=64)
        if config.get('ppo'):
            config['ppo'].update(rounds=2, envs=8, horizon=64, epochs=1,
                                 batch_size=128, critic_warmup=0)
    train(config, args.out, args.large, args.labels, args.resume)


if __name__ == '__main__':
    main()
