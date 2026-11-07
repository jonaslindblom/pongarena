"""Train a Pong policy from random weights, then improve it through game outcomes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from game import Arena, reflect
from teachers import teacher
from models import network, decide
from export import export_model
from evaluate import development_field

HERE = Path(__file__).resolve().parent


def synthetic(rng, n, contacts):
    raw = np.stack([rng.integers(15, 346, n), rng.integers(4, 157, n),
        rng.integers(5, 14, n) * rng.choice([-1, 1], n), rng.integers(-9, 10, n),
        rng.integers(15, 146, n), rng.integers(15, 146, n)], 1)
    # Half the synthetic positions emphasize the final approach to our paddle.
    m = n // 2 if contacts else 0
    raw[:m, 0] = rng.integers(16, 101, m)
    raw[:m, 2] = -np.abs(raw[:m, 2])
    ticks = np.ceil((raw[:m, 0]-15) / -raw[:m, 2])
    landing = reflect(raw[:m, 1] + ticks * raw[:m, 3])
    raw[:m, 4] = np.clip(landing + rng.integers(-24, 25, m), 15, 145)
    return ((raw - [180, 80, 0, 0, 80, 80]) / [180, 80, 6, 6, 80, 80]).astype(np.float32)


def train(config, output, name):
    if output.exists():
        raise FileExistsError('Choose a fresh output directory.')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    policy = network(config['widths'], config['activation'])
    params = sum(p.numel() for p in policy.parameters())
    macs = sum(a * b for a, b in zip(config['widths'], config['widths'][1:]))
    assert config['widths'][0] == 6 and config['widths'][-1] == 3
    assert params <= (256 if name == 'small' else 65536)
    assert macs <= (1024 if name == 'small' else 65536)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config['learning_rate'])
    labels_for = teacher('attack')
    opponents = [teacher(kind) for kind in ('attack', 'defend', 'interceptor')]
    arena = Arena(config['envs'], seed=config['seed'])
    opponent_ids = rng.integers(0, len(opponents), arena.n)
    replay_x, replay_y = [], []
    best = -2.
    started = time.monotonic()
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    def select(candidate, row):
        nonlocal best
        evaluation = development_field(candidate)
        score = float(np.mean([result['point_margin'] for result in evaluation.values()]))
        if not np.isfinite(score):
            raise ValueError('Development evaluation produced a non-finite score.')
        row.update(development_margin=score, opponents=evaluation)
        if score > best:
            best = score
            torch.save({'widths': config['widths'], 'activation': config['activation'],
                        'state_dict': candidate.state_dict()}, output / 'model.pt')
            method = ('PyTorch imitation of geometric scripts; random initialization' if row['phase'] == 'imitation'
                      else 'PyTorch imitation followed by PPO against scripts and past learned policies; random initialization')
            export_model(candidate, output / 'model.onnx', config['activation'], name.capitalize(), method)
            (output / 'selection.json').write_text(json.dumps(row, indent=2) + '\n')

    def record(row):
        with (output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'opponents'}), flush=True)

    for iteration in range(1, config['rounds'] + 1):
        xs, ys = [], []
        beta = max(config.get('teacher_floor', 0.), 1 - iteration / config['teacher_rounds'])
        for _ in range(config['horizon']):
            obs = arena.obs()
            labels = labels_for(obs)
            teacher_actions = labels
            actions = np.where(rng.random(arena.n) < beta, teacher_actions, decide(policy, obs))
            other = np.empty(arena.n, int)
            opposite = arena.obs(1)
            for i in range(len(opponents)):
                mask = opponent_ids == i
                if mask.any():
                    other[mask] = opponents[i](opposite[mask])
            xs.append(obs)
            ys.append(labels)
            outcome, _ = arena.step(np.stack([actions, other], 1))
            ended = np.flatnonzero(outcome >= 0)
            arena.reset(ended)
            opponent_ids[ended] = rng.integers(0, len(opponents), len(ended))
        broad = synthetic(rng, config['synthetic'], config['contacts'])
        xs.append(broad)
        ys.append(labels_for(broad))
        replay_x.append(np.concatenate(xs))
        replay_y.append(np.concatenate(ys))
        replay_x, replay_y = replay_x[-config['replay_rounds']:], replay_y[-config['replay_rounds']:]
        x = torch.from_numpy(np.concatenate(replay_x))
        y = torch.from_numpy(np.concatenate(replay_y))
        critical = (x[:, 2] < 0) & (x[:, 0] < -.45)
        importance = torch.where(critical, float(config.get('contact_weight', 1)), 1.)
        floor = config.get('learning_rate_floor', .15)
        optimizer.param_groups[0]['lr'] = config['learning_rate'] * (floor + (1 - floor) * (1 - (iteration - 1) / config['rounds']))
        total_loss, correct, seen = 0., 0, 0
        for _ in range(config['epochs']):
            for ids in torch.randperm(len(x)).split(config['batch_size']):
                logits = policy(x[ids])
                targets = y[ids]
                loss = (nn.functional.cross_entropy(logits, targets, reduction='none') * importance[ids]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Training produced a non-finite loss.')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                optimizer.step()
                total_loss += float(loss.detach()) * len(ids)
                correct += int((logits.argmax(-1) == targets).sum())
                seen += len(ids)
        row = {'phase': 'imitation', 'round': iteration, 'loss': total_loss / seen, 'accuracy': correct / seen}
        if iteration % config['evaluate_every'] == 0 or iteration == config['rounds']:
            select(policy, row)
        record(row)
    if not (output / 'model.onnx').exists():
        raise RuntimeError('Training did not select a model.')
    reinforcement = None
    if config.get('ppo', {}).get('rounds', 0):
        from ppo import finetune
        saved = torch.load(output / 'model.pt', map_location='cpu', weights_only=True)
        policy.load_state_dict(saved['state_dict'])
        # Use a separate repeatable random stream for outcome learning.
        reinforcement = finetune(policy, config['ppo'], config['seed'] + 1100,
                                 on_evaluate=select, on_progress=record)
    imitation_steps = config['rounds'] * config['envs'] * config['horizon']
    (output / 'run.json').write_text(json.dumps({'parameters': params, 'seconds': time.monotonic() - started,
        'environment_steps': imitation_steps + (reinforcement['environment_steps'] if reinforcement else 0),
        'imitation_environment_steps': imitation_steps,
        'ppo_seed': config['seed'] + 1100 if reinforcement else None,
        'reinforcement': reinforcement,
        'best_development_margin': best}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=['small', 'large'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--seed', type=int, help='Override initialization and training seed')
    parser.add_argument('--smoke', action='store_true', help='Tiny pipeline check; not a strength benchmark')
    args = parser.parse_args()
    config = json.loads((args.config or HERE / 'configs' / (args.size + '.json')).read_text())
    if args.seed is not None:
        config['seed'] = args.seed
    if args.smoke:
        config.update(rounds=1, envs=8, horizon=8, epochs=1, synthetic=64)
        if config.get('ppo'):
            config['ppo'].update(rounds=2, envs=8, horizon=64, epochs=1, batch_size=128, critic_warmup=0)
    train(config, args.out, args.size)


if __name__ == '__main__':
    main()
