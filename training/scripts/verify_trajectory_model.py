"""Verify fixed trajectory features and a selected learned target ONNX export.

Only uses development seeds. It never changes weights, rules or selection.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch

from env import Arena, scripted
from policy import Actor, ROOT, actions
from reference import serve
from train_target import TargetActor, export_target
from trajectory_features import TrajectoryFeatures, check as check_geometry


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_in_arena(path):
    code = '''import { readFileSync } from 'node:fs';
import { validateModel } from './src/validate.ts';
const bytes = readFileSync(process.argv[1]);
const s = validateModel(bytes);
console.log(JSON.stringify({bytes: bytes.length, params: s.params, macs: s.macs,
  nodes: s.nodes, branched: s.architecture.widths === null}));'''
    result = subprocess.run(['node', '--import', 'tsx', '--input-type=module', '-e', code, str(path)],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def reachable_probes(actor, count=8192):
    rng = np.random.default_rng(1969)
    seeds = rng.choice(100_000_000, 24, replace=False) + 1_200_000_000
    frames, sides = [], []
    for side in (0, 1):
        for seed in seeds:
            frame = None
            for _ in range(5):
                frame = serve(int(seed), frame)
                frames.append(frame)
                sides.append(side)
    sides = np.array(sides)
    old = Actor()
    old.load_state_dict(torch.load(ROOT / 'training/results/v2/ppo.pt', weights_only=True))
    old.eval()
    env = Arena(len(frames), loop_detection=True)
    env.load(frames)
    active = np.ones(env.n, bool)
    rows = []
    for tick in range(1600):
        left, right = env.obs(), env.obs(1)
        ours = np.where(sides[:, None] == 0, left, right)
        theirs = np.where(sides[:, None] == 0, right, left)
        if tick % 4 == 0:
            rows.extend([ours[active].copy(), theirs[active].copy()])
        a = actions(actor, ours)
        b = np.where(np.arange(env.n) % 2 == 0, scripted(theirs, True), actions(old, theirs))
        selected = np.stack([np.where(sides == 0, a, b), np.where(sides == 0, b, a)], 1)
        terminal, _ = env.step(selected, active)
        active &= terminal < 0
        if not active.any():
            break
    observations = np.concatenate(rows)
    assert len(observations) >= count
    selected = rng.choice(len(observations), count, replace=False)
    return observations[selected], seeds.tolist(), len(observations)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, default=ROOT / 'training/results/training-improvement/candidate')
    p.add_argument('--onnx', type=Path)
    p.add_argument('--out', type=Path, default=ROOT / 'training/results/training-improvement/verification')
    args = p.parse_args()
    args.candidate = args.candidate.resolve()
    onnx_path = (args.onnx or args.candidate / 'best.onnx').resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    config = json.loads((args.candidate / 'config.json').read_text())
    assert config['rules'] == 'pong-v3' and config['features']
    actor = TargetActor(tuple(config['target_widths']), features=TrajectoryFeatures()).eval()
    actor.load_state_dict(torch.load(args.candidate / 'best.pt', weights_only=True))
    fixed = TrajectoryFeatures()
    for name, value in fixed.state_dict().items():
        assert torch.equal(actor.features.state_dict()[name], value), f'Fixed feature changed: {name}'
    assert all(not p.requires_grad for p in actor.features.parameters())
    assert all(p.requires_grad for p in actor.net.parameters())
    expected = TargetActor(tuple(config['target_widths']), features=TrajectoryFeatures())
    for name in ['feedback', 'target_scores', 'deadband']:
        assert torch.equal(getattr(actor, name), getattr(expected, name)), f'Fixed feedback changed: {name}'

    protocol = json.loads((ROOT / 'training/results/training-improvement/protocol.json').read_text())
    assert sha256(ROOT / 'src/game.ts') == protocol['rules_unchanged']['sha256']
    assert sha256(ROOT / 'training/v3/game.ts') == protocol['rules_unchanged']['contract_sha256']
    assert sha256(ROOT / 'training/scripts/env.py') == protocol['primary_targets']['attacker']['source_sha256']
    assert sha256(ROOT / 'public/models/selfplay-v2.onnx') == protocol['primary_targets']['selfplay-v2']['sha256']

    with tempfile.TemporaryDirectory(prefix='trajectory-verification-') as scratch:
        scratch = Path(scratch)
        log = io.StringIO()
        with redirect_stdout(log):
            check_geometry(scratch)
        geometry = json.loads((scratch / 'check.json').read_text())
        (args.out / 'trajectory-geometry.json').write_text(json.dumps(geometry, indent=2) + '\n')
        (args.out / 'trajectory-geometry.log').write_text(log.getvalue())
        regenerated = scratch / 'regenerated.onnx'
        export_target(actor, regenerated)
        exported = onnx.load(onnx_path)
        # Metadata may document publication; the graph and every embedded tensor
        # must still be exactly the selected checkpoint's graph.
        assert exported.graph.SerializeToString() == onnx.load(regenerated).graph.SerializeToString()
    metadata = {p.key: p.value for p in exported.metadata_props}
    assert metadata.get('rules') == 'pong-v3'
    assert 'fixed' in metadata.get('architecture', '').lower()
    assert 'physics prior' in metadata.get('training', '').lower()
    graph = validate_in_arena(onnx_path)
    parameters = dict(fixed_trajectory=sum(p.numel() for p in actor.features.parameters()),
                      learned_target_head=sum(p.numel() for p in actor.net.parameters()),
                      fixed_feedback=sum(getattr(actor, name).numel() for name in ['feedback', 'target_scores', 'deadband']))
    assert sum(parameters.values()) == graph['params']

    observations, seeds, source_count = reachable_probes(actor)
    with torch.no_grad():
        expected = actor(torch.from_numpy(observations)).numpy()
        assert torch.isfinite(actor(torch.zeros(1, 6))).all()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), opts, providers=['CPUExecutionProvider'])
    native = np.array([session.run(None, {'observation': row[None]})[0][0] for row in observations])
    agreement = expected.argmax(-1) == native.argmax(-1)
    result = dict(candidate=str(args.candidate.relative_to(ROOT)) if args.candidate.is_relative_to(ROOT) else str(args.candidate),
                  onnx=str(onnx_path.relative_to(ROOT)) if onnx_path.is_relative_to(ROOT) else str(onnx_path),
                  checkpoint_sha256=sha256(args.candidate / 'best.pt'), onnx_sha256=sha256(onnx_path),
                  source_sha256={name: sha256(ROOT / 'training/scripts' / name) for name in ['trajectory_features.py', 'train_target.py', 'verify_trajectory_model.py']},
                  rules_and_reference_opponents_unchanged=True, graph_matches_checkpoint=True,
                  fixed_weights_equal_original_construction=True, parameters=parameters, arena_validator=graph,
                  inference_contract='raw6 -> fixed trajectory ReLU features -> learned scalar target -> fixed own-paddle feedback ->3scores',
                  geometry=geometry, observations=len(observations), source_reachable_observations=source_count,
                  probe_seeds=seeds, probe_purpose='Development-only reachable-state parity, not final model evaluation',
                  action_agreement=float(agreement.mean()), action_mismatches=int((~agreement).sum()),
                  max_abs_score_error=float(np.max(np.abs(expected-native))),
                  classification_weight_reproduction=config.get('classification_weight', .01 if '.01 cross entropy' in config['loss'] else None))
    (args.out / 'selected-model.json').write_text(json.dumps(result, indent=2) + '\n')
    np.testing.assert_allclose(native, expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(native.argmax(-1), expected.argmax(-1))
    print(json.dumps({k: result[k] for k in ['onnx_sha256', 'parameters', 'arena_validator', 'observations', 'action_agreement', 'max_abs_score_error']}, indent=2))
    print('PASS: fixed geometry, selected graph, original rules/opponents, and native ONNX actions verified.')


if __name__ == '__main__':
    main()
