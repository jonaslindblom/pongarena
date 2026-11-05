"""Check that a raw-input MLP export contains exactly its trainable checkpoint.

This excludes geometry features, feedback branches and other untrained inference
logic. It checks architecture and numerical correspondence, not the historical
origin of arbitrary weights; training provenance remains in the run records.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch

from competition_eval import load_mlp
from env import Arena, scripted
from policy import ROOT, actions, export
from reference import serve


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reachable_probes(actor):
    rng = np.random.default_rng(1970)
    seeds = rng.choice(100_000_000, 24, replace=False) + 1_300_000_000
    frames, sides = [], []
    for side in (0, 1):
        for seed in seeds:
            frame = None
            for _ in range(5):
                frame = serve(int(seed), frame)
                frames.append(frame)
                sides.append(side)
    sides = np.array(sides)
    old = load_mlp(ROOT / 'training/results/v2/ppo.pt')
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
        chosen = np.stack([np.where(sides == 0, a, b), np.where(sides == 0, b, a)], axis=1)
        terminal, _ = env.step(chosen, active)
        active &= terminal < 0
        if not active.any():
            break
    observations = np.concatenate(rows)
    indices = rng.choice(len(observations), min(8192, len(observations)), replace=False)
    return observations[indices], seeds.tolist(), len(observations)


def main():
    parser = argparse.ArgumentParser()
    baseline = ROOT / 'training/results/training-improvement/runs/focus-outcome-stage2-s73'
    parser.add_argument('--checkpoint', type=Path, default=baseline / 'best.pt')
    parser.add_argument('--onnx', type=Path, default=baseline / 'best.onnx')
    parser.add_argument('--output', type=Path, default=ROOT / 'training/results/fully-learned/baseline/model-check.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    actor = load_mlp(args.checkpoint)
    assert all(p.requires_grad for p in actor.parameters())
    assert not list(actor.buffers())
    with tempfile.TemporaryDirectory(prefix='plain-model-check-') as directory:
        regenerated = Path(directory) / 'model.onnx'
        export(actor, regenerated)
        original = onnx.load(args.onnx)
        assert original.graph.SerializeToString() == onnx.load(regenerated).graph.SerializeToString(), \
            'Export must contain exactly the plain trainable checkpoint, with no extra branches or calculations.'
    onnx.checker.check_model(original)
    assert {p.key: p.value for p in original.metadata_props}.get('rules') == 'pong-v3'
    code = '''import { readFileSync } from 'node:fs';
import { validateModel } from './src/validate.ts';
const bytes = readFileSync(process.argv[1]); const s = validateModel(bytes);
console.log(JSON.stringify({bytes:bytes.length,params:s.params,macs:s.macs,nodes:s.nodes,widths:s.architecture.widths}));'''
    verified = subprocess.run(['node', '--import', 'tsx', '--input-type=module', '-e', code, str(args.onnx.resolve())],
                              cwd=ROOT, capture_output=True, text=True, check=True)
    stats = json.loads(verified.stdout)
    assert stats['params'] == sum(p.numel() for p in actor.parameters())
    observations, seeds, source_count = reachable_probes(actor)
    with torch.no_grad():
        expected = actor(torch.from_numpy(observations)).numpy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=['CPUExecutionProvider'])
    actual = np.array([session.run(None, {'observation': row[None]})[0][0] for row in observations])
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    def relative(path):
        return str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path.resolve())
    result = dict(checkpoint=relative(args.checkpoint), onnx=relative(args.onnx),
                  checkpoint_sha256=sha256(args.checkpoint), onnx_sha256=sha256(args.onnx),
                  graph_matches_trainable_checkpoint=True, fixed_feature_or_feedback_parameters=0,
                  trainable_parameters=stats['params'], arena_validator=stats,
                  inference_contract='Original six observations → learned Tanh MLP → three action scores; no branches or preprocessing.',
                  observations=len(observations), source_reachable_observations=source_count, probe_seeds=seeds,
                  probe_purpose='Development-only numerical verification, not final evaluation',
                  action_agreement=1., max_abs_score_error=float(np.max(np.abs(actual-expected))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print('PASS: standalone graph matches the plain trainable MLP and native ONNX actions match Torch.')


if __name__ == '__main__':
    main()
