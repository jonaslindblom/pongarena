"""Verify frozen experimental exports on every captured learned-policy input.

This replays no training and selects no checkpoint. Inputs are the already
recorded final-confirmation matches; terminal display frames have no action.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
import onnxruntime as ort
import torch

from competition_eval import load_mlp
from policy import ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observation(frame, side):
    x, y, vx, vy = frame['ball']
    paddles = frame['paddles']
    mirror = 1 if side == 0 else -1
    return [mirror*(x-180)/180, (y-80)/80, mirror*vx/6, vy/6,
            (paddles[side]-80)/80, (paddles[1-side]-80)/80]


def validate_in_arena(path):
    code = '''import { readFileSync } from 'node:fs';
import { validateModel } from './src/validate.ts';
const bytes = readFileSync(process.argv[1]); const s = validateModel(bytes);
console.log(JSON.stringify({bytes:bytes.length,params:s.params,macs:s.macs,nodes:s.nodes,widths:s.architecture.widths}));'''
    checked = subprocess.run(['node', '--import', 'tsx', '--input-type=module', '-e', code, str(path.resolve())],
                             cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(checked.stdout)


def main():
    study = ROOT/'training/results/offense-study'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=study/'evaluation/confirmation.json')
    parser.add_argument('--output', type=Path, default=study/'training/native-replay-verification.json')
    args = parser.parse_args()
    manifest_path = study/'manifest.json'
    manifest = json.loads(manifest_path.read_text())

    def check_frozen():
        for path, expected in manifest['files'].items():
            assert digest(ROOT/path) == expected, f'Frozen input changed: {path}'

    check_frozen()
    confirmation = json.loads(args.input.read_text())
    assert confirmation['complete'] is True
    assert confirmation['manifest_sha256'] == digest(manifest_path)
    torch.set_num_threads(1)
    result = dict(purpose='Numerical verification of frozen exports, not another policy-strength evaluation',
                  confirmation_sha256=digest(args.input), manifest_sha256=digest(manifest_path),
                  source_sha256=digest(Path(__file__)), arena_validator_sha256=digest(ROOT/'src/validate.ts'),
                  input_contract='The six v3 observation normalizations, including unbounded velocity/6 values',
                  scope='Every actual trained-policy input in captured trained:attack replays; first final seed, five rallies per rule',
                  configs={})
    for name, matchups in confirmation['configs'].items():
        directory = study/'training/search'/name
        checkpoint, model_path = directory/'best.pt', directory/'best.onnx'
        actor = load_mlp(checkpoint)
        graph = onnx.load(model_path)
        onnx.checker.check_model(graph)
        assert {p.key: p.value for p in graph.metadata_props}['rules'] == f'offense-study-{name}'
        arena = validate_in_arena(model_path)
        assert arena['params'] == sum(p.numel() for p in actor.parameters())
        assert arena['widths'] == [6, 64, 64, 3]
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
        replay_results, maximum_error, total = [], 0., 0
        total_native_mismatch, total_recorded_mismatch = 0, 0
        replays = matchups['trained:attack']['replays']
        assert len(replays) == 5
        for replay in replays:
            frames = [frame for frame in replay['frames'] if frame['actions'] is not None]
            side = replay['side']
            observations = np.asarray([observation(frame, side) for frame in frames], dtype=np.float32)
            recorded = np.asarray([frame['actions'][side] for frame in frames])
            with torch.no_grad():
                expected = actor(torch.from_numpy(observations)).numpy()
            native = np.asarray([session.run(None, {'observation': row[None]})[0][0] for row in observations])
            native_mismatch = int((native.argmax(-1) != expected.argmax(-1)).sum())
            recorded_mismatch = int((native.argmax(-1) != recorded).sum())
            torch_recorded_mismatch = int((expected.argmax(-1) != recorded).sum())
            maximum = float(np.max(np.abs(native-expected)))
            assert np.allclose(native, expected, atol=2e-5, rtol=2e-5), (name, replay['rally'], maximum)
            assert len(frames) == replay['ticks']
            maximum_error = max(maximum_error, maximum)
            total += len(frames)
            total_native_mismatch += native_mismatch
            total_recorded_mismatch += recorded_mismatch + torch_recorded_mismatch
            replay_results.append(dict(seed=replay['seed'], rally=replay['rally'], side=side,
                                       action_frames=len(frames), native_vs_torch_action_mismatches=native_mismatch,
                                       native_vs_recorded_action_mismatches=recorded_mismatch,
                                       torch_vs_recorded_action_mismatches=torch_recorded_mismatch,
                                       maximum_score_error=maximum))
        result['configs'][name] = dict(checkpoint_sha256=digest(checkpoint), onnx_sha256=digest(model_path),
                                      arena_validator=arena, action_frames=total,
                                      native_vs_torch_action_mismatches=total_native_mismatch,
                                      recorded_action_mismatches=total_recorded_mismatch,
                                      maximum_score_error=maximum_error, replays=replay_results)
    check_frozen()
    result['frozen_inputs_unchanged'] = True
    result['action_frames'] = sum(r['action_frames'] for r in result['configs'].values())
    result['passed'] = all(r['native_vs_torch_action_mismatches'] == 0 and r['recorded_action_mismatches'] == 0
                           for r in result['configs'].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert result['passed'], 'Export action mismatch; see saved verification report.'


if __name__ == '__main__':
    main()
