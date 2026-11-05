"""Construct and evaluate disclosed physics weights for Classic v4, without fitting.

The target table minimizes one-return recovery risk, breaking ties toward center
contact. It does not check whether that contact center is currently reachable.
Use a fresh output directory; the candidate and test seeds freeze before matches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
import torch
from torch import nn

from env import reflect
from pace_study import CONFIGS, decode, make_policy, run_pair, tables
from policy import OnnxPolicy, ROOT
from trajectory_features import TrajectoryFeatures, export_with_features, linear, selectors


RULES = CONFIGS['pace13-v9-scaled']
OUT = ROOT / 'training/results/handcrafted-v4'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(',', ':')) + '\n')


def trajectory():
    features = TrajectoryFeatures()
    vertical = np.arange(-9, 10)
    features.vy_select = selectors(3, list(vertical))
    features.travel = linear(vertical[:, None], np.zeros(19))
    weight = np.zeros((19, 6), np.float32)
    weight[:, 1] = 80
    # y + ticks*vy is in [-590,750]. The offset keeps the selected channel
    # positive; the gate suppresses every other channel throughout that range.
    features.origin = linear(weight, np.full(19, 680))
    features.landing_gate = linear(np.eye(19)*1400, np.full(19, -1400))
    features.sum_landing = linear(np.ones((1, 19)), [-600])
    features.fold = nn.Sequential(
        linear(np.ones((9, 1)), [604, 452, 300, 148, -4, -156, -308, -460, -612]),
        nn.ReLU(), linear([[1, -2, 2, -2, 2, -2, 2, -2, 2]], [4]))
    return features.eval()


def target_tables():
    tab = tables(RULES)
    centers = tab['centers']
    quality = -tab['risk'] - .0001*np.abs(centers - np.arange(157)[:, None])[None]
    best = np.argmax(np.where(tab['legal'][None], quality, -1e12), axis=2)
    incoming = np.take_along_axis(np.broadcast_to(centers, tab['risk'].shape),
                                  best[:, :, None], axis=2)[:, :, 0]
    return incoming, tab['mid']


def target_head():
    incoming, outgoing = target_tables()
    velocities = list(range(-13, -4)) + list(range(5, 14))
    values = np.array([incoming[abs(v), 4:] if v < 0 else outgoing[v, 4:]
                       for v in velocities], np.float32)
    values = (values - 80)/80
    # Each pair of hinges changes from height k to k+1 between k+.25 and k+.75.
    # Plateaus tolerate numeric error while reproducing the exact integer table.
    knots = (np.arange(4, 156)[:, None] + [.25, .75]).reshape(-1)
    count = len(knots)
    weight = np.zeros((count + 4*len(velocities), 8), np.float32)
    weight[:count, 6] = 80
    weight[count:, 2] = 6
    bias = np.concatenate([80-knots,
        np.array([[.75-v, .25-v, -.25-v, -.75-v] for v in velocities]).reshape(-1)])
    gated = np.zeros((18, len(bias)), np.float32)
    differences = np.diff(values, axis=1)
    gated[:, :count:2] = 2*differences
    gated[:, 1:count:2] = -2*differences
    # A velocity selector is one on its integer plateau and zero elsewhere.
    # relu(target+1 + 2*(selector-1)) selects exactly one positive target.
    for i in range(18):
        gated[i, count+4*i:count+4*i+4] = [4, -4, -4, 4]
    return nn.Sequential(linear(weight, bias), nn.ReLU(),
        linear(gated, values[:, 0]-1), nn.ReLU(), linear(np.ones((1, 18)), [-1])).eval()


def expected_scores(obs):
    incoming, outgoing = target_tables()
    vx, own, _, _, landing = decode(obs)
    stage = np.minimum(13, np.abs(vx))
    target = np.where(vx < 0, incoming[stage, landing], outgoing[stage, landing])
    error = (target-own)/80
    return np.stack([-error, np.full(len(obs), 1.001/80), error], axis=1)


def probes():
    rng = np.random.default_rng(2026110504)
    n = 8192
    x, y = rng.integers(15, 346, n), rng.integers(4, 157, n)
    vx = rng.integers(5, 14, n)*rng.choice([-1, 1], n)
    vy = rng.integers(-9, 10, n)
    own, other = rng.integers(15, 146, (2, n))
    obs = np.stack([(x-180)/180, (y-80)/80, vx/6, vy/6,
                    (own-80)/80, (other-80)/80], axis=1).astype(np.float32)
    ticks = np.maximum(1, np.ceil(np.where(vx < 0, x-15, 345-x)/np.abs(vx)))
    return obs, ticks, reflect(y+ticks*vy)


def verify(path, features, head):
    onnx.checker.check_model(onnx.load(path))
    obs, ticks, landing = probes()
    with torch.no_grad():
        actual = torch.cat([features(row) for row in torch.from_numpy(obs).split(1024)]).numpy()
        targets = head(torch.from_numpy(actual)).numpy()[:, 0]
    np.testing.assert_allclose(actual[:, 7]*66, ticks, atol=.002, rtol=0)
    np.testing.assert_allclose(actual[:, 6]*80+80, landing, atol=.005, rtol=0)
    expected = expected_scores(obs)
    np.testing.assert_allclose(targets-obs[:, 4], expected[:, 2], atol=.0005, rtol=0)
    session = OnnxPolicy(path).session
    native = np.array([session.run(None, {'observation': row[None]})[0][0] for row in obs])
    assert np.isfinite(native).all()
    np.testing.assert_allclose(native, expected, atol=.0005, rtol=0)
    np.testing.assert_array_equal(native.argmax(1), expected.argmax(1))
    # Exhaust all integer table entries independently of the flight network.
    direct = np.zeros((18*153, 8), np.float32)
    direct[:, 2] = np.repeat(np.r_[np.arange(-13, -4), np.arange(5, 14)]/6, 153)
    direct[:, 6] = np.tile((np.arange(4, 157)-80)/80, 18)
    incoming, outgoing = target_tables()
    table_truth = np.r_[incoming[np.arange(13, 4, -1), 4:].ravel(),
                        outgoing[np.arange(5, 14), 4:].ravel()]
    with torch.no_grad():
        table_actual = head(torch.from_numpy(direct)).numpy()[:, 0]*80+80
    np.testing.assert_allclose(table_actual, table_truth, atol=.01, rtol=0)
    raw = np.arange(-590, 751, dtype=np.float32)
    with torch.no_grad():
        folded = features.fold(torch.from_numpy(raw[:, None])).numpy()[:, 0]
    np.testing.assert_array_equal(folded, reflect(raw))
    code = """import {readFileSync} from 'node:fs';
import {validateArenaModel} from './src/validate.ts';
const bytes=readFileSync(process.argv[1]);const s=validateArenaModel(bytes);
console.log(JSON.stringify({bytes:bytes.length,params:s.params,macs:s.macs,nodes:s.nodes}));"""
    validation = json.loads(subprocess.run(['node', '--import', 'tsx', '--input-type=module',
        '-e', code, str(path)], cwd=ROOT, check=True, capture_output=True, text=True).stdout)
    return dict(numeric_probes=len(obs), table_entries=len(direct), fold_integers=len(raw),
        action_mismatches=0, max_score_error=float(np.max(np.abs(native-expected))),
        max_landing_error_units=float(np.max(np.abs(actual[:, 6]*80+80-landing))),
        arena=validation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory; retain frozen results.')
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    features, head = trajectory(), target_head()
    path = args.out/'handcrafted-v4.onnx'
    export_with_features(features, head, path, target_feedback=True)
    model = onnx.load(path)
    onnx.helper.set_model_props(model, dict(rules='pong-v4',
        method='Analytically constructed fixed physics and target-table weights; no training',
        limitation='One-return recovery heuristic; contact target may be unreachable',
        architecture='Fixed ReLU trajectory + plateau target table + own-paddle feedback'))
    onnx.save(model, path)
    verification = verify(path, features, head)
    write_json(args.out/'verification.json', verification)
    # These seeds are separate from the 24-seed development set and all archived
    # result JSON seed lists. Freeze them together with model/source hashes.
    rng = np.random.default_rng(2026110503)
    seeds = (rng.choice(100_000_000, 192, replace=False)+3_800_000_000).tolist()
    blocks = [seeds[i:i+64] for i in range(0, 192, 64)]
    protocol = dict(model_sha256=digest(path), rules='pong-v4', blocks=blocks,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
            (Path(__file__), ROOT/'training/scripts/pace_study.py',
             ROOT/'training/scripts/trajectory_features.py', ROOT/'training/v4/reference.py')},
        method='Fixed analytical weights; no optimizer, gradient updates, distillation or mutation search',
        development='24 seeds, RNG2026110502, sampled without replacement from[3700000000,3800000000)',
        primary='Beat Pacer in all three blocks: positive whole-seed margin CI, wins exceed losses on both sides, majority paired contests',
        secondary='First block against remaining active presets and adaptive scripts; report every pairing',
        inference='Actual exported batch-one ONNX files, native CPU runtime; fixed deterministic first argmax',
        verification=verification)
    write_json(args.out/'freeze.json', protocol)
    print('Frozen', digest(path), verification, flush=True)
    candidate = OnnxPolicy(path)
    result = dict(complete=False, model_sha256=digest(path), head_to_head=[], field={})
    pacer = OnnxPolicy(ROOT/'public/models/pacer-v4.onnx')
    for index, block in enumerate(blocks):
        row = run_pair(RULES, candidate, pacer, block)
        result['head_to_head'].append(row)
        write_json(args.out/'evaluation.json', result)
        s = row['summary']
        print('block', index+1, 'pacer', s['wins'], s['losses'], s['draws'], s['point_margin_ci95'], flush=True)
    roster = ['apprentice-v4', 'contender-v3', 'strategist-v3', 'selfplay-v3',
              'apprentice-v3', 'selfplay-v2', 'challenger', 'tracker', 'anticipator']
    for name in roster + ['attack', 'defend']:
        opponent = (make_policy(RULES, name) if name in ('attack', 'defend')
                    else OnnxPolicy(ROOT/f'public/models/{name}.onnx'))
        result['field'][name] = run_pair(RULES, candidate, opponent, blocks[0])
        if name in ('attack', 'defend'):
            result['field']['pacer:'+name] = run_pair(RULES, pacer, opponent, blocks[0])
        write_json(args.out/'evaluation.json', result)
        s = result['field'][name]['summary']
        print(name, s['wins'], s['losses'], s['draws'], flush=True)
    assert digest(path) == protocol['model_sha256']
    for name, expected in protocol['source_sha256'].items():
        assert digest(ROOT/name) == expected, name
    result['primary_passed'] = all(r['summary']['acceptance']['passed'] for r in result['head_to_head'])
    result['complete'] = True
    write_json(args.out/'evaluation.json', result)
    print('Completed; primary passed:', result['primary_passed'], flush=True)


if __name__ == '__main__':
    main()
