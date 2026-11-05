"""Build Nano/Small physics policies without training; validate their ONNX budgets.

Scalar MatMul products replace the large trajectory selector networks. The Small
policy retains every target table from handcrafted_v4; Nano keeps incoming cap-13
placement and recovers to center. Sparse ReLU hinges encode the tables exactly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import helper, numpy_helper

from handcrafted_v4 import target_tables
from policy import OnnxPolicy, ROOT


class Graph:
    def __init__(self):
        self.nodes, self.weights, self.constants = [], [], {}

    def constant(self, value):
        value = np.asarray(value, np.float32)
        key = (value.shape, value.tobytes())
        if key not in self.constants:
            name = 'c' + str(len(self.weights))
            self.weights.append(numpy_helper.from_array(value, name))
            self.constants[key] = name
        return self.constants[key]

    def op(self, kind, *inputs, **attrs):
        name = 'h' + str(len(self.nodes))
        self.nodes.append(helper.make_node(kind, list(inputs), [name], **attrs))
        return name

    def add(self, source, value):
        return self.op('Add', source, self.constant(value))

    def affine(self, source, weight, bias=0, **attrs):
        return self.op('Gemm', source, self.constant(weight),
                       self.constant(np.atleast_1d(bias)), **attrs)

    def scalar(self, source, scale=1, bias=0):
        # Alpha/beta are the supported Gemm scaling attributes. Sharing scalar
        # one avoids storing a new tensor for each fixed arithmetic coefficient.
        return self.affine(source, [[1]], 1, alpha=float(scale), beta=float(bias))

    def project(self, column, scale, bias=0):
        weight = np.zeros((6, 1), np.float32)
        weight[column, 0] = scale
        return self.affine('observation', weight, bias)

    def product(self, left, right):
        return self.op('MatMul', left, right)

    def relu(self, source):
        return self.op('Relu', source)

    def sum(self, values):
        result = values[0]
        for value in values[1:]:
            result = self.op('Add', result, value)
        return result

    def curve(self, source, heights, values):
        slopes = np.diff(values)/np.diff(heights)
        coefficients = np.r_[slopes[0], np.diff(slopes)]
        used = np.flatnonzero(coefficients)
        hinges = self.relu(self.add(source, -heights[:-1][used]))
        return self.affine(hinges, coefficients[used, None], values[0])


def landing(graph):
    vx = graph.project(2, 6)
    speed = graph.op('Add', graph.relu(vx), graph.relu(graph.scalar(vx, -1)))
    positive = graph.relu(graph.add(vx, [1]))
    negative = graph.relu(graph.add(vx, [-1]))
    sign = graph.scalar(graph.op('Add', positive, graph.scalar(negative, -1)), 1, -1)
    # Reciprocal is piecewise linear and exact at every legal integer speed.
    basis = graph.relu(graph.add(speed, -np.arange(5, 13)))
    slopes = np.diff(1/np.arange(5, 14, dtype=np.float64))
    inverse = graph.affine(basis, np.r_[slopes[0], np.diff(slopes)][:, None], 1/5)
    signed_x = graph.product(sign, graph.project(0, 180))
    distance = graph.scalar(signed_x, -1, 165)
    fractional_time = graph.product(distance, inverse)
    # For integer distance and speed5..13, the smallest positive fractional part
    # is1/13. Binary ramp widths give exact plateau differences in float32.
    lower = graph.relu(graph.add(fractional_time, -np.arange(1, 66)-1/64))
    upper = graph.relu(graph.add(fractional_time, -np.arange(1, 66)-1/32))
    negative_upper = graph.product(graph.constant([[-1]]), upper)
    differences = graph.op('Add', lower, negative_upper)
    ticks = graph.affine(differences, np.ones((65, 1)), 1, alpha=64.)
    unfolded = graph.op('Add', graph.project(1, 80, 80),
                        graph.product(ticks, graph.project(3, 6)))
    hinges = graph.relu(graph.add(unfolded, [604, 452, 300, 148, -4, -156, -308, -460, -612]))
    height = graph.affine(hinges, np.array([1, -2, 2, -2, 2, -2, 2, -2, 2])[:, None], 4)
    return height, basis, sign


def build(size, path):
    graph = Graph()
    height, speed_basis, sign = landing(graph)
    incoming, outgoing = target_tables()
    heights = np.arange(4, 157)
    if size == 'nano':
        toward = graph.curve(height, heights, incoming[13, 4:])
        away = graph.constant([[80]])
    else:
        toward_parts, away_parts = [], []
        for stage in range(5, 13):
            coefficients = np.zeros(8)
            if stage == 5:
                coefficients[:2] = [-1, 1]
            elif stage == 12:
                coefficients[-2:] = [1, -1]
            else:
                coefficients[stage-6:stage-3] = [1, -2, 1]
            gate = graph.affine(speed_basis, coefficients[:, None], int(stage == 5))
            toward = graph.curve(height, heights, incoming[stage, 4:])
            away = graph.curve(height, heights, outgoing[stage, 4:])
            toward_parts.append(graph.product(gate, toward))
            away_parts.append(graph.product(gate, away))
        toward, away = graph.sum(toward_parts), graph.sum(away_parts)
    incoming_gate = graph.scalar(sign, -.5, .5)
    outgoing_gate = graph.scalar(sign, .5, .5)
    target = graph.op('Add', graph.product(incoming_gate, toward),
                      graph.product(outgoing_gate, away))
    error = graph.op('Add', target, graph.project(4, -80, -80))
    scores = graph.affine(error, [[-1, 0, 1]], [0, 1.1, 0])
    graph.nodes[-1].output[0] = 'action_scores'
    model = helper.make_model(helper.make_graph(graph.nodes, 'Compact handcrafted Pong v4',
        [helper.make_tensor_value_info('observation', 1, [1, 6])],
        [helper.make_tensor_value_info('action_scores', 1, [1, 3])], graph.weights),
        opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    helper.set_model_props(model, dict(rules='pong-v4', division=size,
        method='Analytical scalar arithmetic and sparse ReLU target tables; no training',
        targets='Cap13 incoming, center recovery' if size == 'nano' else 'Full stage-aware handcrafted-v4 targets',
        deadband='1.1 court units; same intended actions as1.001 on integer/half-integer targets'))
    onnx.checker.check_model(model)
    onnx.save(model, path)


def reference(obs, size):
    from pace_study import decode
    incoming, outgoing = target_tables()
    vx, own, _, _, height = decode(obs)
    stage = np.abs(vx)
    target = np.where(vx < 0, incoming[13 if size == 'nano' else stage, height],
                      80 if size == 'nano' else outgoing[stage, height])
    error = target-own
    return np.stack([-error, np.full(len(obs), 1.1), error], axis=1)


def validate(path, size):
    from handcrafted_v4 import probes
    obs, _, _ = probes()
    expected = reference(obs, size)
    session = OnnxPolicy(path).session
    native = np.array([session.run(None, {'observation': row[None]})[0][0] for row in obs])
    assert np.isfinite(native).all()
    np.testing.assert_allclose(native, expected, atol=.05, rtol=0)
    np.testing.assert_array_equal(native.argmax(1), expected.argmax(1))
    code = """import {readFileSync} from 'node:fs';import {validateArenaModel} from './src/validate.ts';
import {smallestDivision} from './src/competition.ts';
const bytes=readFileSync(process.argv[1]);const s=validateArenaModel(bytes);
console.log(JSON.stringify({bytes:bytes.length,params:s.params,macs:s.macs,nodes:s.nodes,division:smallestDivision(s)}));"""
    arena = json.loads(subprocess.run(['node', '--import', 'tsx', '--input-type=module',
        '-e', code, str(path)], cwd=ROOT, check=True, capture_output=True, text=True).stdout)
    assert arena['division'] == size, arena
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), arena=arena,
        numeric_probes=len(obs), action_mismatches=0,
        max_score_error_units=float(np.max(np.abs(native-expected))))


def evaluate(out, verification):
    from pace_study import CONFIGS, make_policy, run_pair
    rules = CONFIGS['pace13-v9-scaled']
    seeds = (np.random.default_rng(2026110510).choice(50_000_000, 192,
             replace=False)+4_050_000_000).tolist()
    blocks = [seeds[i:i+64] for i in range(0, 192, 64)]
    # Check against every retained integer or numeric-string leaf, a conservative
    # superset of archived seed lists. Do this before consuming any final result.
    final = set(seeds)
    overlaps, files = [], 0

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(key, path)
                visit(child, path)
        elif isinstance(value, list):
            for child in value:
                visit(child, path)
        elif isinstance(value, int) and not isinstance(value, bool) and value in final:
            overlaps.append([str(path.relative_to(ROOT)), value])
        elif isinstance(value, str) and value.isdigit() and int(value) in final:
            overlaps.append([str(path.relative_to(ROOT)), value])

    for root in (ROOT/'training/results', ROOT/'public/training'):
        for path in root.rglob('*'):
            if path.suffix not in ('.json', '.jsonl') or path.is_relative_to(out.resolve()):
                continue
            objects = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                       if path.suffix == '.jsonl' else [json.loads(path.read_text())])
            for obj in objects:
                visit(obj, path)
            files += 1
    assert not overlaps, overlaps
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    original = ROOT/'training/results/handcrafted-v4/handcrafted-v4.onnx'
    references = list((ROOT/'public/models').glob('*.onnx')) + [original]
    sources = [Path(__file__), ROOT/'training/scripts/handcrafted_v4.py',
               ROOT/'training/scripts/pace_study.py', ROOT/'training/scripts/policy.py',
               ROOT/'training/v4/reference.py']
    freeze = dict(blocks=blocks, models=verification,
        development='24 seeds from RNG2026110509 in[4000000000,4050000000)',
        method='Analytical graph construction and sparse table encoding; no fitting or optimizer',
        primary='Each size beats Pacer in all3blocks: positive whole-seed margin CI, wins exceed losses on each side, majority paired contests',
        secondary='First block versus nine other active presets, original Open policy, adaptive scripts, and the other compact model',
        inference='Actual static batch-one native CPU ONNX files; v4; deterministic first argmax; exact loops',
        seed_audit=dict(files_scanned=files, final_seeds=len(seeds), overlaps=overlaps),
        references={str(p.relative_to(ROOT)): digest(p) for p in references},
        sources={str(p.relative_to(ROOT)): digest(p) for p in sources})
    (out/'freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
    models = {size: OnnxPolicy(out/f'handcrafted-{size}-v4.onnx') for size in verification}
    pacer = OnnxPolicy(ROOT/'public/models/pacer-v4.onnx')
    result = dict(complete=False, head_to_head={size: [] for size in models}, field={})

    def record():
        (out/'evaluation.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')

    def play(candidate, opponent, block, label):
        row = run_pair(rules, candidate, opponent, block)
        s = row['summary']
        print(label, s['wins'], s['losses'], s['draws'], s['point_margin_ci95'], flush=True)
        return row

    for size, candidate in models.items():
        for index, block in enumerate(blocks):
            result['head_to_head'][size].append(play(candidate, pacer, block, f'{size}:pacer block{index+1}'))
            record()
    roster = ['apprentice-v4', 'contender-v3', 'strategist-v3', 'selfplay-v3',
              'apprentice-v3', 'selfplay-v2', 'challenger', 'tracker', 'anticipator',
              'original-open', 'attack', 'defend']
    for name in roster:
        opponent = (make_policy(rules, name) if name in ('attack', 'defend') else
                    OnnxPolicy(original if name == 'original-open' else ROOT/f'public/models/{name}.onnx'))
        for size, candidate in models.items():
            result['field'][size+':'+name] = play(candidate, opponent, blocks[0], size+':'+name)
            record()
    result['field']['nano:small'] = play(models['nano'], models['small'], blocks[0], 'nano:small')
    for size in models:
        assert digest(out/f'handcrafted-{size}-v4.onnx') == verification[size]['sha256']
    for name, expected in dict(freeze['references'], **freeze['sources']).items():
        assert digest(ROOT/name) == expected, name
    result['primary_passed'] = {size: all(row['summary']['acceptance']['passed'] for row in rows)
                                for size, rows in result['head_to_head'].items()}
    result['complete'] = True
    record()
    print('Completed:', result['primary_passed'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory; preserve previous evidence.')
    args.out.mkdir(parents=True)
    result = {}
    for size in ('nano', 'small'):
        path = args.out/f'handcrafted-{size}-v4.onnx'
        build(size, path)
        result[size] = validate(path, size)
        print(size, result[size], flush=True)
    (args.out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    if args.evaluate:
        evaluate(args.out, result)


if __name__ == '__main__':
    main()
