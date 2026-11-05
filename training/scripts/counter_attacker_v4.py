"""Construct a compact predictive v4 attacker without training.

Nineteen outgoing vertical velocities define contact-center bands. Incoming
scores compare feasible bands for each first move; outgoing recovery forecasts
the farther reachable endpoint. Ordinary movement parity is retained, while
wall-clamp parity changes and longer-horizon play remain approximations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import helper

from compact_handcrafted_v4 import Graph
from pace_study import CONFIGS, tables
from policy import OnnxPolicy, ROOT


class CounterGraph(Graph):
    """Share repeated expressions and track shapes for scalar/column broadcasting."""
    def __init__(self):
        super().__init__()
        self.shapes = {'observation': (1, 6)}
        self.expressions = {}

    def constant(self, value):
        name = super().constant(value)
        self.shapes[name] = np.asarray(value).shape
        return name

    def op(self, kind, *inputs, **attrs):
        key = (kind, inputs, tuple(sorted(attrs.items())))
        if key in self.expressions:
            return self.expressions[key]
        name = super().op(kind, *inputs, **attrs)
        self.expressions[key] = name
        left = self.shapes[inputs[0]]
        if kind == 'Add':
            shape = np.broadcast_shapes(left, self.shapes[inputs[1]])
        elif kind in ('Gemm', 'MatMul'):
            right = self.shapes[inputs[1]]
            if attrs.get('transA'):
                left = left[::-1]
            if attrs.get('transB'):
                right = right[::-1]
            shape = (left[0], right[1])
        else:
            shape = left
        self.shapes[name] = shape
        return name

    def neg(self, source):
        return self.affine(source, np.eye(1), 1, alpha=-1., beta=0.)

    def sub(self, left, right):
        left_shape, right_shape = self.shapes[left], self.shapes[right]
        output = np.broadcast_shapes(left_shape, right_shape)
        if left_shape[0] == output[0]:
            source, bias, alpha, beta = left, right, 1., -1.
        else:
            source, bias, alpha, beta = right, left, -1., 1.
        columns = self.shapes[source][1]
        weight = np.eye(columns) if columns == output[1] else np.ones((1, output[1]))
        return self.op('Gemm', source, self.constant(weight), bias, alpha=alpha, beta=beta)

    def maximum(self, left, right):
        return self.op('Add', right, self.relu(self.sub(left, right)))

    def minimum(self, left, right):
        return self.sub(left, self.relu(self.sub(left, right)))

    def absolute(self, source):
        return self.op('Gemm', self.relu(source), self.constant(np.eye(1)), source,
                       alpha=2., beta=-1.)

    def clip(self, source, lower, upper):
        return self.sub(self.op('Add', lower, self.relu(self.sub(source, lower))),
                        self.relu(self.sub(source, upper)))

    def pool(self, source, rows):
        """Reduce candidate rows by pairwise max, keeping the three action scores."""
        while rows > 1:
            count = (rows+1)//2
            difference = np.zeros((count, rows))
            second = np.zeros_like(difference)
            for i in range(count):
                first, last = 2*i, min(2*i+1, rows-1)
                difference[i, first] += 1
                difference[i, last] -= 1
                second[i, last] = 1
            source = self.op('Add', self.relu(self.product(self.constant(difference), source)),
                             self.product(self.constant(second), source))
            rows = count
        return source

    def fold(self, source):
        hinges = self.relu(self.add(source, [604, 452, 300, 148, -4, -156, -308, -460, -612]))
        return self.affine(hinges, np.array([1, -2, 2, -2, 2, -2, 2, -2, 2])[:, None], 4)


def trajectory(graph):
    vx = graph.project(2, 6)
    speed = graph.op('Add', graph.relu(vx), graph.relu(graph.neg(vx)))
    sign = graph.scalar(graph.sub(graph.relu(graph.add(vx, [1])),
                                 graph.relu(graph.add(vx, [-1]))), 1, -1)
    basis = graph.relu(graph.add(speed, -np.arange(5, 13)))
    slopes = np.diff(1/np.arange(5, 14, dtype=float))
    inverse = graph.affine(basis, np.r_[slopes[0], np.diff(slopes)][:, None], 1/5)
    distance = graph.scalar(graph.product(sign, graph.project(0, 180)), -1, 165)
    fractional_time = graph.product(distance, inverse)
    lower = graph.relu(graph.add(fractional_time, -np.arange(1, 66)-1/64))
    upper = graph.relu(graph.add(fractional_time, -np.arange(1, 66)-1/32))
    ticks = graph.affine(graph.op('Add', lower, graph.product(graph.constant([[-1]]), upper)),
                         np.ones((65, 1)), 1, alpha=64.)
    height = graph.fold(graph.op('Add', graph.project(1, 80, 80),
                                graph.product(ticks, graph.project(3, 6))))
    return height, ticks, speed, sign


def build(path):
    graph = CounterGraph()
    height, ticks, speed, sign = trajectory(graph)
    own = graph.project(4, 80, 80)
    other = graph.project(5, 80, 80)
    tab = tables(CONFIGS['pace13-v9-scaled'])
    recovery = graph.curve(height, np.arange(4, 157), tab['mid'][13, 4:])
    speed_heights = np.arange(5, 14)
    incoming = graph.scalar(sign, -.5, .5)
    outgoing = graph.scalar(sign, .5, .5)
    actor = graph.op('Add', graph.product(incoming, own), graph.product(outgoing, other))
    receiver = graph.op('Add', graph.product(incoming, other), graph.product(outgoing, own))

    # Every integer outgoing vertical velocity maps to a contact-center band.
    scale = graph.curve(speed, speed_heights, 65/(3*np.minimum(13, speed_heights+1)))
    flight = graph.curve(speed, speed_heights, np.ceil(330/np.minimum(13, speed_heights+1)))
    velocity = np.arange(-9, 10)[:, None]
    offset_lower = graph.product(graph.constant(velocity+.5), scale)
    offset_upper = graph.product(graph.constant(velocity-.5), scale)
    lower = graph.maximum(graph.maximum(graph.sub(height, offset_lower), graph.add(height, [-15])),
                          graph.constant([[15]]))
    upper = graph.minimum(graph.minimum(graph.sub(height, offset_upper), graph.add(height, [15])),
                          graph.constant([[145]]))

    # Keep ordinary two-unit movement parity. The saturated Tanh steps select
    # the first center on that lattice; wall-clamp parity changes are omitted.
    half_actor = graph.affine(graph.op('Tanh', graph.add(graph.scalar(actor, 4096),
        -8192*(np.arange(8, 73)-1/512))), np.ones((65, 1)), 39.5, alpha=.5)
    parity = graph.sub(actor, graph.scalar(half_actor, 2))
    scaled_lower = graph.scalar(graph.sub(lower, parity), 4096)
    half_lower = graph.affine(graph.op('Tanh', graph.add(scaled_lower,
        -8192*(np.arange(7, 73)+1/512))), np.ones((66, 1)), 40, alpha=.5)
    lower = graph.op('Add', parity, graph.scalar(half_lower, 2))
    empty_band = graph.sub(lower, upper)
    # A band spans fewer than four units, so its parity lattice contains at
    # most two centers. Retain an empty-band gap before quantizing its upper end.
    second_center = graph.op('Tanh', graph.scalar(graph.sub(upper, lower),
                                                8192, -8192*(2-1/512)))
    upper = graph.op('Add', lower, graph.add(second_center, [1]))
    destination = graph.fold(graph.op('Add', height, graph.product(graph.constant(velocity), flight)))

    # Each column is one first action. Remaining ticks determine whether its
    # next center can still reach each candidate contact band.
    radius = graph.scalar(ticks, 2, -2)
    expected = graph.clip(recovery, graph.sub(receiver, radius), graph.op('Add', receiver, radius))
    next_paddle = graph.clip(graph.add(actor, [-2, 0, 2]),
                            graph.constant([[15, 15, 15]]), graph.constant([[145, 145, 145]]))
    lower_gap = graph.sub(graph.sub(lower, next_paddle), radius)
    upper_gap = graph.sub(graph.sub(next_paddle, upper), radius)
    gap = graph.relu(graph.maximum(graph.maximum(lower_gap, upper_gap), empty_band))
    immediate = graph.maximum(graph.relu(graph.sub(lower, next_paddle)),
                              graph.relu(graph.sub(next_paddle, upper)))
    margin = graph.absolute(graph.sub(destination, expected))
    feasible = graph.affine(gap, np.eye(3), 1, alpha=256., beta=-1.)
    feasible = graph.sub(graph.relu(feasible), graph.relu(graph.add(feasible, [-1])))
    # Integer centers give half-integer geometry/risk and integer travel costs.
    # Scale the teacher's .05/.001 coefficients by2000 before max reduction:
    # all incoming scores are exact integers, below float32's exact-integer limit.
    quality = graph.op('Gemm', feasible, graph.constant(np.eye(3)), margin,
                       alpha=-1024000., beta=2000.)
    quality = graph.op('Gemm', immediate, graph.constant(np.eye(3)), quality, alpha=-2., beta=1.)

    safe_lower = graph.curve(destination, np.arange(4, 157), tab['low'][13, 4:])
    safe_upper = graph.curve(destination, np.arange(4, 157), tab['high'][13, 4:])
    slack = graph.scalar(graph.sub(safe_lower, safe_upper), .5)
    return_reach = graph.scalar(flight, 2)
    risk = graph.relu(graph.maximum(graph.maximum(slack,
        graph.sub(graph.sub(safe_lower, upper), return_reach)),
        graph.sub(graph.sub(lower, return_reach), safe_upper)))
    quality = graph.op('Gemm', quality, graph.constant(np.eye(3)), risk, alpha=1., beta=-100.)
    attack = graph.add(graph.pool(quality, 19), [-.5, 0, -.5])

    # Forecast the reachable landing endpoints while outgoing. Recover toward
    # the endpoint farther from the recovery position expected by the attacker.
    reach = graph.scalar(ticks, 2)
    endpoint_gap = graph.relu(graph.maximum(graph.maximum(graph.sub(graph.sub(lower, actor), reach),
        graph.sub(graph.sub(actor, upper), reach)), empty_band))
    penalty = graph.scalar(endpoint_gap, 256, -1)
    penalty = graph.scalar(graph.sub(graph.relu(penalty), graph.relu(graph.add(penalty, [-1]))), 512)
    top = graph.pool(graph.sub(destination, penalty), 19)
    bottom = graph.neg(graph.pool(graph.sub(graph.neg(destination), penalty), 19))
    expected_own = graph.clip(recovery, graph.sub(receiver, radius), graph.op('Add', receiver, radius))
    difference = graph.sub(graph.op('Add', top, bottom), graph.scalar(expected_own, 2))
    gate = graph.scalar(graph.relu(difference), 64)
    gate = graph.sub(gate, graph.relu(graph.add(gate, [-1])))
    target = graph.op('Add', bottom, graph.product(gate, graph.sub(top, bottom)))
    error = graph.sub(target, own)
    defense = graph.affine(error, [[-1, 0, 1]], [0, 1.1, 0])
    graph.op('Add', graph.product(incoming, attack), graph.product(outgoing, defense))
    graph.nodes[-1].output[0] = 'action_scores'

    model = helper.make_model(helper.make_graph(graph.nodes, 'Compact predictive attacker',
        [helper.make_tensor_value_info('observation', 1, [1, 6])],
        [helper.make_tensor_value_info('action_scores', 1, [1, 3])], graph.weights),
        opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    helper.set_model_props(model, dict(rules='pong-v4',
        method='Handcrafted19-angle attack and predictive recovery; no training',
        limitation='Endpoint forecast and cap13 safety prior; future wall-clamp parity changes omitted'))
    onnx.checker.check_model(model)
    onnx.save(model, path)


def validate(path):
    scores = OnnxPolicy(path).session.run(None, {'observation': np.zeros((1, 6), np.float32)})[0]
    assert scores.shape == (1, 3) and np.isfinite(scores).all()
    code = """import {readFileSync} from 'node:fs';import {validateArenaModel} from './src/validate.ts';
import {smallestDivision} from './src/competition.ts';
const bytes=readFileSync(process.argv[1]);const s=validateArenaModel(bytes);
console.log(JSON.stringify({bytes:bytes.length,params:s.params,macs:s.macs,nodes:s.nodes,division:smallestDivision(s)}));"""
    arena = json.loads(subprocess.run(['node', '--import', 'tsx', '--input-type=module',
        '-e', code, str(path)], cwd=ROOT, check=True, capture_output=True, text=True).stdout)
    assert arena['division'] == 'small', arena
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), arena=arena,
                zero_input_scores=scores[0].tolist(), finite_zero_input=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory; preserve previous evidence.')
    args.out.mkdir(parents=True)
    path = args.out/'counter-attacker-v4.onnx'
    build(path)
    verification = validate(path)
    (args.out/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
