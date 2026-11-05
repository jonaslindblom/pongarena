"""Disclosed fixed physics features inside a supported ONNX network.

The unchanged six observations enter a frozen ReLU network that computes the
next-contact landing and flight time, then a learned head. No external predictor,
new input, hidden controller state, or unsupported ONNX operator is required.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import torch
from torch import nn


def linear(weight, bias):
    weight, bias = np.asarray(weight, np.float32), np.asarray(bias, np.float32)
    layer = nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight))
        layer.bias.copy_(torch.from_numpy(bias))
    layer.requires_grad_(False)
    return layer


def selectors(feature, values):
    """A plateau at each integer velocity absorbs float32 observation error."""
    first = np.zeros((4 * len(values), 6), np.float32)
    first[:, feature] = 6
    bias = np.array([[.75 - v, .25 - v, -.25 - v, -.75 - v] for v in values]).reshape(-1)
    second = np.zeros((len(values), 4 * len(values)), np.float32)
    for i in range(len(values)):
        second[i, 4*i:4*i+4] = [2, -2, -2, 2]
    return nn.Sequential(linear(first, bias), nn.ReLU(), linear(second, np.zeros(len(values))))


class TrajectoryFeatures(nn.Module):
    """Original six inputs + normalized landing Y + time /66, shape [n,8].

    Correct on active integer states of frozen Pong v3. Tiny floating-point error
    in landing coordinates is expected; plateau selectors avoid integer-tick
    ambiguity. Columns6/7 refer to the next contacted paddle, even while outgoing.
    """
    def __init__(self):
        super().__init__()
        velocities = list(range(-13, -4)) + list(range(5, 14))
        self.vx_select = selectors(2, velocities)
        self.times = nn.ModuleList()
        self.time_gates = nn.ModuleList()
        for i, velocity in enumerate(velocities):
            # ceil(d/s) = 1 + sum_{k=1..65} I(d > k*s), clamped to at least1.
            # The two ReLU hinges put integer d safely on a 0/1 plateau.
            weight = np.zeros((130, 6), np.float32)
            weight[:, 0] = 360 if velocity < 0 else -360
            bias = np.array([[330 - 2*k*abs(velocity) - .5,
                              330 - 2*k*abs(velocity) - 1.5] for k in range(1, 66)]).reshape(-1)
            self.times.append(nn.Sequential(linear(weight, bias), nn.ReLU(),
                                             linear(np.tile([1, -1], 65)[None], [1])))
            gate = np.zeros((1, 18), np.float32)
            gate[0, i] = 66
            self.time_gates.append(linear(gate, [-66]))
        self.vy_select = selectors(3, list(range(-5, 6)))
        self.travel = linear(np.arange(-5, 6)[:, None], np.zeros(11))
        y_weight = np.zeros((11, 6), np.float32)
        y_weight[:, 1] = 80
        self.origin = linear(y_weight, np.full(11, 410))
        self.landing_gate = linear(np.eye(11) * 820, np.full(11, -820))
        self.sum_landing = linear(np.ones((1, 11)), [-330])
        # Triangle-wave wall reflection over raw Y in [-326,486].
        self.fold = nn.Sequential(linear(np.ones((7, 1)), [400, 300, 148, -4, -156, -308, -460]),
                                  nn.ReLU(), linear([[-1, 2, -2, 2, -2, 2, -2]], [104]))
        raw = np.zeros((8, 6), np.float32)
        raw[:6] = np.eye(6)
        self.raw_features = linear(raw, np.zeros(8))
        landing = np.zeros((8, 1), np.float32)
        landing[6, 0] = 1 / 80
        bias = np.zeros(8)
        bias[6] = -1
        self.landing_feature = linear(landing, bias)
        flight = np.zeros((8, 1), np.float32)
        flight[7, 0] = 1 / 66
        self.time_feature = linear(flight, np.zeros(8))

    def forward(self, obs):
        vx = self.vx_select(obs)
        candidates = [torch.relu(time(obs) + gate(vx)) for time, gate in zip(self.times, self.time_gates)]
        flight = candidates[0]
        for candidate in candidates[1:]:
            flight = flight + candidate
        unfolded = self.sum_landing(torch.relu(self.travel(flight) + self.origin(obs) + self.landing_gate(self.vy_select(obs))))
        landing = self.fold(unfolded)
        return self.raw_features(obs) + self.landing_feature(landing) + self.time_feature(flight)

    def append_onnx(self, nodes, weights, input_name='observation', prefix='trajectory'):
        def layer(module, source, name):
            if isinstance(module, nn.Sequential):
                for i, child in enumerate(module):
                    source = layer(child, source, f'{name}_{i}')
                return source
            if isinstance(module, nn.Linear):
                weights.extend([numpy_helper.from_array(module.weight.detach().cpu().numpy(), name + '_w'),
                                numpy_helper.from_array(module.bias.detach().cpu().numpy(), name + '_b')])
                nodes.append(helper.make_node('Gemm', [source, name + '_w', name + '_b'], [name], transB=1))
            else:
                assert isinstance(module, nn.ReLU)
                nodes.append(helper.make_node('Relu', [source], [name]))
            return name

        def add(a, b, name):
            nodes.append(helper.make_node('Add', [a, b], [name]))
            return name

        vx = layer(self.vx_select, input_name, prefix + '_vx')
        candidates = []
        for i, (time, gate) in enumerate(zip(self.times, self.time_gates)):
            a = layer(time, input_name, f'{prefix}_t{i}')
            b = layer(gate, vx, f'{prefix}_g{i}')
            c = add(a, b, f'{prefix}_tg{i}')
            out = f'{prefix}_selected{i}'
            nodes.append(helper.make_node('Relu', [c], [out]))
            candidates.append(out)
        flight = candidates[0]
        for i, candidate in enumerate(candidates[1:], 1):
            flight = add(flight, candidate, f'{prefix}_time_sum{i}')
        vy = layer(self.vy_select, input_name, prefix + '_vy')
        travel = layer(self.travel, flight, prefix + '_travel')
        origin = layer(self.origin, input_name, prefix + '_origin')
        gate = layer(self.landing_gate, vy, prefix + '_landing_gate')
        raw = add(add(travel, origin, prefix + '_raw0'), gate, prefix + '_raw1')
        positive = prefix + '_positive_landing'
        nodes.append(helper.make_node('Relu', [raw], [positive]))
        unfolded = layer(self.sum_landing, positive, prefix + '_unfolded')
        landing = layer(self.fold, unfolded, prefix + '_fold')
        raw_features = layer(self.raw_features, input_name, prefix + '_raw_features')
        landing_feature = layer(self.landing_feature, landing, prefix + '_landing_feature')
        time_feature = layer(self.time_feature, flight, prefix + '_time_feature')
        return add(add(raw_features, landing_feature, prefix + '_features0'), time_feature, prefix + '_features')


def export_with_features(features, head, path, target_feedback=False, deadband=1.001/80):
    """Export a learned8-input MLP; optionally convert its scalar target to actions."""
    nodes, weights = [], []
    current = features.append_onnx(nodes, weights)
    widths = [8]
    for i, module in enumerate(head):
        output = f'head_{i}'
        if isinstance(module, nn.Linear):
            weights.extend([numpy_helper.from_array(module.weight.detach().cpu().numpy(), output + '_w'),
                            numpy_helper.from_array(module.bias.detach().cpu().numpy(), output + '_b')])
            nodes.append(helper.make_node('Gemm', [current, output + '_w', output + '_b'], [output], transB=1))
            widths.append(module.out_features)
        elif isinstance(module, (nn.Tanh, nn.ReLU)):
            nodes.append(helper.make_node('Tanh' if isinstance(module, nn.Tanh) else 'Relu', [current], [output]))
        else:
            raise TypeError(f'Unsupported learned head layer: {module}')
        current = output
    assert widths[-1] == (1 if target_feedback else 3)
    if target_feedback:
        feedback = np.zeros((6, 3), np.float32)
        feedback[4, 0], feedback[4, 2] = 1, -1
        for name, value in [('feedback', feedback), ('target_scores', np.array([[-1, 0, 1]], np.float32)),
                            ('deadband', np.array([0, deadband, 0], np.float32))]:
            weights.append(numpy_helper.from_array(value, name))
        nodes.extend([helper.make_node('MatMul', [current, 'target_scores'], ['target_actions']),
                      helper.make_node('MatMul', ['observation', 'feedback'], ['feedback_actions']),
                      helper.make_node('Add', ['target_actions', 'feedback_actions'], ['error_actions']),
                      helper.make_node('Add', ['error_actions', 'deadband'], ['action_scores'])])
    else:
        nodes[-1].output[0] = 'action_scores'
    graph = helper.make_graph(nodes, 'Fixed trajectory features plus learned policy',
                              [helper.make_tensor_value_info('observation', 1, [1, 6])],
                              [helper.make_tensor_value_info('action_scores', 1, [1, 3])], weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    helper.set_model_props(model, {'rules': 'pong-v3',
        'architecture': 'fixed ReLU trajectory features6-8; learned head ' + '-'.join(map(str, widths)) + ('; own-paddle feedback' if target_feedback else ''),
        'training': 'Disclosed fixed physics prior; learned PyTorch head; unchanged six-input contract'})
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return model


def check(out):
    from env import Arena, scripted, reflect
    torch.set_num_threads(1)
    features = TrajectoryFeatures().eval()
    observations, truth = [], []
    env = Arena(128, seed=4519)
    for _ in range(256):
        for side in (0, 1):
            observations.append(env.obs(side).copy())
            x, y, vx, vy = env.ball.T
            distance = np.where(vx < 0, x - 15, 345 - x)
            flight = np.maximum(1, np.ceil(distance / np.abs(vx)))
            truth.append(np.stack([(reflect(y + flight * vy) - 80) / 80, flight / 66], 1))
        terminal, _ = env.step(np.stack([scripted(env.obs(), True), scripted(env.obs(1))], 1))
        env.reset(np.flatnonzero(terminal >= 0))
    # Exhaust all active court X, integer VX/VY, with wall/center Y probes.
    for vx in list(range(-13, -4)) + list(range(5, 14)):
        for vy in range(-5, 6):
            for y in [4, 5, 80, 155, 156]:
                x = np.arange(15, 346)
                distance = x - 15 if vx < 0 else 345 - x
                flight = np.maximum(1, np.ceil(distance / abs(vx)))
                obs = np.column_stack([(x - 180) / 180, np.full(len(x), (y - 80) / 80),
                                        np.full(len(x), vx / 6), np.full(len(x), vy / 6),
                                        np.zeros(len(x)), np.zeros(len(x))]).astype(np.float32)
                observations.append(obs)
                truth.append(np.column_stack([(reflect(y + flight * vy) - 80) / 80, flight / 66]))
    obs, expected = np.concatenate(observations), np.concatenate(truth)
    with torch.no_grad():
        actual = torch.cat([features(x) for x in torch.from_numpy(obs).split(4096)]).numpy()
    np.testing.assert_array_equal(actual[:, :6], obs)
    np.testing.assert_allclose(actual[:, 7] * 66, expected[:, 1] * 66, atol=.001, rtol=0)
    np.testing.assert_allclose(actual[:, 6] * 80 + 80, expected[:, 0] * 80 + 80, atol=.002, rtol=0)
    out.mkdir(parents=True, exist_ok=True)
    # Expose feature values through a three-score head for runtime and validator checks.
    head = nn.Sequential(nn.Linear(8, 3))
    with torch.no_grad():
        head[0].weight.zero_()
        head[0].bias.zero_()
        head[0].weight[0, 6] = 1
        head[0].weight[1, 7] = 1
    path = out / 'feature-probe.onnx'
    model = export_with_features(features, head, path)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
    indices = np.random.default_rng(4717).choice(len(obs), 8192, replace=False)
    native = np.array([session.run(None, {'observation': row[None]})[0][0] for row in obs[indices]])
    np.testing.assert_allclose(native[:, :2], expected[indices], atol=.002 / 80, rtol=0)
    large_head = nn.Sequential(nn.Linear(8, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(),
                               nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
    full = export_with_features(features, large_head, out / 'target-probe.onnx', target_feedback=True)
    # Every matrix initializer in this exporter is used exactly once by a [1,n]
    # product. The independent TypeScript importer remains the authority on limits.
    macs = sum(int(np.prod(w.dims)) for w in full.graph.initializer if len(w.dims) == 2)
    assert macs <= 65536 and len(full.graph.node) <= 256
    assert (out / 'target-probe.onnx').stat().st_size <= 256 * 1024
    result = dict(observations=len(obs), reachable_observations=128*256*2,
                  max_time_error_ticks=float(np.max(np.abs(actual[:, 7] - expected[:, 1])) * 66),
                  max_landing_error_units=float(np.max(np.abs(actual[:, 6] - expected[:, 0])) * 80),
                  native_observations=len(indices), native_max_feature_error=float(np.max(np.abs(native[:, :2] - expected[indices]))),
                  target_probe_bytes=(out / 'target-probe.onnx').stat().st_size,
                  target_probe_nodes=len(full.graph.node), target_probe_macs=macs,
                  fixed_features='frozen disclosed next-contact geometry')
    (out / 'check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('training/runs/trajectory-probe'))
    check(p.parse_args().out)
