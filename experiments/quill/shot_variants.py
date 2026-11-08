"""Analytical discrete-shot hypotheses, compiled into dense ReLU weights.

No model weights or scores are fitted. A construction-time geometric search
enumerates the game's 19 possible integer outgoing vertical velocities. It
chooses the legal paddle position whose next destination is nearest the corner
away from the opponent. One hypothesis restricts launch direction toward that
corner; the other permits bank shots in either direction.

This model deliberately assumes the maximum horizontal speed (13), hence a
26-tick outgoing flight. It is an attack hypothesis, not an exact solution for
the slower opening returns. All geometry is compiled into ordinary dense
Gemm/Relu layers; no table lookup or search occurs during gameplay.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('quill_precision_for_shots', HERE / 'precision_variants.py')
precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision)
sys.modules.setdefault('precision_variants', precision)
recovery_spec = importlib.util.spec_from_file_location('quill_recovery_for_shots', HERE / 'recovery_variants.py')
recovery_module = importlib.util.module_from_spec(recovery_spec)
recovery_spec.loader.exec_module(recovery_module)


def reflect(height):
    z = (height - 4) % 304
    return 4 + np.where(z <= 152, z, 304 - z)


def shot_target(landing, corner, *, allow_banks):
    """Choose a legal center from exact max-speed integer contact outcomes."""
    velocities = np.arange(-9, 10)
    centers = landing[:, None] - velocities[None] / .6
    legal = (centers >= 15) & (centers <= 145)
    # With a ball beyond a paddle-center bound, no launch can point toward
    # one of the corners. Fall back to all legal launch directions there.
    if not allow_banks:
        direct = legal & ((velocities <= 0) if corner == 4 else (velocities >= 0))
        legal = np.where(direct.any(axis=1)[:, None], direct, legal)
    destination = reflect(landing[:, None] + 26 * velocities[None])
    # Geometric corner distance has strict priority; tiny offset and signed
    # velocity terms only choose deterministic ties, not experimental scores.
    objective = np.abs(destination - corner) + np.abs(velocities)[None] * 1e-7
    choice = np.where(legal, objective, np.inf).argmin(axis=1)
    if not legal[np.arange(len(landing)), choice].all():
        raise ValueError('The chosen landing height admits no legal contact')
    return centers[np.arange(len(landing)), choice]


def weights(*, allow_banks, use_recovery=False):
    recovery_layers = recovery_module.weights(step=.25) if use_recovery else None
    layers = (recovery_layers[:3] if use_recovery else
              precision.weights(flight_bias=0, exact_ticks=True)[:3])
    # One physical-pixel grid across every reachable unfolded height. Folding
    # happens while constructing the target function, before writing weights.
    knots = np.arange(-720, 721, dtype=np.float64) / 80
    landing = reflect(knots * 80 + 80)
    targets = [(shot_target(landing, corner, allow_banks=allow_banks) - 80) / 80
               for corner in (4, 156)]
    # Each corner bank has a gated constant plus standard 1D ReLU hinges.
    # H3 carries r+16, own+1, opponent-high, opponent-low, incoming.
    # On actual even-integer paddle positions, .5*(high-low) is binary.
    hinges = []
    for values in targets:
        slopes = np.diff(values) / np.diff(knots)
        coefficients = np.r_[slopes[0], np.diff(slopes)]
        changes = np.flatnonzero(np.abs(coefficients) > 1e-7)
        hinges.append((changes, coefficients[changes]))
    recovery_hinges = len(recovery_layers[3][1]) - 11 if use_recovery else 0
    width = 1 + sum(1 + len(changes) for changes, _ in hinges) + recovery_hinges
    w4, b4 = np.zeros((width, 6 if use_recovery else 5), np.float32), np.zeros(width, np.float32)
    direction = np.zeros(width, np.float32)
    if use_recovery:
        # Preserve the outgoing-only minimax recovery interpolation and use
        # a positive carry offset for its linear part minus our paddle.
        original_constant = 32 - 16 * (83 / 78) + 1
        recovery_constant = recovery_layers[3][1][0] - original_constant
        w4[0, 1], w4[0, 5] = -1, recovery_layers[3][0][0, 5]
        b4[0], direction[0] = 33 + recovery_constant, 1
    else:
        w4[0, 1], direction[0] = 1, -1  # own+1
    start = 1
    for bank, values in enumerate(targets):
        gate = np.array([0, 0, .5, -.5, 0], np.float32)
        if bank == 1:
            gate = np.array([0, 0, -.5, .5, 1], np.float32)
        w4[start, :5] = gate
        direction[start] = values[0]
        changes, coefficients = hinges[bank]
        for index, (k, coefficient) in enumerate(zip(changes, coefficients)):
            i = start + 1 + index
            w4[i, :5] = 64 * gate
            w4[i, 0] = 1
            b4[i] = -16 - knots[k] - 64
            direction[i] = coefficient
        start += 1 + len(changes)
    if use_recovery:
        w4[start:], b4[start:] = recovery_layers[3][0][11:], recovery_layers[3][1][11:]
        direction[start:] = recovery_layers[4][0][2, 11:]
    w5 = np.stack([-direction, np.zeros(width, np.float32), direction])
    b5 = np.array([32, 1.1 / 80, -32] if use_recovery else [-1, 1.1 / 80, 1], np.float32)
    return layers + [(w4, b4), (w5, b5)]


def variants():
    return {
        'shot-direct': {'allow_banks': False},
        'shot-banked': {'allow_banks': True},
        'shot-direct-recovery': {'allow_banks': False, 'use_recovery': True},
        'shot-banked-recovery': {'allow_banks': True, 'use_recovery': True},
    }


def export(path, **settings):
    layers = weights(**settings)
    nodes, tensors, previous = [], [], 'observation'
    for i, (weight, bias) in enumerate(layers):
        final = i == len(layers) - 1
        output = 'action_scores' if final else f'linear_{i}'
        tensors.extend([numpy_helper.from_array(weight, f'w{i}'),
                        numpy_helper.from_array(bias, f'b{i}')])
        nodes.append(helper.make_node('Gemm', [previous, f'w{i}', f'b{i}'],
                                      [output], transB=1))
        previous = output
        if not final:
            previous = f'hidden_{i}'
            nodes.append(helper.make_node('Relu', [output], [previous]))
    graph = helper.make_graph(nodes, 'QuillDiscreteShots',
        [helper.make_tensor_value_info('observation', TensorProto.FLOAT, [1, 6])],
        [helper.make_tensor_value_info('action_scores', TensorProto.FLOAT, [1, 3])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)],
                              ir_version=10, producer_name='PongArena')
    helper.set_model_props(model, {'method':
        'Analytical speed banks and integer-shot geometry; no training or fitting'})
    onnx.checker.check_model(model)
    count = sum(weight.size + bias.size for weight, bias in layers)
    encoded = model.SerializeToString()
    if count > 65536 or len(encoded) > 256 * 1024:
        raise ValueError('The analytical shot network exceeds the large-class limits')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('/tmp/quill-shot-variants'))
    args = parser.parse_args()
    for name, settings in variants().items():
        path = args.directory / f'{name}.onnx'
        count = export(path, **settings)
        print(f'{name}: {count} parameters; {hashlib.sha256(path.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
