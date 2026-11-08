"""Speed-bank Quill hypotheses: analytic weights, no training or fitting.

Each legal incoming horizontal speed gets a gated square-interpolation bank.
Only the matching bank is active. Its difference of squares computes distance
times vertical velocity, scaled by that speed's reciprocal. A final hinge bank
reflects up to five wall crossings. Everything at inference is dense Gemm/Relu.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def weights(*, step=.125, flight_bias=.5, exact_ticks=False, aim_scale=83 / 78, aim_offset=19 / 312):
    # H1: legal incoming speeds are discrete, 5..13. abs(vx+s/6) is
    # represented by two ReLUs; 64 times this distance disables other banks.
    speeds = range(5, 14)
    w1, b1 = np.zeros((24, 6), np.float32), np.zeros(24, np.float32)
    for j, speed in enumerate(speeds):
        w1[2 * j, 2], b1[2 * j] = 1, speed / 6
        w1[2 * j + 1, 2], b1[2 * j + 1] = -1, -speed / 6
    w1[18, 0], b1[18] = 1, 11 / 12  # d=(x-15)/180 >= 0
    w1[19, 3], b1[19] = 1, 2         # v=vy/6, carried with offset
    w1[20, 1], b1[20] = 1, 16
    w1[21, 4], b1[21] = 1, 1
    w1[22, 5], b1[22] = 1, 1
    w1[23, 2] = 1                   # outgoing detector

    # On [-1.5,3.5], q(z) interpolates z^2 up to a shared constant.
    # q(d+v)-q(d-v) ~= 4*d*v. Include the left endpoint's initial slope.
    knots = np.arange(-1.5, 3.5, step)
    coefficients = np.full(len(knots), 2 * step)
    coefficients[0] = -3 + step
    bank_width = 2 * len(knots)
    carry = 9 * bank_width
    w2, b2 = np.zeros((carry + 6, 24), np.float32), np.zeros(carry + 6, np.float32)
    for j, speed in enumerate(speeds):
        # Every post-return flight starts at a contact plane, separated by
        # 330 units. Its ceil correction is therefore constant for that speed
        # throughout the flight. Serves have speed 5 and 165/5 is also integral.
        correction = (np.ceil(330 / speed) - 330 / speed) * speed / 180 if exact_ticks else 0
        for sign_index, sign in enumerate((1, -1)):
            for k, knot in enumerate(knots):
                i = j * bank_width + sign_index * len(knots) + k
                w2[i, 18], w2[i, 19] = 1, sign
                w2[i, 2 * j:2 * j + 2] = -64
                b2[i] = -2 * sign - knot + correction
    # These incoming-only signals vanish during departure, so the resulting
    # target is the court center, regardless of opponent position.
    w2[carry, 20], w2[carry, 23] = 1, -64
    w2[carry + 1, 19], w2[carry + 1, 23] = 1, -64
    w2[carry + 2, 21] = 1
    for i, offset in ((3, 2), (4, 0)):
        w2[carry + i, 22], w2[carry + i, 23] = 160, -512
        b2[carry + i] = offset - 160
    w2[carry + 5, 23], b2[carry + 5] = -2, 1

    # H3: signed landing forecast carried as r+16, own paddle, two opponent
    # ramps and incoming indicator. Half a tick approximates ceil(distance/s).
    w3, b3 = np.zeros((5, carry + 6), np.float32), np.zeros(5, np.float32)
    for j, speed in enumerate(speeds):
        scale = 6 * 180 / (speed * 80 * 4)
        start = j * bank_width
        w3[0, start:start + len(knots)] = coefficients * scale
        w3[0, start + len(knots):start + bank_width] = -coefficients * scale
    w3[0, carry], w3[0, carry + 1] = 1, 6 * flight_bias / 80
    w3[0, carry + 5], b3[0] = -16 - 12 * flight_bias / 80, 16
    for i, source in enumerate((carry + 2, carry + 3, carry + 4, carry + 5), 1):
        w3[i, source] = 1

    # H4: triangular-wave wall folding, walls at +/- .95, then every 1.9.
    # Linear carrier stays positive over every reachable game observation.
    w4, b4 = np.zeros((11, 5), np.float32), np.zeros(11, np.float32)
    w4[0] = [aim_scale, -1, aim_offset, -aim_offset, -aim_offset]
    b4[0] = 32 - 16 * aim_scale + 1
    direction = np.zeros(11, np.float32)
    direction[0] = 1
    for k in range(5):
        for side, sign in enumerate((1, -1)):
            i = 1 + 2 * k + side
            w4[i, 0] = sign
            b4[i] = -16 * sign - (.95 + 1.9 * k)
            direction[i] = (-2 if side == 0 else 2) * (-1) ** k * aim_scale
    w5 = np.stack([-direction, np.zeros(11, np.float32), direction])
    b5 = np.array([32, 1.1 / 80, -32], np.float32)
    return [(w1, b1), (w2, b2), (w3, b3), (w4, b4), (w5, b5)]


def variants():
    return {
        'precision': {},
        'precision-no-aim': {'aim_scale': 1, 'aim_offset': 0},
        'precision-no-tick': {'flight_bias': 0},
        'precision-exact-ticks': {'flight_bias': 0, 'exact_ticks': True},
    }


def export(path, **settings):
    nodes, tensors, previous = [], [], 'observation'
    layers = weights(**settings)
    for i, (weight, bias) in enumerate(layers):
        final = i == len(layers) - 1
        output = 'action_scores' if final else f'linear_{i}'
        tensors.extend([numpy_helper.from_array(weight, f'w{i}'), numpy_helper.from_array(bias, f'b{i}')])
        nodes.append(helper.make_node('Gemm', [previous, f'w{i}', f'b{i}'], [output], transB=1))
        previous = output
        if not final:
            previous = f'hidden_{i}'
            nodes.append(helper.make_node('Relu', [output], [previous]))
    graph = helper.make_graph(nodes, 'QuillPrecision',
        [helper.make_tensor_value_info('observation', TensorProto.FLOAT, [1, 6])],
        [helper.make_tensor_value_info('action_scores', TensorProto.FLOAT, [1, 3])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10,
                              producer_name='PongArena')
    helper.set_model_props(model, {'method': 'Handwritten speed banks and square interpolation; no training or fitting'})
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())
    return sum(w.size + b.size for w, b in layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('scratch/quill-experiments'))
    args = parser.parse_args()
    for name, settings in variants().items():
        path = args.directory / f'{name}.onnx'
        count = export(path, **settings)
        print(f'{name}: {count} parameters; {hashlib.sha256(path.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
