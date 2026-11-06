"""Handwrite Quill Nano: 163 parameters, no training or parameter search.

The 6→12→4→3→3 ReLU network compresses Vector Nano's physical prior:
approximate distance × vertical velocity with a five-segment square curve,
fold the forecast at a wall, and aim toward the opposite opponent endpoint.
All knowledge is in ordinary dense weights. No teacher queries, optimizer,
checkpoint loading, feature preprocessing, or inference helpers are used.

Limitations: cap-13 flight time, approximate products and contact rounding,
one-wall forecast, and near-center recovery instead of planning future shots.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def weights():
    # d = observation.x + 11/12 is distance to our contact plane / 180.
    # q(t) = -2t + (4/3) Σ ReLU(t-k), k = [-2/3, 0, 2/3, 4/3].
    # q(d+2v/3)-q(d-2v/3) approximates (8/3)*d*v.
    product_scale = 6 * 180 / (13 * 80) / (4 * 2 / 3)
    aim_scale, aim_offset = 83 / 78, 19 / 312
    w1, b1 = np.zeros((12, 6)), np.zeros(12)
    for j, sign in enumerate([1, -1]):
        for i, knot in enumerate(np.array([-1, 0, 1, 2]) * (2 / 3)):
            w1[4 * j + i, 0] = 1
            w1[4 * j + i, 3] = sign * 2 / 3
            b1[4 * j + i] = 11 / 12 - knot
    w1[8, 1] = 1
    w1[8, 3] = 6 * (8 / 13) / 80 - 3 * (2 / 3) * 2 * (2 / 3) * product_scale
    b1[8] = 4  # Carry signed ball height and the square curve's linear tail.
    w1[9, 4], b1[9] = 1, 1
    w1[10, 2] = 1  # Positive only when the ball travels away.
    w1[11, 5], b1[11] = 160, 2

    w2, b2 = np.zeros((4, 12)), np.zeros(4)
    w2[0, :4] = 2 * (2 / 3) * product_scale
    w2[0, 4:8] = -w2[0, :4]
    w2[0, 8], w2[0, 10] = 1, -20  # Forecast + 4, suppressed on departure.
    w2[1, 9], w2[1, 11], b2[1] = 1, -aim_offset, 10
    w2[2, 10], b2[2] = -2, 1  # Exactly one incoming, zero outgoing at legal speeds.
    w2[3, 11], b2[3] = 1, -2

    # For z in [-2.85, 2.85], reflection is z-2R(z-.95)+2R(-z-.95).
    # The opponent ramps differ by 0 below center and 2 above center.
    # Their coefficient steers the return toward the opposite court endpoint.
    w3 = np.array([[aim_scale, -1, -4 * aim_scale, -aim_offset],
                   [1, 0, -4, 0], [-1, 0, 4, 0]])
    b3 = np.array([17, -.95, -.95])
    down = np.array([1, -2 * aim_scale, 2 * aim_scale])
    w4 = np.stack([-down, np.zeros(3), down])
    b4 = np.array([6 + aim_offset, 1.1 / 80, -6 - aim_offset])
    arrays = [(w1, b1), (w2, b2), (w3, b3), (w4, b4)]
    assert sum(w.size + b.size for w, b in arrays) == 163
    return [(w.astype(np.float32), b.astype(np.float32)) for w, b in arrays]


def export(path):
    nodes, tensors, previous = [], [], 'observation'
    for i, (weight, bias) in enumerate(weights()):
        final = i == 3
        output = 'action_scores' if final else f'linear{i}'
        tensors.extend([numpy_helper.from_array(weight, f'w{i}'),
                        numpy_helper.from_array(bias, f'b{i}')])
        nodes.append(helper.make_node('Gemm', [previous, f'w{i}', f'b{i}'], [output], transB=1))
        previous = output
        if not final:
            previous = f'hidden{i}'
            nodes.append(helper.make_node('Relu', [output], [previous]))
    graph = helper.make_graph(nodes, 'Quill Nano v4: handwritten 163 parameters',
                              [helper.make_tensor_value_info('observation', 1, [1, 6])],
                              [helper.make_tensor_value_info('action_scores', 1, [1, 3])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    helper.set_model_props(model, dict(rules='pong-v4', architecture='6-12-4-3-3 ReLU',
        method='Handwritten formulas; manually revised on development matches; no training or automated fitting',
        reference='Vector Nano physical prior; no copied learned tensors',
        limitations='Cap-speed approximation; one-wall forecast; near-center recovery; weak against adaptive scripts'))
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh path; preserve frozen model evidence.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export(args.out)
    print(f'{args.out}: {hashlib.sha256(args.out.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
