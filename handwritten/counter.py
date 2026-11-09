"""Build Counter's 97 handwritten parameters. No training or fitting required."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def weights():
    """Predict, select a corner and recover through four ordinary dense layers."""
    aim_scale, aim_offset = 83 / 78, 19 / 312
    # d = (ball_x - 15)/180 and v = ball_vy/6. The two hinges give
    # sign(v)*min(d, 1.5*abs(v)) after subtracting 1.5*v. Scaling by .9
    # yields min(.4*distance, 18*abs(ball_vy)) in court units.
    w1, b1 = np.zeros((6, 6), np.float32), np.zeros(6, np.float32)
    for j, sign in enumerate((1, -1)):
        w1[j, 0], w1[j, 3], b1[j] = 1, sign * 1.5, 11 / 12
    w1[2, 1], w1[2, 3], b1[2] = 1, .75 * 6 / 80 - .9 * 1.5, 4
    w1[3, 4], b1[3] = 1, 1
    w1[4, 2] = 1  # ReLU(vx): zero on approach, positive on departure.
    # This ramp and a later hinge select a corner away from the opponent.
    # Ball height breaks near-center ties toward a cross-court return.
    w1[5, 5], w1[5, 1], b1[5] = 80, 4, 1

    # Keep forecast+4 on approach; suppress it on departure. The incoming
    # indicator removes the carry offset only while a forecast is active.
    w2, b2 = np.zeros((4, 6), np.float32), np.zeros(4, np.float32)
    w2[0, :3] = [.9, -.9, 1]
    w2[0, 4] = -20
    w2[1, 3], w2[1, 5], b2[1] = 1, -aim_offset, 10
    w2[1, 4] = -(6 / 13) * aim_offset
    w2[2, 4], b2[2] = -2, 1
    w2[3, 5], w2[3, 4], b2[3] = 1, 12 / 13, -2

    # Quill's recovery cancellation shrinks the opponent-dependent offset
    # to exactly zero at speed 13. At slower legal speeds it stays within
    # three court units of center. Incoming aiming retains Attack's law.
    # The final two layers reflect the forecast at +/- .95 and compare the
    # offset target to our paddle, with the same 1.1-unit movement deadband.
    w3 = np.array([[aim_scale, -1, -4 * aim_scale, -aim_offset],
                   [1, 0, -4, 0], [-1, 0, 4, 0]], np.float32)
    b3 = np.array([17, -.95, -.95], np.float32)
    direction = np.array([1, -2 * aim_scale, 2 * aim_scale], np.float32)
    w4 = np.stack([-direction, np.zeros(3, np.float32), direction])
    b4 = np.array([6 + aim_offset, 1.1 / 80, -6 - aim_offset], np.float32)
    return [(w1, b1), (w2, b2), (w3, b3), (w4, b4)]


def export(path):
    """Serialize fixed weights; all decisions run inside the dense network."""
    nodes, tensors, previous = [], [], 'observation'
    layers = weights()
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
    graph = helper.make_graph(nodes, 'Counter97',
        [helper.make_tensor_value_info('observation', TensorProto.FLOAT, [1, 6])],
        [helper.make_tensor_value_info('action_scores', TensorProto.FLOAT, [1, 3])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)],
                              ir_version=10, producer_name='PongArena')
    helper.set_model_props(model, {
        'method': 'Handwritten formulas with manual play evaluation; no training or automated fitting'})
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('public/models/counter.onnx'))
    args = parser.parse_args()
    export(args.out)
    print(f'{args.out}: {hashlib.sha256(args.out.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
