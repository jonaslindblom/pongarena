"""Build Attack's 75 handwritten parameters. No training or fitting required."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def weights():
    """Return three dense layers implementing a forecast, wall fold and aiming."""
    # Reuse Predict's layout, spending two forecast hinges on shot selection.
    # d = (ball_x - 15) / 180; v = ball_vy / 6. For d >= 0,
    # ReLU(d+v)-ReLU(d-v)-v = sign(v)*min(d, abs(v)).
    # Scale by .9: the forecast leads by at most 12 ticks, with a .75-tick
    # correction for discrete contact. All of this runs inside the network.
    forecast_scale = 12 * 6 / 80
    aim_scale, aim_offset = 83 / 78, 19 / 312
    w1, b1 = np.zeros((6, 6), np.float32), np.zeros(6, np.float32)
    for j, sign in enumerate((1, -1)):
        w1[j, 0], w1[j, 3], b1[j] = 1, sign, 11 / 12
    w1[2, 1], w1[2, 3], b1[2] = 1, .75 * 6 / 80 - forecast_scale, 6
    w1[3, 4], b1[3] = 1, 1

    # Their difference minus one clips 80*opponent + 4*ball_y to [-1, 1].
    # Aim away from the opponent; near center, favor the cross-court corner.
    w1[4, 5], w1[4, 1], b1[4] = 80, 4, 1
    w1[5, 5], w1[5, 1], b1[5] = 80, 4, -1
    forecast = np.array([forecast_scale, -forecast_scale, 1, 0, 0, 0], np.float32)

    # Fold z at +/- .95, then shift contact to send the return toward a corner.
    # At speed 13 the flight lasts 26 ticks: 26*9/15 = 78/5 units of landing
    # shift per unit of paddle offset. Hence 1+5/78 and .95*5/78 for aiming.
    w2 = np.stack([aim_scale * forecast, forecast, -forecast])
    w2[0, 3], w2[0, 4], w2[0, 5] = -1, aim_offset, -aim_offset
    b2 = np.array([7 - 6 * aim_scale - aim_offset, -6 - .95, 6 - .95], np.float32)
    direction = np.array([1, -2 * aim_scale, 2 * aim_scale], np.float32)
    w3 = np.stack([-direction, np.zeros(3, np.float32), direction])
    b3 = np.array([6, 1.1 / 80, -6], np.float32)
    return [(w1, b1), (w2, b2), (w3, b3)]


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
    graph = helper.make_graph(nodes, 'Attack75',
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
    parser.add_argument('--out', type=Path, default=Path('public/models/attack75.onnx'))
    args = parser.parse_args()
    export(args.out)
    print(f'{args.out}: {hashlib.sha256(args.out.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
