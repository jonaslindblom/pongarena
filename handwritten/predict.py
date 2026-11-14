"""Build Predict's 75 handwritten parameters. No training or checkpoints required."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def weights():
    """Return three dense layers implementing a compact forecast and wall fold."""
    # d = ball_x_input + 11/12 is the along-court distance to our contact plane,
    # divided by 180. Input suffixes denote normalized model observations;
    # ball_vy_input is the across-court velocity in court units per tick / 6.
    # q(t) = |t| + 3*ReLU(t-1) approximates a square. The difference
    # q(d+a)-q(d-a) approximates 4*d*a, with a=(2/3)*ball_vy_input.
    product_scale = (6 * 180 / (13 * 80)) / (4 * (2 / 3))
    w1, b1 = np.zeros((6, 6), np.float32), np.zeros(6, np.float32)
    for j, sign in enumerate((1, -1)):
        for i, knot in enumerate((0, 1)):
            row = 2 * j + i
            w1[row, 0], w1[row, 3] = 1, sign * 2 / 3
            b1[row] = 11 / 12 - knot
    w1[4, 1] = 1
    w1[4, 3] = 6 * (8 / 13) / 80 - 2 * (2 / 3) * product_scale
    b1[4] = 6  # Carry signed across-court forecast terms through ReLU.
    w1[5, 4], b1[5] = 1, 1
    forecast = np.zeros(6, np.float32)
    forecast[:2] = np.asarray([2, 3]) * product_scale
    forecast[2:4] = -forecast[:2]
    forecast[4] = 1

    # z = forecast dot hidden_0 - 6 is the unfolded across-court prediction,
    # centered at 80 and divided by 80. Carry z-own_input+6 and the two hinges
    # for reflection at normalized across-court positions +/-0.95.
    w2 = np.stack([forecast, forecast, -forecast])
    w2[0, 5] = -1
    b2 = np.array([1, -6 - .95, 6 - .95], np.float32)
    direction = np.array([1, -2, 2], np.float32)
    w3 = np.stack([-direction, np.zeros(3, np.float32), direction])
    b3 = np.array([6, 1.1 / 80, -6], np.float32)
    return [(w1, b1), (w2, b2), (w3, b3)]


def export(path):
    """Reproduce the frozen exploration artifact, including its original identity."""
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
    # Preserve the original graph name and metadata from the frozen experiment.
    # The arena's display name and submitter live in the model roster.
    graph = helper.make_graph(nodes, 'Predictive_ungated2',
        [helper.make_tensor_value_info('observation', TensorProto.FLOAT, [1, 6])],
        [helper.make_tensor_value_info('action_scores', TensorProto.FLOAT, [1, 3])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)],
                              ir_version=10, producer_name='PongArena exploration')
    helper.set_model_props(model, {'method': 'Handwritten analytical weights; no fitting',
                                   'variant': 'ungated2'})
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('public/models/predict.onnx'))
    args = parser.parse_args()
    export(args.out)
    print(f'{args.out}: {hashlib.sha256(args.out.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
