"""Build Quill Speed's 229 analytical parameters without training or fitting.

The original Quill remains unchanged. Six extra ReLU hinges add a bounded
slow-flight lead to its forecast; reflection, aiming and recovery are retained.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from quill import weights as quill_weights


def weights():
    """Write four dense layers with a fixed, bounded 20-tick correction.

    On approach, let d=(x-15)/180, v=vy/6, w=13-|vx|, and a=20*6/80.
    F(t,v)=ReLU(t+a*v)-ReLU(t-a*v)-a*v equals
    sign(v)*min(t,a*abs(v)) for t>=0. The added forecast lead is

        F(d,v) + F(.04*w,v) - F(d+.04*w,v).

    It is zero at maximum horizontal speed, zero vertical speed, or contact.
    Its magnitude is bounded by 20*abs(vy), 80*d, and 3.2*(13-|vx|)
    in court units. This is a geometric heuristic, not exact time division.
    """
    original = quill_weights()
    old_w1, old_b1 = original[0]
    old_w2, old_b2 = original[1]
    w1, b1 = np.zeros((18, 6), np.float32), np.zeros(18, np.float32)
    w2 = np.zeros((4, 18), np.float32)
    w1[:12], b1[:12] = old_w1, old_b1
    w2[:, :12] = old_w2
    a, speed_scale = 20 * 6 / 80, .04
    for pair, (distance, speed, sign) in enumerate(((1, 0, 1), (0, 1, 1), (1, 1, -1))):
        for half, vertical_sign in enumerate((1, -1)):
            row = 12 + 2 * pair + half
            w1[row, 0] = distance
            w1[row, 2] = speed * speed_scale * 6
            w1[row, 3] = vertical_sign * a
            b1[row] = distance * 11 / 12 + speed * speed_scale * 13
            w2[0, row] = sign * vertical_sign
    # Carry the net signed-velocity term with a positive offset, then remove
    # that offset through Quill's existing incoming-direction indicator.
    w1[8, 3] -= a
    b1[8] = 10
    tail = [(weight.copy(), bias.copy()) for weight, bias in original[2:]]
    tail[0][0][:, 2] = [-10 * (83 / 78), -10, 10]
    return [(w1, b1), (w2, old_b2.copy()), *tail]


def export(path):
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
    graph = helper.make_graph(nodes, 'QuillSpeed',
        [helper.make_tensor_value_info('observation', TensorProto.FLOAT, [1, 6])],
        [helper.make_tensor_value_info('action_scores', TensorProto.FLOAT, [1, 3])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)],
                              ir_version=10, producer_name='PongArena')
    helper.set_model_props(model, {
        'method': 'Handwritten formulas and bounded speed correction; no training or automated fitting'})
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('public/models/quill_speed.onnx'))
    args = parser.parse_args()
    export(args.out)
    print(f'{args.out}: {hashlib.sha256(args.out.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
