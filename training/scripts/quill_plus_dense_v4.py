"""Quill Plus: four handwritten changes, still 163 parameters and 141 MACs.

Extend Quill's final square segment over the legal long-flight range, and
reduce the opponent-dependent recovery offset as horizontal speed increases.
The offset vanishes at cap speed. No training, fitting or automated weight search.
The original Quill constructor and deployed model remain unchanged.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from quill_dense_v4 import weights as original_weights


def weights():
    arrays = original_weights()
    w2 = arrays[1][0]

    # u = distance/180 + (2/3)*(vy/6) can reach 330/180 + 1 = 17/6.
    # The old final square segment starts at 4/3 and assumes an endpoint of 2.
    # For span L, h=2L(sqrt(2)-1) balances the largest positive and negative
    # secant errors. This preserves every forecast feature near contact.
    left, maximum = 4 / 3, 17 / 6
    endpoint = left + 2 * (maximum - left) * (np.sqrt(2) - 1)
    product_scale = 6 * 180 / (13 * 80) / (4 * 2 / 3)
    w2[0, 3] = (left + endpoint - 2) * product_scale
    w2[0, 7] = -w2[0, 3]

    # On departure, the two opponent ramps now leave a recovery offset
    # proportional to (1 - speed/13). At legal integer opponent positions,
    # both opponent sides recenter exactly at cap speed. Incoming aiming is
    # unchanged because the positive-horizontal-velocity neuron is zero.
    k, aim_offset = 12 / 13, 19 / 312
    w2[1, 10] = -k * aim_offset / 2
    w2[3, 10] = k
    assert sum(w.size + b.size for w, b in arrays) == 163
    return arrays


def export(path):
    nodes, tensors, previous = [], [], 'observation'
    for i, (weight, bias) in enumerate(weights()):
        output = 'action_scores' if i == 3 else f'linear{i}'
        tensors.extend([numpy_helper.from_array(weight, f'w{i}'),
                        numpy_helper.from_array(bias, f'b{i}')])
        nodes.append(helper.make_node('Gemm', [previous, f'w{i}', f'b{i}'], [output], transB=1))
        previous = output
        if i != 3:
            previous = f'hidden{i}'
            nodes.append(helper.make_node('Relu', [output], [previous]))
    graph = helper.make_graph(nodes, 'Quill Plus v4: four handwritten refinements',
                              [helper.make_tensor_value_info('observation', 1, [1, 6])],
                              [helper.make_tensor_value_info('action_scores', 1, [1, 3])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    helper.set_model_props(model, dict(rules='pong-v4', architecture='6-12-4-3-3 ReLU',
        method='Four handwritten changes to Quill; manual development evaluation; no training or automated fitting',
        changes='Long-flight square tail; speed-dependent recentering',
        limitations='Cap-speed forecast; one-wall approximation; weak against adaptive scripts'))
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
