"""Finite, analytical Quill hypotheses; no training or fitting.

This file leaves Quill's original constructor and model untouched. Three Small
variants retain its recovery, reflection and aiming layers. Two speed hypotheses add
a bounded correction which is exactly zero at speed 13, vertical speed zero,
and distance zero. Their only differing assumption is a 20/40 tick lead cap.
"""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'handwritten'))
from quill import weights as quill_weights


def dense_square_weights():
    """251 parameters: interpolate t² on a uniform half-unit grid.

    q(t) = -2.5t + sum(ReLU(t-k)), k=-1,-.5,...,2.5, up to a
    constant which cancels in q(d+a)-q(d-a). On the legal argument range
    [-1,17/6], this is the linear interpolation of t², with error <=1/16.
    Dividing q(d+a)-q(d-a) by 4a approximates d*vertical velocity.
    """
    old = quill_weights()
    old_w1, old_b1 = old[0]
    old_w2, old_b2 = old[1]
    knots = np.arange(-1, 3, .5)
    scale, a = (6 * 180 / (13 * 80)) / (4 * (2 / 3)), 2 / 3
    w1, b1 = np.zeros((20, 6), np.float32), np.zeros(20, np.float32)
    w2 = np.zeros((4, 20), np.float32)
    for j, sign in enumerate((1, -1)):
        for i, knot in enumerate(knots):
            row = j * len(knots) + i
            w1[row, 0], w1[row, 3], b1[row] = 1, sign * a, 11 / 12 - knot
            w2[0, row] = sign * scale
    w1[16:], b1[16:] = old_w1[8:], old_b1[8:]
    # q's linear term contributes -2.5*(2*a*v) to the difference.
    w1[16, 3] = 6 * (8 / 13) / 80 - 2.5 * 2 * a * scale
    w2[:, 16:] = old_w2[:, 8:]
    return [(w1, b1), (w2, old_b2.copy()), *[(w.copy(), b.copy()) for w, b in old[2:]]]


def speed_weights(extra_ticks):
    """229 parameters: six ReLU hinges add a bounded slow-flight lead.

    d=(x-15)/180, v=vy/6, w=13+6*vx_input=13-|vx| on approach.
    Let F(t,v)=ReLU(t+a*v)-ReLU(t-a*v)-a*v. For t>=0 this
    equals sign(v)*min(t,a*abs(v)). Add

      C = F(d,v) + F(.04*w,v) - F(d+.04*w,v)

    to the normalized forecast. a=extra_ticks*6/80 makes the correction
    at most extra_ticks*abs(vy) in court units. It is also bounded by
    80*d and 3.2*(13-|vx|), and disappears continuously near contact.
    This is a heuristic extra-flight allowance, not exact division by vx.
    The constants are specified analytically before match evaluation.
    """
    old = quill_weights()
    old_w1, old_b1 = old[0]
    old_w2, old_b2 = old[1]
    w1, b1 = np.zeros((18, 6), np.float32), np.zeros(18, np.float32)
    w2 = np.zeros((4, 18), np.float32)
    w1[:12], b1[:12] = old_w1, old_b1
    w2[:, :12] = old_w2
    a, b = extra_ticks * 6 / 80, .04
    # Each pair supplies ReLU(t+a*v)-ReLU(t-a*v). The three F terms
    # leave a net -a*v, which travels through Quill's signed height carrier.
    for pair, (distance, speed, sign) in enumerate(((1, 0, 1), (0, 1, 1), (1, 1, -1))):
        for half, vertical_sign in enumerate((1, -1)):
            row = 12 + 2 * pair + half
            w1[row, 0], w1[row, 2], w1[row, 3] = distance, speed * b * 6, vertical_sign * a
            b1[row] = distance * 11 / 12 + speed * b * 13
            w2[0, row] = sign * vertical_sign
    w1[8, 3] -= a
    # The additional signed velocity term needs a larger positive carrier
    # offset so ReLU cannot clip it. Remove that offset only on approach,
    # following Quill's existing direction gate.
    b1[8] = 10
    tail = [(w.copy(), bias.copy()) for w, bias in old[2:]]
    tail[0][0][:, 2] = [-10 * (83 / 78), -10, 10]
    return [(w1, b1), (w2, old_b2.copy()), *tail]


def recovery_weights(base='speed20'):
    """Add a five-Gemm analytical recovery branch (409 or 435 parameters).

    An opponent contacting near a wall cannot choose every paddle offset:
    its center is restricted to [15,145]. At maximum horizontal speed, the
    next return flies 26 ticks and its vertical velocity is about .6*offset.
    Reflecting this continuous range gives a minimax recovery center R(h).
    It differs from 80 only for opponent contact heights below 14.34 or
    above 145.66. The six hinges below encode that geometry exactly for
    the continuous approximation; integer rounding remains an approximation.
    """
    old = speed_weights(20) if base == 'speed20' else dense_square_weights()
    (old_w1, old_b1), (old_w2, old_b2), (old_w3, old_b3), (old_w4, old_b4) = old
    n1 = len(old_b1)
    w1, b1 = np.zeros((n1 + 2, 6), np.float32), np.zeros(n1 + 2, np.float32)
    w1[:n1], b1[:n1] = old_w1, old_b1
    # Reconstruct the opposite contact forecast by changing the sign of the
    # existing distance*vertical-velocity product, then adding full-court lead.
    product_count = 8 if base == 'speed20' else 16
    timing = 6 * (8 / 13) / 80
    original_coefficient = quill_weights()[0][0][8, 3] if base == 'speed20' else old_w1[16, 3]
    w1[n1, 1], w1[n1, 3], b1[n1] = 1, 330 * 6 / (13 * 80) + 2 * timing - original_coefficient, 6
    w1[n1 + 1, 2] = -1  # incoming detector
    w2, b2 = np.zeros((6, n1 + 2), np.float32), np.zeros(6, np.float32)
    w2[:4, :n1], b2[:4] = old_w2, old_b2
    w2[4, :product_count] = -old_w2[0, :product_count]
    w2[4, n1], w2[4, n1 + 1] = 1, -20
    w2[5, n1 + 1], b2[5] = -2, 1  # outgoing indicator
    w3, b3 = np.zeros((8, 6), np.float32), np.zeros(8, np.float32)
    w3[:3, :4], b3[:3] = old_w3, old_b3
    # Opponent forecast, normalized and reflected at +/- .95. Its three
    # activations reconstruct the reflected forecast plus 4. Incoming states
    # yield zero forecast and hence the ordinary center recovery target.
    w3[3, 4], w3[3, 5], b3[3] = 1, -6, 4
    w3[4, 4], w3[4, 5], b3[4] = 1, -6, -.95
    w3[5, 4], w3[5, 5], b3[5] = -1, 6, -.95
    w3[6, 4], w3[6, 5], b3[6] = 1, -6, -2.85
    w3[7, 4], w3[7, 5], b3[7] = -1, 6, -2.85
    # For h in [4,156], R(h)-80 = f(h)-f(160-h), where
    # f(h)=8.3*(hC-h)+ -8.8*(hB-h)+ -8.3*(hA-h)+.
    hA, hB, hC = 86 / 16.6, 172 / 17.6, 119 / 8.3
    w4, b4 = np.zeros((7, 8), np.float32), np.zeros(7, np.float32)
    w4[0, :3], b4[0] = old_w4[2], old_b4[2] + 4
    reflected = np.array([1, -2, 2, 2, -2], np.float32)
    for i, height in enumerate((hC, hB, hA)):
        # h = 80*(reflected_normalized+1). Divide hinges by 80
        # so their final coefficients directly produce normalized offsets.
        w4[1 + i, 3:], b4[1 + i] = -reflected, height / 80 + 3
        w4[4 + i, 3:], b4[4 + i] = reflected, height / 80 - 5
    direction = np.array([1, 8.3, -8.8, -8.3, -8.3, 8.8, 8.3], np.float32)
    w5 = np.stack([-direction, np.zeros(7, np.float32), direction])
    b5 = np.array([4, 1.1 / 80, -4], np.float32)
    return [(w1, b1), (w2, b2), (w3, b3), (w4, b4), (w5, b5)]


def variants():
    return {
        'quill-square251': dense_square_weights(),
        'quill-speed20': speed_weights(20),
        'quill-speed40': speed_weights(40),
        'quill-speed-recover': recovery_weights('speed20'),
        'quill-square-recover': recovery_weights('square251'),
    }


def export(path, variant='quill-speed20'):
    layers = variants()[variant]
    nodes, tensors, previous = [], [], 'observation'
    for i, (weight, bias) in enumerate(layers):
        final = i == len(layers) - 1
        output = 'action_scores' if final else f'linear_{i}'
        tensors.extend([numpy_helper.from_array(weight, f'w{i}'), numpy_helper.from_array(bias, f'b{i}')])
        nodes.append(helper.make_node('Gemm', [previous, f'w{i}', f'b{i}'], [output], transB=1))
        previous = output
        if not final:
            previous = f'hidden_{i}'
            nodes.append(helper.make_node('Relu', [output], [previous]))
    graph = helper.make_graph(nodes, variant,
        [helper.make_tensor_value_info('observation', TensorProto.FLOAT, [1, 6])],
        [helper.make_tensor_value_info('action_scores', TensorProto.FLOAT, [1, 3])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10,
                              producer_name='PongArena handwritten experiments')
    helper.set_model_props(model, {'method': 'Analytical handwritten weights; no training or automated fitting',
                                  'variant': variant})
    onnx.checker.check_model(model)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(model.SerializeToString())
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=list(variants()), default='quill-speed20')
    parser.add_argument('--out', type=Path, default=Path('/tmp/quill-speed20.onnx'))
    args = parser.parse_args()
    export(args.out, args.variant)
    layers = variants()[args.variant]
    print(f'{args.out}: {sum(w.size + b.size for w, b in layers)} parameters; '
          f'{hashlib.sha256(args.out.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
