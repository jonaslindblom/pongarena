"""Analytic bidirectional flight banks plus minimax recovery, without training."""
import argparse
from pathlib import Path
import numpy as np

from precision_variants import weights as forecast_weights
import onnx
from onnx import TensorProto, helper, numpy_helper


def recovery(height):
    """Center the range of legal next returns at maximum horizontal speed.

    Enumerate possible integer outgoing velocities from the game's contact
    formula, not opponent data or outcomes. Near a wall, legal paddle centers
    prevent the opponent from using all offsets, narrowing its return range.
    """
    low = max(-15, height - 145)
    high = min(15, height - 15)
    velocities = np.arange(np.floor(.6 * low + .5), np.floor(.6 * high + .5) + 1)
    z = (height + 26 * velocities - 4) % 304
    landings = 4 + np.minimum(z, 304 - z)
    return np.clip((landings.min() + landings.max()) / 2, 15, 145)


def weights(*, exact_ticks=True, step=.25, max_speed_only=False):
    base = forecast_weights(step=step, flight_bias=0 if exact_ticks else .5,
                            exact_ticks=exact_ticks)
    (a, ab), (b, bb), (c, cb), (d, db), (e, eb) = base
    knots = np.arange(-1.5, 3.5, step)
    bank_width = 2 * len(knots)
    bank_total = 9 * bank_width
    # Keep all incoming columns stable; append outgoing speed-distance gates
    # and an incoming detector to suppress the outgoing height carrier.
    w1, b1 = np.zeros((43, 6), np.float32), np.zeros(43, np.float32)
    w1[:24], b1[:24] = a, ab
    for j, speed in enumerate(range(5, 14)):
        w1[24 + 2 * j, 2], b1[24 + 2 * j] = 1, -speed / 6
        w1[25 + 2 * j, 2], b1[25 + 2 * j] = -1, speed / 6
    w1[42, 2] = -1
    outgoing = len(bb)
    carry = outgoing + bank_total
    w2, b2 = np.zeros((carry + 3, 43), np.float32), np.zeros(carry + 3, np.float32)
    w2[:outgoing, :24], b2[:outgoing] = b, bb
    for j, speed in enumerate(range(5, 14)):
        correction = (np.ceil(330 / speed) - 330 / speed) * speed / 180 if exact_ticks else 0
        for sign_index, sign in enumerate((1, -1)):
            for k, knot in enumerate(knots):
                i = outgoing + j * bank_width + sign_index * len(knots) + k
                w2[i, 18], w2[i, 19] = -1, sign
                w2[i, 24 + 2 * j:26 + 2 * j] = -64
                b2[i] = 11 / 6 - 2 * sign - knot + correction
    w2[carry, 20], w2[carry, 42] = 1, -64
    w2[carry + 1, 19], w2[carry + 1, 42] = 1, -64
    w2[carry + 2, 42], b2[carry + 2] = -2, 1
    w3, b3 = np.zeros((6, carry + 3), np.float32), np.zeros(6, np.float32)
    w3[:5, :outgoing], b3[:5] = c, cb
    w3[5, outgoing:carry] = c[0, :bank_total]
    w3[5, carry:carry + 3] = c[0, bank_total:bank_total + 2].tolist() + [c[0, bank_total + 5]]
    b3[5] = cb[0]
    if max_speed_only:
        # At lower speeds the paddle has enough return-flight time to cover
        # the court from center. Restrict asymmetric recovery to speed 13.
        w3[5, outgoing:outgoing + 8 * bank_width] = 0
        for row in (carry, carry + 1):
            w2[row, 42] = 0
            w2[row, 40:42] = -128
        w2[carry + 2, 42] = 0
        w2[carry + 2, 40:42] = -6

    # Compose the triangular wall reflection with the minimax recovery map at
    # one-unit intervals. Compile its piecewise-linear interpolation to hinges.
    grid = np.arange(-720, 721, dtype=np.float64) / 80
    heights = 4 + np.minimum((grid * 80 + 76) % 304, 304 - (grid * 80 + 76) % 304)
    values = np.array([(recovery(h) - 80) / 80 for h in heights])
    slopes = np.diff(values) / np.diff(grid)
    deltas = np.diff(slopes)
    changes = np.flatnonzero(np.abs(deltas) > 1e-7)
    w4, b4 = np.zeros((len(db) + len(changes), 6), np.float32), np.zeros(len(db) + len(changes), np.float32)
    w4[:len(db), :5], b4[:len(db)] = d, db
    w4[0, 5] = slopes[0]
    b4[0] += values[0] - slopes[0] * (16 + grid[0])
    direction = np.zeros(len(b4), np.float32)
    direction[:len(db)] = e[2]
    for i, change in enumerate(changes, len(db)):
        w4[i, 5], b4[i] = 1, -16 - grid[change + 1]
        direction[i] = deltas[change]
    w5 = np.stack([-direction, np.zeros(len(direction), np.float32), direction])
    return [(w1, b1), (w2, b2), (w3, b3), (w4, b4), (w5, eb)]


def export(path, **settings):
    layers = weights(**settings)
    nodes, tensors, previous = [], [], 'observation'
    for i, (weight, bias) in enumerate(layers):
        output = 'action_scores' if i == len(layers) - 1 else f'linear_{i}'
        tensors.extend([numpy_helper.from_array(weight, f'w{i}'), numpy_helper.from_array(bias, f'b{i}')])
        nodes.append(helper.make_node('Gemm', [previous, f'w{i}', f'b{i}'], [output], transB=1))
        previous = output
        if i < len(layers) - 1:
            previous = f'hidden_{i}'
            nodes.append(helper.make_node('Relu', [output], [previous]))
    graph = helper.make_graph(nodes, 'QuillRecovery',
        [helper.make_tensor_value_info('observation', TensorProto.FLOAT, [1, 6])],
        [helper.make_tensor_value_info('action_scores', TensorProto.FLOAT, [1, 3])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10, producer_name='PongArena')
    helper.set_model_props(model, {'method': 'Handwritten speed banks and minimax geometry; no training or fitting'})
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())
    return sum(w.size + b.size for w, b in layers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('scratch/quill-experiments/recovery.onnx'))
    args = parser.parse_args()
    print(f'{args.out}: {export(args.out)} parameters')
