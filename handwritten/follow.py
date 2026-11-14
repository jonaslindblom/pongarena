"""Build Follow's 13 handwritten parameters. No training or checkpoints required."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def weights():
    """Track the ball's current position, with a 1.5-unit movement deadband."""
    # These ball and paddle inputs are across-court positions, centered at 80
    # and divided by 80. Positive ball-minus-paddle error means move right.
    w0 = np.array([[0, 1, 0, 0, -1, 0]], dtype=np.float32)
    b0 = np.zeros(1, np.float32)
    w1 = np.array([[-1], [0], [1]], dtype=np.float32)
    b1 = np.array([0, np.tanh(1.5 / 80), 0], dtype=np.float32)
    return [(w0, b0), (w1, b1)]


def export(path):
    """Serialize the fixed dense network; no position helper runs during play."""
    tensors = []
    for i, (weight, bias) in enumerate(weights()):
        tensors.extend([numpy_helper.from_array(weight, f'w{i}'),
                        numpy_helper.from_array(bias, f'b{i}')])
    nodes = [
        helper.make_node('Gemm', ['observation', 'w0', 'b0'], ['linear_0'], transB=1),
        helper.make_node('Tanh', ['linear_0'], ['hidden_0']),
        helper.make_node('Gemm', ['hidden_0', 'w1', 'b1'], ['action_scores'], transB=1),
    ]
    graph = helper.make_graph(nodes, 'Follow',
        [helper.make_tensor_value_info('observation', TensorProto.FLOAT, [1, 6])],
        [helper.make_tensor_value_info('action_scores', TensorProto.FLOAT, [1, 3])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10,
                              producer_name='PongArena')
    helper.set_model_props(model, {'method': 'Handwritten position tracking; no training or automated fitting'})
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('public/models/follow.onnx'))
    args = parser.parse_args()
    export(args.out)
    print(f'{args.out}: {hashlib.sha256(args.out.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
