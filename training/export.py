"""Write a portable dense ONNX graph from explicit affine-layer arrays."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def export_weights(weights, path, activation='Tanh', name='Pong model', method='Trained with PyTorch'):
    nodes, tensors = [], []
    previous = 'observation'
    for i, (weight, bias) in enumerate(weights):
        final = i == len(weights) - 1
        output = 'action_scores' if final else f'linear_{i}'
        tensors.extend([numpy_helper.from_array(np.asarray(weight, dtype=np.float32), f'w{i}'),
                        numpy_helper.from_array(np.asarray(bias, dtype=np.float32), f'b{i}')])
        nodes.append(helper.make_node('Gemm', [previous, f'w{i}', f'b{i}'], [output], transB=1))
        previous = output
        if not final:
            previous = f'hidden_{i}'
            nodes.append(helper.make_node(activation, [output], [previous]))
    graph = helper.make_graph(nodes, name,
        [helper.make_tensor_value_info('observation', TensorProto.FLOAT, [1, 6])],
        [helper.make_tensor_value_info('action_scores', TensorProto.FLOAT, [1, 3])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10,
                              producer_name='PongArena')
    helper.set_model_props(model, {'method': method})
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())


def export_model(policy, path, activation='Tanh', name='Pong model'):
    weights = [(layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy())
               for layer in policy if hasattr(layer, 'weight')]
    export_weights(weights, path, activation, name, 'PyTorch imitation of geometric scripts; random initialization')
