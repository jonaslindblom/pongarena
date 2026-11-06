from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]


class OnnxPolicy:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])

    def __call__(self, obs):
        # The submission graph fixes batch=1; evaluate larger batches per row.
        return np.array([self.session.run(None, {'observation': row[None]})[0][0].argmax() for row in obs])


def builtin_path(name):
    catalog = json.loads((ROOT/'training/catalog.json').read_text())
    return ROOT/next(model['file'] for model in catalog['models'] if model['id'] == name)


def load_builtin(name):
    """Load a cataloged plain dense policy, including archived training references."""
    model = onnx.load(builtin_path(name))
    weights = {w.name: torch.from_numpy(numpy_helper.to_array(w).copy()) for w in model.graph.initializer}
    layers = []
    for node in model.graph.node:
        if node.op_type == 'Gemm':
            trans = next((a.i for a in node.attribute if a.name == 'transB'), 0)
            weight = weights[node.input[1]] if trans else weights[node.input[1]].T
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                layer.weight.copy_(weight)
                layer.bias.copy_(weights[node.input[2]])
            layers.append(layer)
        elif node.op_type in ('Relu', 'Tanh'):
            layers.append(nn.ReLU() if node.op_type == 'Relu' else nn.Tanh())
        else:
            raise ValueError(f'Expected a current dense policy, got {node.op_type}')
    return nn.Sequential(*layers).eval()


def actions(actor, obs):
    with torch.no_grad():
        return actor(torch.from_numpy(obs)).argmax(-1).numpy()


def export(actor, path):
    """Directly serialize these five supported layers to keep opset18/IR10 portable.

    PyTorch training owns the weights; explicit ONNX construction avoids exporter
    rewrites or auxiliary ops that fall outside the deliberately small arena contract.
    """
    nodes, weights = [], []
    current = 'observation'
    for i, layer in enumerate(actor):
        output = 'action_scores' if i == len(actor) - 1 else f'h{i}'
        if isinstance(layer, nn.Linear):
            weights.extend([numpy_helper.from_array(layer.weight.detach().numpy(), f'w{i}'),
                            numpy_helper.from_array(layer.bias.detach().numpy(), f'b{i}')])
            nodes.append(helper.make_node('Gemm', [current, f'w{i}', f'b{i}'], [output], transB=1))
        elif isinstance(layer, nn.Tanh):
            nodes.append(helper.make_node('Tanh', [current], [output]))
        else:
            raise ValueError(f'Unsupported layer: {layer}')
        current = output
    graph = helper.make_graph(nodes, 'PongArena dense actor',
                              [helper.make_tensor_value_info('observation', 1, [1, 6])],
                              [helper.make_tensor_value_info('action_scores', 1, [1, 3])], weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    linear = [layer for layer in actor if isinstance(layer, nn.Linear)]
    widths = [linear[0].in_features] + [layer.out_features for layer in linear]
    helper.set_model_props(model, {'rules': 'pong-v4', 'architecture': '-'.join(map(str, widths)) + ' Tanh', 'training': 'PyTorch'})
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
