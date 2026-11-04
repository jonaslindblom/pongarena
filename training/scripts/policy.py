from pathlib import Path
import numpy as np
import torch
from torch import nn
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]


class Actor(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))
        for layer in self:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self[-1].weight, gain=.01)


class OnnxPolicy:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])

    def __call__(self, obs):
        # The submission graph fixes batch=1; evaluate larger batches per row.
        return np.array([self.session.run(None, {'observation': row[None]})[0][0].argmax() for row in obs])


def load_builtin(name):
    """Load the supplied 6→16 ReLU→3 ONNX into Torch for batched training."""
    m = onnx.load(ROOT / 'public/models' / (name + '.onnx'))
    weights = {w.name: torch.from_numpy(numpy_helper.to_array(w).copy()) for w in m.graph.initializer}
    net = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 3))
    with torch.no_grad():
        for layer, node in zip([net[0], net[2]], [n for n in m.graph.node if n.op_type == 'Gemm']):
            trans = next((a.i for a in node.attribute if a.name == 'transB'), 0)
            layer.weight.copy_(weights[node.input[1]] if trans else weights[node.input[1]].T)
            layer.bias.copy_(weights[node.input[2]])
    return net.eval()


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
    graph = helper.make_graph(nodes, 'Rally Lab Pong v2 actor',
                              [helper.make_tensor_value_info('observation', 1, [1, 6])],
                              [helper.make_tensor_value_info('action_scores', 1, [1, 3])], weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    helper.set_model_props(model, {'rules': 'pong-v2', 'architecture': '6-64-64-3 Tanh', 'training': 'PyTorch'})
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
