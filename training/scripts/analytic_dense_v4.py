"""Shared dense ONNX serializer for the Vector Nano and Prism Small constructors."""
import onnx
from torch import nn
from onnx import helper, numpy_helper


def export(net,path):
 nodes=[];weights=[];current='observation'
 for i,layer in enumerate(net):
  output='action_scores' if i==len(net)-1 else f'h{i}'
  if isinstance(layer,nn.Linear):
   weights.extend([numpy_helper.from_array(layer.weight.detach().numpy(),f'w{i}'),numpy_helper.from_array(layer.bias.detach().numpy(),f'b{i}')])
   nodes.append(helper.make_node('Gemm',[current,f'w{i}',f'b{i}'],[output],transB=1))
  else:nodes.append(helper.make_node('Relu',[current],[output]))
  current=output
 m=helper.make_model(helper.make_graph(nodes,'Analytic dense cap-speed approximation',[helper.make_tensor_value_info('observation',1,[1,6])],[helper.make_tensor_value_info('action_scores',1,[1,3])],weights),opset_imports=[helper.make_opsetid('',18)],ir_version=10)
 helper.set_model_props(m,dict(rules='pong-v4',architecture='6-29-9-3 ReLU',method='Transparent analytic dense feature approximation; no training',limitation='Uses cap13 flight approximation and6unit opponent-side aim; not exact physics or learned-source transfer'))
 onnx.checker.check_model(m);onnx.save(m,path)
