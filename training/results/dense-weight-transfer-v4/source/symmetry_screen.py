exec(open('/tmp/rally-dense-root/edit_screen.py').read().split('long=')[0])
from torch import nn
from onnx import helper,numpy_helper
import onnx
source=policies['challenger']
net=nn.Sequential(nn.Linear(6,32),nn.ReLU(),nn.Linear(32,3)).eval()
sign=torch.tensor([1,-1,1,-1,-1,-1],dtype=torch.float32)
with torch.no_grad():
 net[0].weight[:16].copy_(source[0].weight);net[0].weight[16:].copy_(source[0].weight*sign)
 net[0].bias[:16].copy_(source[0].bias);net[0].bias[16:].copy_(source[0].bias)
 net[2].weight[:,:16].copy_(source[2].weight*.5);net[2].weight[:,16:].copy_(source[2].weight.flip(0)*.5)
 net[2].bias.copy_((source[2].bias+source[2].bias.flip(0))*.5)
result={'purpose':'Fixed6-32-3ReLU dense NN; transfer Challenger feature rows plus reflected row signs and action votes; no added graphops or training','seeds':seeds,'pairs':{}}
for name,opp in policies.items():
 row=run_pair(CONFIGS['pace13-v9-scaled'],net,opp,seeds);result['pairs'][name]=row
 Path('/tmp/rally-dense-root/symmetry-screen.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
 s=row['summary'];print(name,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
torch.save(net.state_dict(),'/tmp/rally-dense-root/symmetric-challenger.pt')
weights=[numpy_helper.from_array(net[i].weight.detach().numpy(),f'w{i}') for i in [0,2]]+[numpy_helper.from_array(net[i].bias.detach().numpy(),f'b{i}') for i in [0,2]]
nodes=[helper.make_node('Gemm',['observation','w0','b0'],['linear'],transB=1),helper.make_node('Relu',['linear'],['hidden']),helper.make_node('Gemm',['hidden','w2','b2'],['action_scores'],transB=1)]
m=helper.make_model(helper.make_graph(nodes,'Dense Challenger weight symmetry transfer',[helper.make_tensor_value_info('observation',1,[1,6])],[helper.make_tensor_value_info('action_scores',1,[1,3])],weights),opset_imports=[helper.make_opsetid('',18)],ir_version=10)
onnx.save(m,'/tmp/rally-dense-root/symmetric-challenger.onnx')
