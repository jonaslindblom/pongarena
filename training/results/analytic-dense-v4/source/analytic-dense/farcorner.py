"""Two explained output edits; hidden analytic features remain unchanged."""
from pathlib import Path
import sys,json
sys.path.insert(0,'/tmp/rally-analytic-dense')
from build import make,export,OUT
import torch,numpy as np,onnx
sys.path.insert(0,str(Path.cwd()/'training/scripts'))
from competition_eval import load_mlp
from pace_study import CONFIGS,make_policy,run_pair
from policy import ROOT,load_builtin

def candidates():
 stable=make()
 with torch.no_grad():stable[4].bias[1]=1.101/80
 corner=make();gain=1+1/15.5;cue=65/(80*15.5)
 down=np.r_[gain*np.array([1,-2,2,-2,2]),-.95*gain-cue,-1,1,cue].astype(np.float32)
 with torch.no_grad():
  corner[4].weight[0].copy_(torch.from_numpy(-down));corner[4].weight[2].copy_(torch.from_numpy(down));corner[4].bias[1]=1.101/80
 return {'stable-cap13-safeaim':stable,'cap13-farcorner':corner}
if __name__=='__main__':
 torch.set_num_threads(1);nets=candidates();seeds=json.loads(Path('/tmp/rally-causal-failures/seeds.json').read_text())['seeds']
 policies={k:load_builtin(k) for k in ['tracker','anticipator','challenger']};policies['pacer']=load_mlp(ROOT/'training/results/pace-study/training/candidate/pacer/best.pt');policies.update({k:make_policy(CONFIGS['pace13-v9-scaled'],k) for k in ['attack','defend']})
 record={'purpose':'Two explained fixed output edits, no fitting or coefficient sweeps','definitions':{'stable-cap13-safeaim':'Only deadband1.1→1.101courtunits, numerical boundary guard','cap13-farcorner':'targetnorm=(1+1/15.5)*predictedlandingnorm+65/(80*15.5)*clippedopponentcue; desiredcorner15or145, rule-derivedangletravel approximation'},'seeds':seeds,'pairs':{}}
 for name,net in nets.items():
  torch.save(net.state_dict(),OUT/f'{name}.pt');export(net,OUT/f'{name}.onnx')
  model=onnx.load(OUT/f'{name}.onnx');props={p.key:p.value for p in model.metadata_props};props['candidate']=name;props['deadband']='1.101courtunits numericguard';props['target']=record['definitions'][name];onnx.helper.set_model_props(model,props);onnx.save(model,OUT/f'{name}.onnx')
  for other,op in policies.items():
   row=run_pair(CONFIGS['pace13-v9-scaled'],net,op,seeds);record['pairs'][name+':'+other]=row
   (OUT/'farcorner-results.json').write_text(json.dumps(record,separators=(',',':'))+'\n')
   s=row['summary'];print(name,other,s['wins'],s['losses'],s['draws'],s['point_margin_ci95'],flush=True)
